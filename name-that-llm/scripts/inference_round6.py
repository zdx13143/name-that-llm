"""Frozen R6 inference only. No training, labels from test, or network access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import threading
import time

TEXTS = [
    'The small boat reached the quiet shore before sunset.',
    'A careful writer asks: "What changed?" Then answers with three precise examples.',
    'Although the experiment is small, its records should preserve order, punctuation, and every relevant detail.',
    'This deliberately long synthetic sentence checks truncation and padding. ' * 80,
]
VERSIONS = {'numpy':'2.2.6', 'scipy':'1.16.3', 'scikit-learn':'1.7.2',
            'joblib':'1.5.2', 'threadpoolctl':'3.6.0', 'transformers':'4.56.2',
            'safetensors':'0.6.2'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def block_network():
    def denied(*args, **kwargs):
        raise RuntimeError('Network is disabled for frozen inference')
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'


def verify_assets(root):
    value = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if value['candidate'] != 'windows-calibrated-fusion-dev-20261025-r6' or value['style_weight'] != .8:
        raise ValueError('Wrong frozen candidate')
    for name, record in value['files'].items():
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Asset identity mismatch: {name}')
    return value


def verify_runtime(mode):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Expected isolated Python 3.11')
    versions = {key:importlib.metadata.version(key) for key in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError(f'Pinned package mismatch: {versions}')
    versions['torch'] = importlib.metadata.version('torch')
    expected = '2.8.0+cpu' if platform.system() == 'Linux' else '2.8.0+cu128'
    if versions['torch'] != expected or mode not in ('cpu_float32', 'cuda_bfloat16'):
        raise RuntimeError('Unexpected encoder runtime')
    return versions


def load_encoder(root, mode):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20261023)
    device = 'cuda:0' if mode == 'cuda_bfloat16' else 'cpu'
    dtype = torch.bfloat16 if mode == 'cuda_bfloat16' else torch.float32
    if device != 'cpu' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Required CUDA BF16 unavailable')
    directory = root/'encoder'
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = tokenizer.truncation_side = 'right'
    model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, torch_dtype=dtype,
                                     attn_implementation='sdpa', reference_compile=False)
    model.requires_grad_(False).eval().to(device)
    if (model.config.hidden_size != 768 or model.config.num_hidden_layers != 22 or
        model.config._attn_implementation != 'sdpa' or model.config.reference_compile is not False or
        tokenizer.pad_token_id != model.config.pad_token_id or
        tokenizer.cls_token_id != model.config.cls_token_id or
        tokenizer.sep_token_id != model.config.sep_token_id or model.training or
        any(p.requires_grad for p in model.parameters())):
        raise ValueError('Encoder configuration mismatch')
    return tokenizer, model, device


def extract(texts, encoder, before_forward):
    import numpy as np
    import torch
    tokenizer, model, device = encoder
    batch = tokenizer(texts, padding='max_length', truncation=True, max_length=256, return_tensors='pt').to(device)
    before_forward()
    with torch.inference_mode():
        result = model(**batch, output_hidden_states=True)
        mask = batch['attention_mask'].unsqueeze(-1).float()
        features = torch.cat([(result.hidden_states[k].float()*mask).sum(1)/mask.sum(1)
                              for k in (-4,-3,-2,-1)], dim=1).cpu().numpy()
    if features.shape != (len(texts),3072) or not np.isfinite(features).all():
        raise ValueError('Invalid frozen features')
    return features


def load_heads(root):
    import joblib
    import numpy as np
    style = joblib.load(root/'style.joblib')
    with np.load(root/'gaussian.npz', allow_pickle=False) as handle:
        head = {k:handle[k].copy() for k in handle.files}
    if not np.array_equal(style.classes_, np.arange(18)) or not np.array_equal(head['classes'], np.arange(18)):
        raise ValueError('Wrong class order')
    if set(head) != {'mean','scale','coefficients','intercept','classes'}:
        raise ValueError('Wrong Gaussian schema')
    if head['mean'].shape != (3072,) or head['scale'].shape != (3072,) or head['coefficients'].shape != (3072,18) or head['intercept'].shape != (18,):
        raise ValueError('Wrong Gaussian shape')
    if not all(np.isfinite(v).all() for v in head.values()) or not (head['scale'] > 0).all():
        raise ValueError('Invalid Gaussian parameters')
    calibration = json.loads((root/'calibration.json').read_text(encoding='utf-8'))
    return style, head, calibration


def predict(texts, features, heads):
    import numpy as np
    from scipy.special import softmax
    style, head, calibration = heads
    raw_style = np.asarray(style.decision_function(texts), dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    raw_gaussian = ((x-head['mean'])/head['scale']) @ head['coefficients'] + head['intercept']
    a = softmax(raw_style*calibration['style']['inverse_temperature'],axis=1)
    b = softmax(raw_gaussian*calibration['gaussian']['inverse_temperature'],axis=1)
    probability = .8*a + .2*b
    if probability.shape != (len(texts),18) or not np.isfinite(probability).all() or not np.allclose(probability.sum(1),1,atol=1e-12,rtol=0):
        raise ValueError('Invalid probabilities')
    return probability, raw_style, raw_gaussian


def read_test(path):
    with path.open(encoding='utf-8',newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id','text']:
            raise ValueError('Expected test id,text columns only')
        rows = list(reader)
    if not 1 <= len(rows) <= 4096 or any(not row['id'] or not row['text'] or None in row for row in rows):
        raise ValueError('Invalid test rows or exceeded 4096-row budget')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate test ID')
    return [r['id'] for r in rows], [r['text'] for r in rows]


def write_submission(path, ids, probability, labels):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    if len(labels) != 18 or len(set(labels)) != 18 or probability.shape != (len(ids),18):
        raise ValueError('Invalid submission mapping')
    models = [labels[int(i)] for i in np.argmax(probability,axis=1)]
    if path.exists():
        raise FileExistsError(path)
    pq.write_table(pa.table({'id':ids, 'model':models}),path)
    table = pq.read_table(path)
    if table.column_names != ['id','model'] or table['id'].to_pylist() != ids or table['model'].to_pylist() != models:
        raise ValueError('Parquet round-trip mismatch')


def run(args):
    block_network()
    root, out = args.assets.resolve(), args.output.resolve()
    manifest = verify_assets(root)
    versions = verify_runtime(args.mode)
    if args.action == 'test' and (platform.system() != 'Linux' or args.mode != 'cpu_float32'):
        raise ValueError('Deployment requires the validated Linux CPU runtime')
    if out.exists():
        raise FileExistsError('Preserve existing run and do not retry')
    out.mkdir(parents=True)
    claim = {'state':'started','action':args.action,'mode':args.mode,'attempted_forward_calls':0,
             'completed_forward_calls':0,'model_fits':0,'temperature_fits':0,'old_holdout_evaluations':0,
             'manifest_sha256':digest(root/'manifest.json'),'script_sha256':digest(__file__),
             'started_at_unix':time.time()}
    write_json(out/'claim.json',claim,exclusive=True)
    start = time.monotonic()
    limit = 900 if args.action == 'synthetic' else 9*3600
    expired = threading.Event()
    def time_guard():
        if not expired.wait(limit):
            (out/'timeout.txt').write_text('Hard wall-time budget exceeded',encoding='utf-8')
            os._exit(124)
    threading.Thread(target=time_guard,daemon=True).start()
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        ids, texts = (['synthetic-'+str(i) for i in range(4)],list(TEXTS)) if args.action == 'synthetic' else read_test(args.test)
        if args.reverse:
            if args.action != 'synthetic':
                raise ValueError('Reverse is a synthetic-only order check')
            ids, texts = list(reversed(ids)), list(reversed(texts))
        expected_calls = math.ceil(len(texts)/4)
        claim.update(rows=len(texts),max_forward_calls=expected_calls,versions=versions,python=sys.version,platform=platform.platform())
        write_json(out/'claim.json',claim)
        def before_forward():
            if claim['attempted_forward_calls'] >= expected_calls or time.monotonic()-start > limit:
                raise RuntimeError('Inference budget exhausted')
            claim['attempted_forward_calls'] += 1
            write_json(out/'claim.json',claim)
        with threadpool_limits(limits=2):
            encoder = load_encoder(root,args.mode)
            heads = load_heads(root)
            features = []
            for at in range(0,len(texts),4):
                features.append(extract(texts[at:at+4],encoder,before_forward))
                claim['completed_forward_calls'] += 1
                write_json(out/'claim.json',claim)
                if at % 128 == 0:
                    print(json.dumps({'completed_rows':min(at+4,len(texts)), 'total_rows':len(texts)}),flush=True)
            features = np.concatenate(features)
            probability, a, b = predict(texts,features,heads)
        labels = json.loads((root/'labels.json').read_text(encoding='utf-8'))['labels']
        write_submission(out/'submission.parquet',ids,probability,labels)
        if args.action == 'synthetic':
            np.savez_compressed(out/'synthetic.npz',ids=np.array(ids),features=features,probabilities=probability,style_scores=a,gaussian_scores=b)
        claim.update(state='completed',wall_seconds=time.monotonic()-start,output_sha256=digest(out/'submission.parquet'))
        write_json(out/'claim.json',claim)
        print(json.dumps(claim),flush=True)
    except BaseException as error:
        claim.update(state='failed',exception_type=type(error).__name__,error=str(error),wall_seconds=time.monotonic()-start)
        write_json(out/'claim.json',claim)
        raise
    finally:
        expired.set()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['synthetic','test'])
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['cpu_float32','cuda_bfloat16'],default='cpu_float32')
    p.add_argument('--test',type=Path)
    p.add_argument('--reverse',action='store_true')
    run(p.parse_args())
