"""Prepare the required S7 member for nested development folds; never train.

The generated worker is a source candidate, with execution disabled until a
separate producer/resource release is implemented. It preserves the archived
Apache-2.0 training core and changes data routing, saving and metric emission.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from r15_admit_approved_export import load_approved_export  # noqa: E402
from r15_validate_dev_export import (  # noqa: E402
    bounded_file, canonical_hash, load_frozen_protocol, nested_plan, require,
)
from r15_typography_replay import LABELS  # noqa: E402

SOURCE_SHA256 = 'd0c792e1262683e7661ec6b195a73e98facd3713c609a3e849ac85d0513c2f45'
BASE_REVISION = '49e3418fbbbca6ecbdf9608b4d22e5a407081db4'
RECIPE = dict(seed=7, layers=[2, 4, 6, 8, 12, 16, 20, 24], keep=24,
              max_len=447, epochs=3, accum=6, lr_adapter=1e-4, lr_head=1e-3,
              wd=0.01, warmup=0.10, r=16, alpha=32, ldrop=0.05)
OUTPUT = ROOT / 'outputs/r15_s7_adapter'


def load_admitted():
    protocol = load_frozen_protocol()
    directory = ROOT / 'outputs/r15_dev_export'
    policy = json.loads(bounded_file(ROOT / 'configs', 'r15_approved_export_policy_20261029.json', 65536))
    review = json.loads(bounded_file(ROOT / 'configs', 'r15_approved_export_review_20261029.json', 65536))
    data, blob, extraction = load_approved_export(directory, policy, review, protocol)
    admission = json.loads(bounded_file(directory, 'admission_receipt.json', 65536))
    plan = json.loads(bounded_file(directory, 'nested_plan.json', 2 * 1024**2))
    require(admission.get('data_admitted') is True
            and admission.get('training_enabled') is False
            and admission.get('real_scoring_enabled') is False, 'Unexpected admission scope')
    for field, value in [('development_sha256', hashlib.sha256(blob).hexdigest()),
                         ('extraction_receipt_sha256', canonical_hash(extraction)),
                         ('policy_sha256', canonical_hash(policy)), ('review_sha256', canonical_hash(review)),
                         ('nested_plan_sha256', canonical_hash(plan))]:
        require(admission.get(field) == value, 'Admission binding differs')
    require(plan == nested_plan(data, protocol), 'Nested plan differs from frozen data')
    return data, plan, admission


def job_positions(plan, outer_fold, inner_fold=None):
    require(type(outer_fold) is int and 0 <= outer_fold < 5, 'Invalid outer fold')
    require(inner_fold is None or (type(inner_fold) is int and 0 <= inner_fold < 5),
            'Invalid inner fold')
    outer = plan['folds'][outer_fold]
    require(outer['outer_fold'] == outer_fold, 'Outer order changed')
    selected = outer if inner_fold is None else outer['inner'][inner_fold]
    if inner_fold is not None:
        require(selected['inner_fold'] == inner_fold, 'Inner order changed')
    fit, predict = selected['train_rows'], selected['valid_rows']
    require(len(fit) >= 3 and bool(predict) and not set(fit) & set(predict), 'Invalid row envelope')
    if inner_fold is not None:
        require(set(fit) | set(predict) == set(outer['train_rows']), 'Inner envelope escapes outer train')
    return fit, predict


def materialize_envelope(data, plan, outer_fold, inner_fold=None):
    """All labels exposed to a worker belong to fit; predictions carry no truth."""
    fit, predict = job_positions(plan, outer_fold, inner_fold)
    fit_groups = {data['groups'][i] for i in fit}
    pred_groups = {data['groups'][i] for i in predict}
    require(not fit_groups & pred_groups, 'Prompt leakage')
    require(sorted({data['labels'][data['truth'][i]] for i in fit}) == list(LABELS),
            'Training envelope must contain all canonical classes')
    def record(i):
        return dict(id=data['ids'][i], prompt_id=data['groups'][i], text=data['texts'][i])
    fitting = [dict(record(i), model=data['labels'][data['truth'][i]]) for i in fit]
    prediction = [record(i) for i in predict]
    return {'fit': fitting, 'predict': prediction, 'labels': list(LABELS),
            'fit_positions': list(fit), 'predict_positions': list(predict),
            'anchor_positions': list(fit[:3]),
            'fit_groups_sha256': canonical_hash(sorted(fit_groups)),
            'predict_groups_sha256': canonical_hash(sorted(pred_groups))}


def worker_inputs(config_path):
    """Reserved entry for a future reviewed runtime. No current training release."""
    # The current preparation command never calls this function or imports torch.
    # A complete execution release needs model-file provenance, a supervisor and
    # a measured device budget. A boolean in an editable job JSON is not a release.
    raise ValueError('S7 source prepared only: training/resource release is not implemented')


def archived_worker(source):
    require(hashlib.sha256(source).hexdigest() == SOURCE_SHA256, 'Archived trainer changed')
    tree = ast.parse(source.decode('utf-8'))
    matches = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'WORKER' for t in node.targets)]
    require(len(matches) == 1 and isinstance(matches[0], str), 'Missing literal worker')
    return matches[0]


def adapt_worker(source):
    worker = archived_worker(source)
    def replace(old, new):
        nonlocal worker
        require(worker.count(old) == 1, 'Source adaptation anchor changed')
        worker = worker.replace(old, new, 1)

    # The closed release gate runs before torch/transformers imports or device work.
    replace('import numpy as np, pandas as pd, torch', '''from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from build_r15_s7_adapter import worker_inputs
cfg, envelope = worker_inputs(sys.argv[1])
import numpy as np, pandas as pd, torch''')
    replace("cfg = json.loads(sys.argv[1]); jobs = sys.argv[2].split(',')\ntrain = pd.read_csv(cfg['train_csv'])\nLABELS = sorted(train.model.unique()); y = train.model.map({l: i for i, l in enumerate(LABELS)}).values\npid = train.prompt_id.values\nprompts = np.array(sorted(train.prompt_id.unique()))\nrng0 = np.random.RandomState(cfg['fold_seed']); rng0.shuffle(prompts)   # == scripts/common.py folds()\nfold = train.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).values", '''jobs = [cfg['job_id']]
train = pd.DataFrame(envelope['fit'] + envelope['predict'])
LABELS = envelope['labels']
nfit = len(envelope['fit'])
y = np.array([LABELS.index(row['model']) for row in envelope['fit']] + [-1] * len(envelope['predict']))
pid = train.prompt_id.values
fit = np.arange(nfit)
anchor = np.arange(3)
prediction_rows = np.arange(nfit, len(train))
valid = np.concatenate([anchor, prediction_rows])''')
    replace("outdir = os.path.join('lora_out', f'job_{job}'); os.makedirs(outdir, exist_ok=True)",
            "outdir = cfg['output_dir']; os.makedirs(outdir, exist_ok=False)")
    replace("rng = np.random.RandomState(cfg['seed'] + (99 if job == 'final' else int(job)))\n    fit = np.arange(len(train)) if job == 'final' else np.where(fold != int(job))[0]\n    valid = np.arange(3) if job == 'final' else np.where(fold == int(job))[0]",
            "rng = np.random.RandomState(cfg['seed'] + 99)  # final-style fit within this envelope")
    replace("np.savez_compressed(os.path.join(outdir, 'valid_features.npz'), ids=train.id.values[valid], features=Z.astype(np.float16))",
            "np.savez_compressed(os.path.join(outdir, 'valid_features.npz'), ids=train.id.values[anchor], features=Z[:3].astype(np.float16))")
    replace("acc = float((lg.argmax(1) == y[valid]).mean())",
            "np.savez_compressed(os.path.join(outdir, 'heldout_predictions.npz'), ids=train.id.values[prediction_rows], classes=np.array(LABELS), logits=lg[3:])")
    replace("if job == 'final':", "if True:  # Persist a complete, paired bundle for EVERY isolated job.")
    replace("json.dump({'job': job, 'valid_acc': acc, 'steps': step, 'skipped': skipped, 'secs': time.time() - t0},\n              open(os.path.join(outdir, 'COMPLETE.json'), 'w'))\n    print(f'job {job} done: valid acc {acc:.4f} steps {step} skipped {skipped} ({time.time() - t0:.0f}s)', flush=True)",
            "json.dump({'job': job, 'steps': step, 'skipped': skipped, 'secs': time.time() - t0,\n               'fit_rows': nfit, 'prediction_rows': len(prediction_rows),\n               'fit_groups_sha256': envelope['fit_groups_sha256'],\n               'predict_groups_sha256': envelope['predict_groups_sha256'],\n               'heldout_metrics_computed': False, 'runtime_binding': cfg['runtime_binding']},\n              open(os.path.join(outdir, 'COMPLETE.json'), 'w'))\n    print(f'job {job} done: steps {step} skipped {skipped} ({time.time() - t0:.0f}s)', flush=True)")
    header = '# Derived from Lucas Boesen S7 V1 / 358911769; Apache-2.0.\n'
    header += '# Distinct R15 nested producer source candidate; current runtime release is closed.\n'
    result = header + worker
    compile(result, '<r15-s7-adapted-worker>', 'exec')
    return result


def fp16_weight_lower_bound(config):
    """Embedding plus attention/MLP matrices only; excludes all training overhead."""
    require(config['model_type'] == 'qwen3' and config['hidden_size'] == 4096
            and config['num_hidden_layers'] == 36, 'Wrong generic base architecture')
    hidden = config['hidden_size']
    q = config['num_attention_heads'] * config['head_dim']
    kv = config['num_key_value_heads'] * config['head_dim']
    per_layer = 2 * hidden * q + 2 * hidden * kv + 3 * hidden * config['intermediate_size']
    parameters = hidden * config['vocab_size'] + RECIPE['keep'] * per_layer
    return {'matrix_parameters': parameters, 'fp16_bytes': 2 * parameters,
            'fp16_gib': 2 * parameters / 1024**3,
            'excludes': ['norms', 'LoRA parameters', 'readouts', 'gradients', 'optimizer', 'activations', 'allocator']}


def verify_sampling(worker, envelope):
    """Execute only the unchanged NumPy sampler, never the model worker."""
    import numpy as np
    functions = [n for n in ast.parse(worker).body if isinstance(n, ast.FunctionDef) and n.name == 'triplets']
    require(len(functions) == 1, 'Sampler is missing')
    pid = np.asarray([r['prompt_id'] for r in envelope['fit']])
    namespace = {'np': np, 'pid': pid, 'cfg': RECIPE}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<unchanged-s7-sampler>', 'exec'), namespace)
    rng = np.random.RandomState(RECIPE['seed'] + 99)
    plans = [namespace['triplets'](np.arange(len(pid)), rng) for _ in range(RECIPE['epochs'])]
    for epoch in plans:
        require(len(epoch) % RECIPE['accum'] == 0, 'Incomplete accumulation group')
        for batch in epoch:
            require(len(batch) == 3 and all(0 <= i < len(pid) for i in batch), 'Sampler escaped fit rows')
            require(len(set(pid[batch])) == 1, 'Sampler mixed prompts')
    return {'microbatches_per_epoch': [len(p) for p in plans],
            'optimizer_steps_planned': sum(len(p) for p in plans) // RECIPE['accum'],
            'sampler_fit_only_verified': True}


def main():
    import time
    start = time.process_time()
    require(not OUTPUT.exists(), 'Prepared output exists; no overwrite')
    source = bounded_file(ROOT / 'reference_sources/r15_p0', 's7_v1.py.txt', 32768)
    worker = adapt_worker(source)
    data, plan, admission = load_admitted()
    base = json.loads(bounded_file(ROOT / 'reference_sources/r15_p0e/qwen3_8b_base', 'config.json', 16384))
    jobs = []
    for outer in range(5):
        for inner in [None, 0, 1, 2, 3, 4]:
            envelope = materialize_envelope(data, plan, outer, inner)
            job_id = f'outer_{outer}_' + ('final' if inner is None else f'inner_{inner}')
            jobs.append({'job_id': job_id, 'outer_fold': outer, 'inner_fold': inner,
                         'fit_rows': len(envelope['fit']), 'predict_rows': len(envelope['predict']),
                         'fit_groups_sha256': envelope['fit_groups_sha256'],
                         'predict_groups_sha256': envelope['predict_groups_sha256'],
                         'row_positions_sha256': canonical_hash({k: envelope[k] for k in
                             ('fit_positions', 'predict_positions', 'anchor_positions')}),
                         'sampling_plan': verify_sampling(worker, envelope),
                         'training_enabled': False})
    report = {'schema': 'ntl-r15-s7-source-adaptation-v1', 'member': 'comp.lora24_seed7',
              'producer': 'r15p1-nested-dev120-draft', 'source_sha256': SOURCE_SHA256,
              'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(),
              'development_sha256': admission['development_sha256'],
              'nested_plan_sha256': admission['nested_plan_sha256'], 'recipe': RECIPE,
              'sampling_seed_policy': 'seed + 99 for each final-style envelope; no old seed-0 fold discovery',
              'base_repository': 'Qwen/Qwen3-8B-Base', 'base_revision_candidate': BASE_REVISION,
              'original_kaggle_base_revision_proven': False, 'weight_lower_bound': fp16_weight_lower_bound(base),
              'jobs': jobs, 'training_enabled': False, 'real_scoring_enabled': False,
              'missing_runtime_release': ['reviewed local base/tokenizer files', 'resource supervisor and measured budget',
                                          'full nested producer integration and calibration specification'],
              'process_cpu_seconds_before_write': time.process_time() - start}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / 'worker.py').write_text(worker, encoding='utf-8', newline='\n')
    (OUTPUT / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'source_adapted_execution_disabled', 'job_envelopes': len(jobs),
                      'fp16_weight_lower_bound_gib': report['weight_lower_bound']['fp16_gib'],
                      'training_enabled': False, 'process_cpu_seconds': time.process_time() - start}, indent=2))


if __name__ == '__main__':
    main()
