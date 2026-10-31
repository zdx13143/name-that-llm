"""Kaggle Save only: extracted actual function, artificial text, real fixed assets."""
import ast
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

started = time.monotonic()
torch.set_grad_enabled(False)
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(gpu_names) != 2 or any('T4' not in name for name in gpu_names):
    raise RuntimeError('R18 requires the approved free T4 x2 runtime: ' + str(gpu_names))
os.environ['R18_REQUIRE_CUDA'] = '1'
configuration = json.loads(Path('r18_preflight_specs.json').read_text())
LABELS = configuration['labels']
log = lambda *args: print(*args, flush=True)
trace = Path('/kaggle/working/r18_lifecycle.jsonl')
trace.unlink(missing_ok=True)
os.environ['R18_TRACE_PATH'] = str(trace)
tree = ast.parse(Path('infer.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'l24_predict')
exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-r18-l24>', 'exec'))
texts = ['The small blue boat crossed the quiet lake.', 'A careful observer counted seven red stones beside a tree.']
results = []
for spec in configuration['specs']:
    family = spec['name']
    if family not in ('q14v5', 'q8v5') or len(spec['adapters']) != 6:
        raise ValueError('R18 preflight requires both six-adapter Qwen families')
    expected = [f'{family}_f{i}' for i in range(5)] + [family]
    if [Path(p).name for p in spec['adapters']] != expected:
        raise ValueError('R18 adapter order mismatch')
    for adapter in spec['adapters']:
        if 'nlm-l24-adapters-v5' not in Path(adapter).parts:
            raise ValueError('R18 unexpected adapter mount')
        for filename in ('adapter_config.json', 'adapter_model.safetensors', 'heads.pt'):
            if not (Path(adapter) / filename).is_file() or (Path(adapter) / filename).stat().st_size == 0:
                raise ValueError('R18 missing asset: ' + adapter + '/' + filename)
    p, c, features = l24_predict(spec, texts, groups=np.array([0, 0]), train_texts=texts)
    if p.shape != (2, 18) or c.shape != (2, 18) or not all(np.isfinite(x).all() for x in (p, c, *features)):
        raise ValueError('R18 preflight output invalid')
    results.append({'family': family, 'shape': list(p.shape), 'finite': True, 'adapters': expected})
events = [json.loads(line) for line in trace.read_text().splitlines()]
for family in ('q14v5', 'q8v5'):
    family_events = [e for e in events if e['family'] == family]
    if [e['event'] for e in family_events] != ['before_load', 'loaded', 'released'] * 6:
        raise ValueError('R18 incomplete lifecycle receipt')
    fingerprints = {e['sample_sha256'] for e in family_events if e['event'] == 'loaded'}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError('R18 missing fresh-base fingerprint evidence')
receipt = {'passed': True, 'artificial_passages': 2, 'families': results, 'lifecycle_events': len(events),
    'gpu_names': gpu_names,
    'seconds': time.monotonic() - started, 'accuracy_evaluated': False,
    'sample_fingerprint_is_full_asset_hash': False}
Path('/kaggle/working/r18_preflight_receipt.json').write_text(json.dumps(receipt, indent=2))
print('R18_PREFLIGHT_PASS ' + json.dumps(receipt), flush=True)
