"""Package the single R15 typography change on the frozen R14 candidate."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/team_e654_r14_public660.ipynb'
OUTPUT = ROOT / 'notebooks/team_e654_r16_typography_half.ipynb'
BASE_SHA256 = '5fd8bf520af59a1841fb19a035e922df123934bb1d1d05625b6853f6351b9dae'
OLD = '    _safe_logits = _raw + _gate[:, None] * _evidence\n'
NEW = '    _safe_logits = _raw + 0.5 * _gate[:, None] * _evidence\n'


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def build():
    raw = BASE.read_bytes()
    if sha(raw) != BASE_SHA256:
        raise ValueError('Frozen R14 candidate byte identity mismatch')
    base = json.loads(raw)
    candidate = json.loads(raw)
    codes = [c for c in candidate['cells'] if c['cell_type'] == 'code']
    if len(codes) != 7 or sum(source(c).count(OLD) for c in codes) != 1:
        raise ValueError('Unexpected typography patch location')
    infer = next(c for c in codes if source(c).startswith('%%writefile infer.py\n'))
    infer['source'] = source(infer).replace(OLD, NEW, 1).splitlines(keepends=True)
    before = [source(c) for c in base['cells'] if c['cell_type'] == 'code']
    after = [source(c) for c in codes]
    if [c.replace(NEW, OLD) for c in after] != before:
        raise ValueError('Unexpected code change beyond typography strength')
    for code in after:
        compile(code.split('\n', 1)[1] if code.startswith('%%writefile ') else code, '<r16-cell>', 'exec')
    candidate['cells'][0]['source'] = '''# R16: one exploratory typography half-strength trial

Based on the frozen complete R14 ensemble, from Roman Tamrazov V32 / 363222852.
Only the final typography evidence strength changes from 1.0 to 0.5. All 31
input references, model settings, gates and operational safeguards are retained.
Our established R12 public result is 0.654; the external V32 result is 0.660.
This candidate has no score and no completed full-ensemble local OOF gate.
One private Save and one formal submission are authorized. No score-based tuning.
Normal Save writes the three-row placeholder; hidden rerun performs inference.
The original source attribution is retained below.
'''.splitlines(keepends=True)
    candidate['metadata']['r16'] = {
        'base_candidate_sha256': BASE_SHA256,
        'typography_strength': 0.5,
        'local_complete_ensemble_cv': 'not_evaluated',
        'public_score': None,
        'saved_runs_budget': 1,
        'formal_submissions_budget': 1,
    }
    return candidate


if __name__ == '__main__':
    candidate = build()
    packed = (json.dumps(candidate, ensure_ascii=False, indent=1) + '\n').encode('utf-8')
    with OUTPUT.open('xb') as handle:
        handle.write(packed)
    recipe = {
        'base_candidate_sha256': BASE_SHA256,
        'candidate_sha256': sha(packed),
        'candidate_code_sha256': [sha(source(c).encode()) for c in candidate['cells'] if c['cell_type'] == 'code'],
        'pinned_inputs': candidate['metadata']['kaggle']['dataSources'],
        'typography_strength': 0.5,
        'saved_runs_budget': 1,
        'formal_submissions_budget': 1,
        'local_complete_ensemble_cv': 'not_evaluated',
    }
    with (ROOT / 'configs/r16_frozen_recipe_20261029.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(recipe, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'candidate_sha256': recipe['candidate_sha256'], 'code_cells': 7, 'pinned_inputs': len(recipe['pinned_inputs']), 'changed_code_cells': 1}))
