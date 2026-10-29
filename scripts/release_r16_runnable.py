"""Repair asset initialization for one available-input R16 submission."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/team_e654_r16_typography_half_available.ipynb'
OUTPUT = ROOT / 'notebooks/team_e654_r16_typography_half_runnable.ipynb'
SOURCE_SHA256 = '5bc63b9586b86a047255c61177a7a16535352fe50f144c2ef1291b7fb14a1760'
REMOVE_PREFIXES = (
    'HEADS = ', 'I8, I14 = ', 'A8 = ', 'A14 = ', 'A25 = ',
    'print(B8, B14, B25,', 'N8, N8C = ', "print('nishant',",
    'BL = ', 'AL = ', "print('llama',", 'A8S7 = ', "print('seed7',",
    'ALL8V5 = ', "print('v5',",
)


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def build():
    raw = SOURCE.read_bytes()
    assert digest(raw) == SOURCE_SHA256
    nb = json.loads(raw)
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    before = [source(c) for c in cells]
    lines = before[1].splitlines(keepends=True)
    for prefix in REMOVE_PREFIXES:
        assert sum(line.startswith(prefix) for line in lines) == 1, prefix
    init = ''.join(line for line in lines if not line.startswith(REMOVE_PREFIXES))
    old = "B8, B14, B25 = mdir('/qwen-3/', '/8b-base/'), mdir('/qwen-3/', '/14b-base/'), mdir('/qwen2.5/', '/7b/')"
    new = "B8, B14 = mdir('/qwen-3/', '/8b-base/'), mdir('/qwen-3/', '/14b-base/')"
    assert init.count(old) == 1
    cells[1]['source'] = init.replace(old, new, 1).splitlines(keepends=True)
    main_lines = before[6].splitlines(keepends=True)
    removed = [line for line in main_lines if '"name": "ll8v5"' in line]
    assert len(removed) == 1
    cells[6]['source'] = ''.join(line for line in main_lines if line not in removed).splitlines(keepends=True)
    after = [source(c) for c in cells]
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [1, 6]
    for code in after:
        compile(code.split('\n', 1)[1] if code.startswith('%%writefile ') else code, '<r16-runnable>', 'exec')
    old_models = next(n for n in ast.walk(ast.parse(before[6])) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MODELS' for t in n.targets))
    new_models = next(n for n in ast.walk(ast.parse(after[6])) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MODELS' for t in n.targets))
    old_models.value.elts = [n for n in old_models.value.elts if not any(isinstance(k, ast.Constant) and k.value == 'name' and isinstance(v, ast.Constant) and v.value == 'll8v5' for k, v in zip(n.keys, n.values))]
    assert ast.dump(old_models) == ast.dump(new_models)
    nb['cells'][0]['source'] = '''# R16: one available-input ensemble trial

This exploratory trial uses the R14/V32 recipe with final typography strength
0.5. Two unavailable members are explicitly omitted: COMP lora24_public
(deleted input 362275911, plain 0.177 and centered 0.15) and main ll8v5
(missing gated Llama-3.1-8B base, weight 1.0 and twin 0.35).
The remaining main members are q14v5, q8v5, n14b, q25_14 and deb. Their parameters,
the other COMP members, ATLAS and the 30 remaining input references are retained.
Unused legacy asset discovery is removed. V1 failed during initialization before
model inference; V2 is a packaging repair, with one formal submission in total.
This is not an exact complete-ensemble reproduction or typography-only comparison.
No local complete-ensemble OOF gain or new score is claimed. Historical R12 Public
Score is 0.654; external V32 is 0.660. No tuning after seeing this public score.
Normal Save writes three placeholder rows; hidden rerun performs inference.
The original source attribution is retained below.
'''.splitlines(keepends=True)
    nb['metadata']['r16'].update({'omitted_members': ['comp.lora24_public', 'main.ll8v5'],
        'saved_runs_budget': 2, 'formal_submissions_budget': 1,
        'engineering_retry_reason': 'V1 missing gated Llama base before inference; omit ll8v5 and unused discovery',
        'full_frozen_ensemble': False})
    return nb


if __name__ == '__main__':
    nb = build()
    packed = (json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode()
    with OUTPUT.open('xb') as handle:
        handle.write(packed)
    receipt = {'candidate_sha256': digest(packed), 'previous_candidate_sha256': SOURCE_SHA256,
               'code_sha256': [digest(source(c).encode()) for c in nb['cells'] if c['cell_type'] == 'code'],
               'changed_code_cell_indices_from_v1': [1, 6], 'omitted_members': ['comp.lora24_public', 'main.ll8v5'],
               'pinned_inputs': nb['metadata']['kaggle']['dataSources'], 'saved_runs_budget': 2, 'formal_submissions_budget': 1}
    with (ROOT / 'configs/r16_runnable_recipe_20261029.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'candidate_sha256': digest(packed), 'main_models': 5, 'input_references': 30, 'changed_code_cells': [1, 6]}))
