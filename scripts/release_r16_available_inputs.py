"""Release R16 with the single inaccessible Kaggle input omitted."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/team_e654_r16_typography_half.ipynb'
SOURCE_SHA256 = 'f5d1f8f3622183863a74f03be7b7b1cea5156edd88bc0dd284e00c2356cbab77'
OUTPUT = ROOT / 'notebooks/team_e654_r16_typography_half_available.ipynb'
OMIT = {'sourceType': 'kernelVersion', 'sourceId': 362275911}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


if __name__ == '__main__':
    raw = SOURCE.read_bytes()
    assert digest(raw) == SOURCE_SHA256, 'Initial R16 candidate identity mismatch'
    original = json.loads(raw)
    nb = json.loads(raw)
    inputs = nb['metadata']['kaggle']['dataSources']
    assert len(inputs) == 31 and inputs.count(OMIT) == 1
    nb['metadata']['kaggle']['dataSources'] = [item for item in inputs if item != OMIT]
    nb['cells'][0]['source'] = '''# R16: typography half-strength with available inputs

Exploratory trial based on Roman Tamrazov V32 / 363222852 and our R14 package.
Final typography evidence strength is 0.5. The inaccessible kernel input
lightningv08/ntl-lora24-fivefold-training (362275911) is omitted. The unchanged
source skips lora24_public, including its plain (0.177) and centered (0.15)
contributions. The other 30 pinned inputs, coefficients and gates are retained.
This is not an exact complete-ensemble reproduction or an isolated typography
comparison. No completed full-ensemble local OOF gate or new score is claimed.
Our historical R12 Public Score is 0.654; the external V32 result is 0.660.
One private Save and one formal submission; no tuning after seeing the score.
Normal Save writes three placeholder rows; hidden rerun performs inference.
The original source attribution is retained below.
'''.splitlines(keepends=True)
    nb['metadata']['r16']['omitted_input'] = OMIT
    nb['metadata']['r16']['omitted_member'] = 'lora24_public'
    nb['metadata']['r16']['full_frozen_ensemble'] = False
    nb['metadata']['r16']['scope_authorization'] = 'User delegated best handling of missing member on 2026-10-29'
    code = [source(c) for c in nb['cells'] if c['cell_type'] == 'code']
    assert code == [source(c) for c in original['cells'] if c['cell_type'] == 'code']
    assert len(nb['metadata']['kaggle']['dataSources']) == 30
    for cell in code:
        compile(cell.split('\n', 1)[1] if cell.startswith('%%writefile ') else cell, '<r16-release-cell>', 'exec')
    packed = (json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode('utf-8')
    with OUTPUT.open('xb') as handle:
        handle.write(packed)
    receipt = {'source_candidate_sha256': SOURCE_SHA256, 'candidate_sha256': digest(packed),
               'candidate_code_sha256': [digest(c.encode()) for c in code],
               'omitted_input': OMIT, 'omitted_member': 'lora24_public',
               'pinned_inputs': nb['metadata']['kaggle']['dataSources'],
               'code_unchanged_from_initial_r16': True, 'typography_strength': 0.5,
               'full_frozen_ensemble': False, 'saved_runs_budget': 1, 'formal_submissions_budget': 1}
    with (ROOT / 'configs/r16_available_inputs_recipe_20261029.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'candidate_sha256': digest(packed), 'unchanged_code_cells': len(code), 'remaining_inputs': 30}))
