"""Package the tested fingerprint repair; preserve every frozen V1 artifact."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/r18'
FIX = EVIDENCE / 'fingerprint_fix_local_20261102'


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def main():
    recipe = json.loads((ROOT / 'configs/r18_recipe_20261101.json').read_text())
    original_path = ROOT / recipe['candidate']
    original_bytes = original_path.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == recipe['candidate_sha256']
    old = json.loads(original_bytes)
    new = json.loads(original_bytes)
    original_helper = (ROOT / 'scripts/r18_runtime_contract.py').read_text(encoding='utf-8')
    fixed_helper = (FIX / 'r18_runtime_contract.py').read_text(encoding='utf-8')
    diagnostic = json.loads((FIX / 'receipt.json').read_text())
    assert diagnostic['status'] == 'original_failed_fixed_9_cases_passed_local_only'
    assert len(diagnostic['cases']) == 9 and diagnostic['gpu_operations'] == 0
    assert hashlib.sha256(fixed_helper.encode()).hexdigest() == diagnostic['fixed_source_text_sha256']
    indexes = [i for i, c in enumerate(new['cells']) if source(c).startswith('%%writefile r18_contract.py\n')]
    assert len(indexes) == 1
    index = indexes[0]
    assert source(new['cells'][index]) == '%%writefile r18_contract.py\n' + original_helper
    new['cells'][index]['source'] = ('%%writefile r18_contract.py\n' + fixed_helper).splitlines(keepends=True)
    assert [i for i, (a, b) in enumerate(zip(old['cells'], new['cells'])) if a != b] == [index]
    assert old['metadata'] == new['metadata']
    for cell in new['cells']:
        if cell['cell_type'] == 'code':
            text = source(cell)
            compile(text.split('\n', 1)[1] if text.startswith('%%writefile ') else text, '<r18-v2>', 'exec')
    candidate = ROOT / 'notebooks/team_e659_r18_adapter_isolation_v2.ipynb'
    packed = (json.dumps(new, ensure_ascii=False, indent=1) + '\n').encode()
    candidate.write_bytes(packed)
    target = EVIDENCE / 'cli_release_v2'
    target.mkdir(exist_ok=True)
    (target / candidate.name).write_bytes(packed)
    metadata = json.loads((EVIDENCE / 'cli_release/kernel-metadata.json').read_text())
    metadata['code_file'] = candidate.name
    (target / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    receipt = {'version_intended': 2, 'candidate': candidate.relative_to(ROOT).as_posix(),
        'candidate_sha256': hashlib.sha256(packed).hexdigest(),
        'v1_candidate_preserved_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'changed_cell_index': index, 'change': 'integer-safe sampled fingerprint indices; skip empty parameters',
        'all_other_cells_exact': True, 'all_notebook_metadata_exact': True,
        'external_metadata_changes': ['code_file'], 'prediction_code_changes': 0,
        'input_versions_changed': 0, 'preflight_budget_seconds': 900,
        'save_attempts_max': 2, 'no_third_save': True, 'formal_submissions_max': 1,
        'diagnostic_receipt': 'outputs/r18/fingerprint_fix_local_20261102/receipt.json'}
    (ROOT / 'configs/r18_v2_repair_20261102.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
