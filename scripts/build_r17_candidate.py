"""Prepare the one R17 Llama-restoration candidate; no execution or upload."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/team_e654_r16_typography_half_runnable.ipynb'
BASE_SHA256 = '8b04a036a5aa43c41907fd6404849179cd6bee1d4588b6c866484487f6bd6075'
ORIGINAL = ROOT / 'notebooks/team_e654_r16_typography_half.ipynb'
ORIGINAL_SHA256 = 'f5d1f8f3622183863a74f03be7b7b1cea5156edd88bc0dd284e00c2356cbab77'
CONTRACT = ROOT / 'configs/r17_llama_asset_contract_20261030.json'
OUTPUT = ROOT / 'notebooks/team_e659_r17_restore_llama_pending_access.ipynb'


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def frozen_bytes(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        # Git on Windows may change only JSON's outer line endings on checkout.
        # Accept that transport form only when normalization restores the exact
        # frozen digest; code, metadata or any other byte change still fails.
        raw = raw.replace(b'\r\n', b'\n')
        if sha(raw) != expected:
            raise ValueError('R17 frozen scored R16 candidate changed')
    return raw


def build():
    raw = frozen_bytes(BASE, BASE_SHA256)
    nb = json.loads(raw)
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    before = [source(c) for c in cells]
    original = [source(c) for c in json.loads(frozen_bytes(ORIGINAL, ORIGINAL_SHA256))['cells'] if c['cell_type'] == 'code']
    contract = json.loads(CONTRACT.read_bytes())
    helper = (ROOT / 'scripts/r17_runtime_contract.py').read_text(encoding='utf-8')
    helper_cell = {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                   'outputs': [], 'source': ('%%writefile r17_contract.py\n' + helper).splitlines(keepends=True)}
    nb['cells'].insert(nb['cells'].index(cells[1]), helper_cell)
    cells[0]['source'] = ('import time as _r17time\nR17_START = _r17time.monotonic()\n' + before[0]).splitlines(keepends=True)
    init = before[1] + '''\n# R17: exact approved base; do not silently substitute or drop the member.
from r17_contract import verify_llama_assets, run_logged, remaining_seconds, validate_comp_log, validate_main_log
BL = '/kaggle/input/llama-3.1/transformers/8b/1'
def _r17_adir(name):
    hits = glob.glob(f'/kaggle/input/**/nlm-l24-adapters-v5/{name}/heads.pt', recursive=True)
    if len(hits) != 1:
        raise ValueError('R17 requires one unique pinned adapter directory: ' + name)
    return os.path.dirname(hits[0])
ALL8V5 = [_r17_adir(f'll8v5_f{k}') for k in range(5)] + [_r17_adir('ll8v5')]
'''
    init += 'R17_ASSET_CONTRACT = ' + repr(contract) + '\n'
    init += '''R17_ASSETS = verify_llama_assets(BL, ALL8V5, R17_ASSET_CONTRACT, R17_START)
with open('/kaggle/working/r17_asset_verification.json', 'w') as _r17f:
    json.dump(R17_ASSETS, _r17f, indent=2)
print('R17 verified Llama base and all six adapters', ALL8V5, flush=True)
'''
    cells[1]['source'] = init.splitlines(keepends=True)
    comp_old = "    subprocess.run([sys.executable, 'comp_stack.py'], check=True)\n"
    comp_new = "    _r17_comp_log = run_logged([sys.executable, 'comp_stack.py'], '/kaggle/working/r17_comp.log', remaining_seconds(R17_START))\n    R17_COMP = validate_comp_log(_r17_comp_log)\n"
    if before[5].count(comp_old) != 1:
        raise ValueError('R17 unexpected COMP invocation')
    cells[5]['source'] = before[5].replace(comp_old, comp_new, 1).splitlines(keepends=True)
    llama_line = [line for line in original[6].splitlines(keepends=True) if '"name": "ll8v5"' in line]
    qwen_line = [line for line in before[6].splitlines(keepends=True) if '"name": "q8v5"' in line]
    if len(llama_line) != 1 or len(qwen_line) != 1:
        raise ValueError('R17 unexpected model list')
    main = before[6].replace(qwen_line[0], qwen_line[0] + llama_line[0], 1)
    budget_old = '    ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)\n'
    if main.count(budget_old) != 1:
        raise ValueError('R17 unexpected ATLAS budget calculation')
    main = main.replace(budget_old, '    ATLAS_BUDGET = int(remaining_seconds(R17_START) - 600)\n', 1)
    old = '    subprocess.run(cmd, check=True)\n'
    new = '''    _r17_main_log = run_logged(cmd, '/kaggle/working/r17_main.log', remaining_seconds(R17_START))
    R17_MAIN = validate_main_log(_r17_main_log)
    with open('/kaggle/working/r17_runtime_receipt.json', 'w') as _r17f:
        json.dump({'comp': R17_COMP, 'main': R17_MAIN, 'wall_seconds': _r17time.monotonic() - R17_START}, _r17f, indent=2)
'''
    if main.count(old) != 1:
        raise ValueError('R17 unexpected main invocation')
    cells[6]['source'] = main.replace(old, new, 1).splitlines(keepends=True)
    after = [source(c) for c in cells]
    if any(before[i] != after[i] for i in (2, 3, 4)):
        raise ValueError('R17 prediction script changed')
    for cell in [c for c in nb['cells'] if c['cell_type'] == 'code']:
        code = source(cell)
        compile(code.split('\n', 1)[1] if code.startswith('%%writefile ') else code, '<r17-cell>', 'exec')
    nb['cells'][0]['source'] = '''# R17: restore the Llama member, pending official input access

Scored base: our R16 V2 / 363487968, Public Score 0.659. This candidate has no score.
Restore original main ll8v5 with its six v5 adapters, weight 1.0 and twin 0.35.
Keep typography strength 0.5 and every other prediction setting unchanged.
The three prediction scripts are byte-identical to scored R16. Added checks verify
asset bytes and actual runtime members, enforce the 8.3-hour ceiling, and reject
COMP/ATLAS error fallbacks. The historical public-L24 checkpoint remains absent.
Normal Save performs asset verification and writes three placeholder rows; only
the formal hidden rerun runs models. No complete fold-isolated local OOF claim.

RELEASE BLOCKED: the official metaresearch/llama-3.1/Transformers/8b/1 input must
be accessible, attached and bound to its exact version ID before import/Save.
This pending file preserves the 30 R16 inputs; it is not an upload-ready release.
One next-round formal submission only; no score-driven retry or weight search.
The original authors and attribution below remain preserved.
'''.splitlines(keepends=True)
    nb['metadata'].pop('r16', None)
    nb['metadata']['r17'] = {'scored_base_sha256': BASE_SHA256, 'scored_base_version': 363487968,
                            'scored_base_public_score': 0.659, 'typography_strength': 0.5,
                            'restored_member': 'main.ll8v5', 'omitted_members': ['comp.lora24_public'],
                            'required_model_ref': contract['kaggle_ref'], 'release_ready': False,
                            'local_complete_ensemble_cv': 'not_evaluated', 'formal_submissions_budget': 1,
                            'saves_budget': 2, 'public_score': None}
    return nb


if __name__ == '__main__':
    nb = build()
    packed = (json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode('utf-8')
    OUTPUT.write_bytes(packed)
    recipe = {'candidate': OUTPUT.relative_to(ROOT).as_posix(), 'candidate_sha256': sha(packed),
              'base_sha256': BASE_SHA256, 'code_sha256': [sha(source(c).encode()) for c in nb['cells'] if c['cell_type'] == 'code'],
              'retained_inputs': nb['metadata']['kaggle']['dataSources'], 'required_additional_model': nb['metadata']['r17']['required_model_ref'],
              'release_ready': False, 'formal_submissions': 0, 'saves': 0}
    (ROOT / 'configs/r17_pending_recipe_20261030.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'candidate_sha256': recipe['candidate_sha256'], 'code_cells': len(recipe['code_sha256']),
                      'retained_inputs': len(recipe['retained_inputs']), 'release_ready': False}))
