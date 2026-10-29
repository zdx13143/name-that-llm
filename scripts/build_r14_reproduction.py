"""Package the exact scored V32 full ensemble with fail-closed run boundaries."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'notebooks/reference_public660_v32.ipynb'
OUTPUT = ROOT / 'notebooks/team_e654_r14_public660.ipynb'
RAW_SHA256 = 'b7b3c05c25abc2a71bb6d25fdea2d5e596d711c33a459cfeed6c53acf7b74f53'
INPUT_SHA256 = '5d479c39ef803dbaf93d2774aeca6e6e3d21d281924b87ed493f210e94b0e6f8'
CODE_SHA256 = [
    '4fc27dca93a92adfb2759ca65260e5285898d73fc8ccd1acd401982e5f98ed37',
    'b53e305fd7a755d138f931c55ac5f3762ed3978d73a65594b5ce2523fdf165ff',
    '7e42fda861d0dcf132d2b514ca6e33ad6dc52c4e78cd398f1f037623ba2d5c42',
    'e3aa49d5a11e4ce2b504b826a7f8dddb13077e4b512e92d55a0a2807ca82cc18',
    '5dcb5ebe86d6b3cb21dba316fdae1aa8bf6777504d4581552856af982e7d2b92',
    '9fc7b18227a17e268eb4f00ed4b910a0a39f1c324f696db285e9580ce88b2342',
]
OLD_COMP = "!python comp_stack.py 2>&1 | grep -vE 'Warning|warn'"
NEW_COMP = """import subprocess, sys
subprocess.run([sys.executable, 'comp_stack.py'], check=True)
import numpy as _r14np
with open('/kaggle/working/comp_meta.json', encoding='utf-8') as _r14f:
    _r14meta = json.load(_r14f)
_r14logit = _r14np.load('/kaggle/working/comp_logit.npy', allow_pickle=False)
_r14_validate_comp(_r14meta, _r14logit, _R14_TEST['id'].astype(str).tolist())
""".rstrip()

VALIDATION = """def _r14_validate_comp(meta, logit, ids):
    import numpy as np
    if meta['ids'] != ids:
        raise ValueError('R14 COMP ID sequence mismatch')
    labels = meta['labels']
    if len(labels) != len(set(labels)) or set(labels) != set(_R14_LABELS):
        raise ValueError('R14 COMP label set mismatch')
    if logit.shape != (len(ids), len(_R14_LABELS)) or not np.isfinite(logit).all():
        raise ValueError('R14 COMP invalid logits')
    if not np.isfinite(float(meta['seconds'])) or float(meta['seconds']) < 0:
        raise ValueError('R14 COMP invalid runtime metadata')

def _r14_validate_submission(submission, ids):
    if set(submission.columns) != {'id', 'model'}:
        raise ValueError('R14 submission columns mismatch')
    if submission['id'].isna().any() or not submission['id'].is_unique:
        raise ValueError('R14 submission invalid IDs')
    if submission['id'].astype(str).tolist() != ids:
        raise ValueError('R14 submission ID sequence mismatch')
    if not submission['model'].isin(_R14_LABELS).all():
        raise ValueError('R14 submission illegal label')
"""

INIT = """# R14: one frozen recipe; a normal Save checks packaging only.
import os, json
from pathlib import Path as _R14Path
import pandas as _r14pd
RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None
_R14_DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/test.csv'))
_R14_TEST = _r14pd.read_csv(os.path.join(_R14_DATA, 'test.csv'), usecols=['id'])
if not RERUN and len(_R14_TEST) != 3:
    raise RuntimeError('R14 ordinary Save must use exactly three public placeholder rows')
if _R14_TEST['id'].isna().any() or not _R14_TEST['id'].is_unique or len(_R14_TEST) == 0:
    raise ValueError('R14 invalid input IDs')
for _r14name in ('submission.parquet', 'comp_logit.npy', 'comp_meta.json', 'v31_diagnostics.json'):
    _R14Path(_r14name).unlink(missing_ok=True)
print('R14 competition rerun:', RERUN, '; test rows:', len(_R14_TEST))
"""

PLACEHOLDER = """else:
    _r14placeholder = _r14pd.DataFrame({'id': _R14_TEST['id'], 'model': _R14_LABELS[0]})
    _r14_validate_submission(_r14placeholder, _R14_TEST['id'].astype(str).tolist())
    _r14placeholder.to_parquet('/kaggle/working/submission.parquet', index=False)
    print('R14 placeholder only: three rows; full model runs in the official hidden rerun')
"""


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def inputs_digest(inputs):
    return digest(json.dumps(inputs, sort_keys=True, separators=(',', ':')))


def stripped(notebook):
    notebook = json.loads(json.dumps(notebook))
    for cell in notebook['cells']:
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['metadata'].pop('papermill', None)
    return notebook


def write_notebook(path, notebook):
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    path.write_text(content, encoding='utf-8', newline='\n')
    return digest(content)


def labels_from_source(script):
    tree = ast.parse(script.split('\n', 1)[1])
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LABELS' for t in n.targets))
    labels = ast.literal_eval(node.value)
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError('Expected 18 distinct source labels')
    return labels


def build(reference=REFERENCE):
    notebook = stripped(json.loads(Path(reference).read_bytes()))
    code = [source(c) for c in notebook['cells'] if c['cell_type'] == 'code']
    if [digest(c) for c in code] != CODE_SHA256:
        raise ValueError('V32 source-cell fingerprints differ')
    kaggle = notebook['metadata']['kaggle']
    if inputs_digest(kaggle['dataSources']) != INPUT_SHA256:
        raise ValueError('V32 ordered input references differ')
    if kaggle.get('dockerImageVersionId') != 31499 or kaggle.get('isInternetEnabled') is not False:
        raise ValueError('V32 image/internet configuration differs')
    labels = labels_from_source(code[2])
    header = """# R14: fixed complete-ensemble reproduction of public V32 / 0.660

Exact source: Roman Tamrazov, Kaggle scriptVersionId 363222852 (V32), observed Public Score 0.660. Its internal method heading is V40B; latest Kaggle V33 was 0.658 and is not used.
Our R12 V1 / 363188587 scored 0.654. This new candidate has no score yet and no complete-stack local CV claim.
All three prediction scripts, model settings and 31 pinned inputs are preserved from V32. Operational safeguards check complementary subprocess/output integrity and final IDs/labels. Normal Save writes only the three-row placeholder; the complete fixed pipeline runs during the formal hidden rerun.
At most one saved version and one formal submission; free T4 x2, Internet off. Public scores do not establish private performance or a causal effect of an individual change.
The original Apache 2.0 source attribution is retained below.
"""
    markdown = [c for c in notebook['cells'] if c['cell_type'] == 'markdown']
    if code[4].count(OLD_COMP) != 1:
        raise ValueError('Unexpected complementary launch structure')
    comp = 'if RERUN:\n' + textwrap.indent(code[4].replace(OLD_COMP, NEW_COMP, 1), '    ')
    main = 'if RERUN:\n' + textwrap.indent(code[5], '    ')
    main += "    _r14_validate_submission(submission, _R14_TEST['id'].astype(str).tolist())\n" + PLACEHOLDER
    generated = [INIT + '\n_R14_LABELS = ' + repr(labels) + '\n' + VALIDATION, *code[:4], comp, main]
    notebook['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': header.splitlines(keepends=True)}, *markdown]
    for item in generated:
        compile(item.split('\n', 1)[1] if item.startswith('%%writefile ') else item, '<r14-cell>', 'exec')
        notebook['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': item.splitlines(keepends=True)})
    kaggle['isGpuEnabled'] = True
    notebook['metadata']['r14'] = {'source_version': 363222852, 'external_public_score': '0.660', 'own_public_score': None, 'local_complete_stack_cv': None}
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-download', type=Path)
    args = parser.parse_args()
    if args.freeze_download:
        raw = args.freeze_download.read_bytes()
        if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
            raise ValueError('Downloaded V32 fingerprint differs')
        if REFERENCE.exists():
            raise FileExistsError('Frozen V32 reference already exists')
        write_notebook(REFERENCE, stripped(json.loads(raw)))
    result = build()
    candidate_hash = write_notebook(OUTPUT, result)
    receipt = {
        'source_version': 363222852, 'source_raw_sha256': RAW_SHA256,
        'source_code_sha256': CODE_SHA256, 'source_ordered_inputs_sha256': INPUT_SHA256,
        'candidate_sha256': candidate_hash,
        'candidate_code_sha256': [digest(source(c)) for c in result['cells'] if c['cell_type'] == 'code'],
        'pinned_inputs': result['metadata']['kaggle']['dataSources'], 'requested_container_image': 31499,
        'requested_gpu': 'T4 x2', 'internet_enabled': False,
        'external_public_score': '0.660', 'own_r12_public_score': '0.654',
        'saved_runs_budget': 1, 'formal_submissions_budget': 1,
    }
    (ROOT / 'configs/r14_frozen_recipe_20261029.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'candidate_sha256': candidate_hash, 'code_cells': len(receipt['candidate_code_sha256']), 'input_count': len(receipt['pinned_inputs'])}))


if __name__ == '__main__':
    main()
