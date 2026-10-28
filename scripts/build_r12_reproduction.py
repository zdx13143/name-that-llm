"""Freeze the scored public V4 recipe and apply only operational safeguards."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'notebooks/reference_public654_v4.ipynb'
RAW_SHA256 = '49ff9e5feb235a632d03646007d61bf37d8e7971f655c846aae63a6f64024874'
CODE_SHA256 = [
    '221bd0793a59d264bc7cdeb018efb2351751485b571e3744b86d6fd9f08ede25',
    '4fc27dca93a92adfb2759ca65260e5285898d73fc8ccd1acd401982e5f98ed37',
    'b53e305fd7a755d138f931c55ac5f3762ed3978d73a65594b5ce2523fdf165ff',
    '3216840a23bafb6d79f90cf2a9fb63be715210247ddbaa1dd88814eceece8bbb',
    'f1bb8c65e926f0fde39e384cd500aa98162527492251dd63651e3f78b29232bf',
    '4dfd75235f678516647e31a4cfd39e5f035e37510ba2076c04f22a2f2c0853d6',
    '70b7a5f07e292d7e0fe8fcb496e04fabc02123a3951637e41ed24c8de14f5af7',
]
OLD_SIGNAL = 'prompt-heldout validated Bayesian typography'
NEW_SIGNAL = 'fixed public V4 reproduction; no local complete-ensemble OOF claim'
OLD_COMP = "    !python comp_stack.py 2>&1 | grep -vE 'Warning|warn'"
NEW_COMP = '''    import subprocess, sys
    subprocess.run([sys.executable, 'comp_stack.py'], check=True)
    import numpy as _r12np, pandas as _r12pd, json as _r12json
    _r12test = _r12pd.read_csv(os.path.join(DATA, 'test.csv'), usecols=['id'])
    _r12meta = _r12json.load(open('/kaggle/working/comp_meta.json'))
    _r12logit = _r12np.load('/kaggle/working/comp_logit.npy', allow_pickle=False)
    assert _r12meta['ids'] == _r12test['id'].astype(str).tolist(), 'R12 COMP ID mismatch'
    assert len(_r12meta['labels']) == len(set(_r12meta['labels'])) == 18, 'R12 COMP class mismatch'
    assert _r12logit.shape == (len(_r12test), 18) and _r12np.isfinite(_r12logit).all(), 'R12 COMP invalid output'
'''.rstrip()
INIT = '''# R12: discard only this notebook's previous generated outputs before setup.
from pathlib import Path as _R12Path
for _r12name in ('submission.parquet', 'comp_logit.npy', 'comp_meta.json',
                 'candidate_reference_0651.parquet', 'candidate_typobayes_full.parquet', 'v38a_diagnostics.json'):
    _R12Path(_r12name).unlink(missing_ok=True)
'''


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def stripped(nb):
    nb = json.loads(json.dumps(nb))
    for cell in nb['cells']:
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    nb['metadata'].pop('papermill', None)
    return nb


def write_notebook(path, nb):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
    path.write_text(content, encoding='utf-8', newline='\n')
    return hashlib.sha256(content.encode()).hexdigest()


def build(reference=REFERENCE):
    nb = stripped(json.loads(Path(reference).read_bytes()))
    code = [source(c) for c in nb['cells'] if c['cell_type'] == 'code']
    hashes = [hashlib.sha256(c.encode()).hexdigest() for c in code]
    if hashes != CODE_SHA256:
        raise RuntimeError('Public V4 code differs from pinned seven-cell source')
    assert code[3].count(OLD_SIGNAL) == 1 and code[5].count(OLD_COMP) == 1
    code[3] = code[3].replace(OLD_SIGNAL, NEW_SIGNAL, 1)
    code[5] = code[5].replace(OLD_COMP, NEW_COMP, 1)
    header = '''# R12: fixed complete-ensemble reproduction

Source: alexchilton V4, scriptVersion 363071115, publicly scored 0.654; based on Roman Tamrazov and Dhruv Tibarewal's full 0.651 recipe.
This is a reproduction, not an original model or a new local CV result. The observed external improvement is only 0.003 and may not reproduce.
All main models, score operations and parameters are preserved. Changes only check the complementary subprocess, clear stale generated files and correct an unsupported diagnostic claim.
The normal save writes a three-row placeholder; the real full ensemble runs only in the formal competition rerun.
All upstream authors and assets remain credited in the frozen reference notebook. Apache 2.0 source attribution is retained.
'''
    nb['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': header.splitlines(keepends=True)}]
    for item in [INIT] + code:
        compiled = item.split('\n', 1)[1] if item.startswith('%%writefile ') else item
        compile(compiled, '<r12-cell>', 'exec')
        nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': item.splitlines(keepends=True)})
    nb['metadata']['r12'] = {'source_version': 363071115, 'source_code_sha256': CODE_SHA256,
                              'external_public_score': '0.654', 'local_cv': None}
    return nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-download', type=Path)
    args = parser.parse_args()
    if args.freeze_download:
        raw = args.freeze_download.read_bytes()
        if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
            raise RuntimeError('Downloaded V4 hash differs')
        if REFERENCE.exists():
            raise RuntimeError('Frozen reference already exists')
        write_notebook(REFERENCE, stripped(json.loads(raw)))
    result = build()
    out = ROOT / 'notebooks/team_e651_r12_public654.ipynb'
    sha = write_notebook(out, result)
    receipt = {
        'source_version': 363071115, 'source_raw_sha256': RAW_SHA256,
        'source_code_sha256': CODE_SHA256, 'candidate_sha256': sha,
        'candidate_code_sha256': [hashlib.sha256(source(c).encode()).hexdigest() for c in result['cells'] if c['cell_type'] == 'code'],
        'pinned_inputs': result['metadata']['kaggle']['dataSources'],
        'container_image': result['metadata']['kaggle'].get('dockerImageVersionId'),
        'saved_runs_budget': 1, 'formal_submissions_budget': 1,
    }
    receipt_path = ROOT / 'configs/r12_frozen_recipe_20261028.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'output': str(out), 'sha256': sha, 'input_count': len(receipt['pinned_inputs'])}, indent=2))


if __name__ == '__main__':
    main()
