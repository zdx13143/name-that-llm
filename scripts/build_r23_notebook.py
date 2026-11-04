"""Rebuild R23 from the saved v15 notebook; never reads competition data."""
import copy
import difflib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(r"D:/kaggle/name_that_llm/outputs/codex_runs/r21-plan/sources/areebbb__ntl-we-back-0-666")
OUT = ROOT / "kaggle/r23"
OLD_POOL = "crtproduction/name-that-llm-open-pool"
NEW_POOL = "zephyrpong/ntl-r21-open-pool-v11"
OLD_GEN = "abdullah1afify/ntl-generations"
NEW_GEN = "zephyrpong/ntl-r23-generations-v9"  # exact v9 bytes, current at v15 submission 57915690

RULE = r'''def r23_rule(tot, labels, pred, perpassage, post):
    import numpy as np, json
    if perpassage != 1 or post != 0:
        raise ValueError("R23 requires perpassage=1 and post=0")
    if len(labels) != 18 or labels.index("gemini-3.7-flash") != 7 or labels.index("gemini-3.8-flash") != 8:
        raise ValueError("R23 label order changed")
    tot = np.asarray(tot)
    pred = np.asarray(pred)
    if tot.ndim != 2 or tot.shape[1] != 18 or pred.shape != (len(tot),) or not np.isfinite(tot).all():
        raise ValueError("R23 invalid or nonfinite scores")
    if not np.array_equal(pred, tot.argmax(1)):
        raise ValueError("R23 expected per-row argmax")
    before = pred.copy()
    eligible = before == 8
    gaps = tot[eligible, 8] - tot[eligible, 7]
    if not np.isfinite(gaps).all():
        raise ValueError("R23 nonfinite gaps")
    pred = before.copy()
    pred[eligible & ((tot[:, 8] - tot[:, 7]) < 1.0)] = 7
    changed = pred != before
    if not np.all((before[changed] == 8) & (pred[changed] == 7)):
        raise ValueError("R23 unexpected prediction change")
    b = np.bincount(before, minlength=18)
    c = np.bincount(pred, minlength=18)
    expected = b.copy(); expected[7] += int(changed.sum()); expected[8] -= int(changed.sum())
    if not np.array_equal(c, expected):
        raise ValueError("R23 counts do not reconcile")
    q = np.quantile(gaps, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]).tolist() if len(gaps) else [None] * 6
    diag = {"n_rows": len(tot), "eligible_3_8": int(eligible.sum()), "switched": int(changed.sum()), "gap_quantiles": dict(zip(["5%", "10%", "25%", "50%", "75%", "90%"], q)), "counts_before": b.tolist(), "counts_after": c.tolist(), "all_finite": True}
    print("R23_RULE " + json.dumps(diag, allow_nan=False, sort_keys=True), flush=True)
    return pred

pred = r23_rule(tot, LABELS, pred, a.perpassage, a.post)
'''

MANIFEST = r'''# R23: inventory mounted inputs before any notebook input reads.
import os as _r23_os, hashlib as _r23_hashlib, json as _r23_json
_r23_files = []
for _r23_dir, _r23_dirs, _r23_names in _r23_os.walk('/kaggle/input', followlinks=True):
    _r23_dirs.sort()
    for _r23_name in sorted(_r23_names):
        _r23_path = _r23_os.path.join(_r23_dir, _r23_name)
        if not _r23_os.path.isfile(_r23_path):
            continue
        _r23_size = _r23_os.path.getsize(_r23_path)
        _r23_entry = {'path': _r23_path, 'size': _r23_size, 'sha256': None}
        if _r23_size <= 200_000_000:
            _r23_h = _r23_hashlib.sha256()
            with open(_r23_path, 'rb') as _r23_f:
                for _r23_chunk in iter(lambda: _r23_f.read(8 * 1024 * 1024), b''):
                    _r23_h.update(_r23_chunk)
            _r23_entry['sha256'] = _r23_h.hexdigest()
        _r23_files.append(_r23_entry)
_r23_manifest = {'hash_cap_bytes': 200000000, 'files': _r23_files}
with open('/kaggle/working/r23_manifest.json', 'w') as _r23_f:
    _r23_json.dump(_r23_manifest, _r23_f, sort_keys=True)
print('R23_MANIFEST ' + _r23_json.dumps(_r23_manifest, sort_keys=True), flush=True)
_r23_gen = [f for f in _r23_files if f['path'].endswith('/ntl_generations.csv')]
if len(_r23_gen) != 1 or _r23_gen[0]['sha256'] != '33a47a1dda5ecc9c3d155d581afe0cb3851433d60deb87f209993162d5d197a5':
    raise RuntimeError('R23 ntl_generations.csv must be the single pinned v9 copy')
'''
POOL = r'''import hashlib as _r23_hashlib, os as _r23_os
_r23_candidates = ['/kaggle/input/datasets/zephyrpong/ntl-r21-open-pool-v11/ntl_open_pool.parquet', '/kaggle/input/ntl-r21-open-pool-v11/ntl_open_pool.parquet']
_r23_paths = {_r23_os.path.realpath(p) for p in _r23_candidates if _r23_os.path.isfile(p)}
if len(_r23_paths) != 1:
    raise RuntimeError('R23 private Open Pool missing or ambiguous')
_r23_pool = next(iter(_r23_paths))
_r23_digest = _r23_hashlib.sha256()
with open(_r23_pool, 'rb') as _r23_f:
    for _r23_chunk in iter(lambda: _r23_f.read(8 * 1024 * 1024), b''):
        _r23_digest.update(_r23_chunk)
if _r23_digest.hexdigest() != '38e48b40e1db0ab1bf6e4f29d3697e37307c49d1ae5020a7af3cd59c484657d5':
    raise RuntimeError('R23 private Open Pool SHA256 mismatch')
_op = _pd.read_parquet(_r23_pool, columns=['text', 'model', 'match_weight'])'''

def replace_once(nb, old, new):
    cells = nb['cells']
    count = sum(''.join(c.get('source', [])).count(old) for c in cells)
    if count != 1:
        raise ValueError(f'anchor must occur exactly once; found {count}: {old[:80]!r}')
    for cell in cells:
        src = ''.join(cell.get('source', []))
        if old in src:
            cell['source'] = src.replace(old, new, 1).splitlines(keepends=True)
            return

def notebook_text(nb):
    return ''.join(f"### CELL {i} ({c['cell_type']})\n" + ''.join(c.get('source', [])) + '\n' for i, c in enumerate(nb['cells']))

def build(source_dir=SOURCE_DIR, out=OUT):
    source_dir, out = Path(source_dir), Path(out)
    nb = json.loads((source_dir / 'ntl-we-back-0-666.ipynb').read_text(encoding='utf-8'))
    original = copy.deepcopy(nb)
    anchor = 'pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)'
    replace_once(nb, anchor, anchor + '\n' + RULE.rstrip())
    anchor = "import glob, os, subprocess, json, sys"
    replace_once(nb, anchor, MANIFEST + '\n' + anchor)
    anchor = "_op = _pd.read_parquet(_gl.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)[0], columns=['text', 'model', 'match_weight'])"
    replace_once(nb, anchor, POOL)
    anchor = "if whl: print(subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)"
    replace_once(nb, anchor, "if whl:\n    _r23_pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True)\n    print(_r23_pip.stdout, end=''); print(_r23_pip.stderr, end='', file=sys.stderr); print(_r23_pip.returncode)\n    _r23_pip.check_returncode()")
    meta = json.loads((source_dir / 'kernel-metadata.json').read_text(encoding='utf-8'))
    if meta['dataset_sources'].count(OLD_POOL) != 1:
        raise ValueError('Open Pool metadata anchor must occur exactly once')
    if meta['dataset_sources'].count(OLD_GEN) != 1:
        raise ValueError('ntl-generations metadata anchor must occur exactly once')
    meta['dataset_sources'] = [{OLD_POOL: NEW_POOL, OLD_GEN: NEW_GEN}.get(x, x) for x in meta['dataset_sources']]
    meta.update(id='zephyrpong/ntl-r23-v15-pairwise', title='NTL R23 v15 pairwise', code_file='ntl-r23-v15-pairwise.ipynb', is_private=True, enable_internet=False)
    meta.pop('id_no', None)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ntl-r23-v15-pairwise.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    diff = difflib.unified_diff(notebook_text(original).splitlines(keepends=True), notebook_text(nb).splitlines(keepends=True), fromfile='v15/source', tofile='r23/notebook')
    (out / 'source.diff').write_text(''.join(diff), encoding='utf-8', newline='\n')

if __name__ == '__main__':
    build()
