"""Build the R26 notebook from the saved v15 source without reading competition data."""
import base64
import copy
import difflib
import hashlib
import json
from pathlib import Path

try:
    from scripts import build_r23_notebook as r23
except ImportError:
    import build_r23_notebook as r23

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'kaggle/r26'
SOLVER = ROOT / 'scripts/r24_assignment.py'

RULE = r'''    # R26_RULE_BEGIN
    import os as _r26_os, sys as _r26_sys, json as _r26_json
    _r26_sys.path.insert(0, '/kaggle/working')
    import r24_assignment as _r26_assignment
    try:
        _L = np.asarray(_safe_logits, dtype=float)
        _groups = np.asarray(Gte)
        _ids = np.asarray(test.id.values)
        _expected_labels = ('claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5','gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max')
        if _L.ndim != 2 or _L.shape[0] == 0 or _L.shape[1] != 18 or not np.isfinite(_L).all() or tuple(LABELS) != _expected_labels:
            raise ValueError('R26 invalid logits or label contract')
        if _groups.ndim != 1 or len(_groups) != len(_L) or pd.isna(_groups).any() or _ids.ndim != 1 or len(_ids) != len(_L) or pd.isna(_ids).any() or not pd.Index(_ids).is_unique:
            raise ValueError('R26 invalid groups or ids')
        if _groups.dtype.kind in 'fc' and not np.isfinite(_groups).all():
            raise ValueError('R26 nonfinite groups')
        _sizes = np.unique(_groups, return_counts=True)[1]
        _sorted_logits = np.sort(_L, axis=1)
        _gaps = _sorted_logits[:, -1] - _sorted_logits[:, -2]
        _positive = _gaps[_gaps > 0]
        if not len(_positive):
            raise ValueError('R26 no positive top-two gaps')
        _scale = float(np.median(_positive))
        if not np.isfinite(_scale) or _scale <= 0:
            raise ValueError('R26 invalid scale')
        _z = (_L - _L.max(axis=1, keepdims=True)) / _scale
        if not np.isfinite(_z).all():
            raise ValueError('R26 invalid normalized logits')
        _before = _L.argmax(axis=1)
        _after = np.asarray(_r26_assignment.predict(_z, _groups, 1.0))
        if _after.shape != (len(_L),) or _after.dtype.kind not in 'iu' or np.any((_after < 0) | (_after >= 18)):
            raise ValueError('R26 invalid predictions')
        _safe = pd.DataFrame({'id': _ids, 'model': [LABELS[int(i)] for i in _after]})
        _tmp = _out.with_name(_out.name + '.r26.tmp')
        def _r26_verify(_path):
            _check = pd.read_parquet(_path)
            if list(_check.columns) != ['id', 'model'] or len(_check) != len(_ids) or not np.array_equal(_check.id.values, _ids) or not np.array_equal(_check.model.values, _safe.model.values):
                raise ValueError('R26 parquet verification failed')
        _safe.to_parquet(_tmp, index=False)
        _r26_verify(_tmp)
        _r26_os.replace(_tmp, _out)
        _r26_verify(_out)
        _hist = {str(int(k)): int(( _sizes == k).sum()) for k in np.unique(_sizes)}
        _eligible = (_sizes >= 12) & (_sizes <= 18)
        _diag = {'alpha': 1.0, 'beta': 0, 'scale': _scale, 'cluster_size_histogram': _hist, 'eligible_clusters': int(_eligible.sum()), 'eligible_rows': int(_sizes[_eligible].sum()), 'changed_rows': int((_after != _before).sum()), 'pre_class_counts': np.bincount(_before, minlength=18).tolist(), 'post_class_counts': np.bincount(_after, minlength=18).tolist()}
        with open('/kaggle/working/r26_diagnostics.json', 'w') as _f:
            _r26_json.dump(_diag, _f, sort_keys=True, allow_nan=False)
        print('R26_RULE ' + _r26_json.dumps(_diag, sort_keys=True, allow_nan=False), flush=True)
    except BaseException:
        _out.unlink(missing_ok=True)
        _out.with_name(_out.name + '.r26.tmp').unlink(missing_ok=True)
        raise
    # R26_RULE_END
'''

SELFTEST = r'''# R26 synthetic self-test, executed even in FAST_COMMIT
import base64, hashlib, itertools, pathlib, sys
_r26_bytes = base64.b64decode(SOLVER_BASE64, validate=True)
if hashlib.sha256(_r26_bytes).hexdigest() != SOLVER_SHA256:
    raise RuntimeError('R26 embedded solver SHA256 mismatch')
_r26_path = pathlib.Path('/kaggle/working/r24_assignment.py')
_r26_path.write_bytes(_r26_bytes)
if hashlib.sha256(_r26_path.read_bytes()).hexdigest() != SOLVER_SHA256:
    raise RuntimeError('R26 solver file SHA256 mismatch')
sys.path.insert(0, str(_r26_path.parent))
import r24_assignment as _r26_assignment
import numpy as np
for _z, _alpha in [(np.array([[0., 0.], [0., 0.], [0., 0.]]), 1.), (np.array([[2., 0.], [2., 1.], [1., 0.]]), 1.), (np.array([[0., 1., 0.], [0., 1., 0.]]), 0.5)]:
    _candidates = list(itertools.product(range(_z.shape[1]), repeat=len(_z)))
    _values = [_r26_assignment.objective(_z, np.array(_x), _alpha) for _x in _candidates]
    _best = max(_values)
    _tol = 8 * np.finfo(float).eps * max(1., abs(_best), len(_z) * np.max(np.abs(_z)), len(_z) * _alpha)
    assert tuple(_r26_assignment.solve(_z, _alpha)) == next(_x for _x, _v in zip(_candidates, _values) if _v >= _best - _tol)
_sizes = [11, 12, 18, 19]
_groups = np.repeat(np.arange(4), _sizes)
_z = np.zeros((sum(_sizes), 18)); _z[:, 0] = 0.1
_pred = _r26_assignment.predict(_z, _groups, 1.0)
for _g, _size in enumerate(_sizes):
    _p = _pred[_groups == _g]
    assert (len(set(_p)) > 1) == (12 <= _size <= 18)
    if _size not in (12, 18): assert np.all(_p == 0)
_scale_sample = np.array([[3., 2.], [5., 5.], [9., 5.]])
_top = np.sort(_scale_sample, axis=1)
_gaps = _top[:, -1] - _top[:, -2]
assert np.median(_gaps[_gaps > 0]) == 2.5
try:
    _r26_assignment.solve(np.array([[float('nan'), 0.]]), 1.)
except ValueError:
    pass
else:
    raise AssertionError('R26 nonfinite input accepted')
compile(pathlib.Path('infer.py').read_text(), 'infer.py', 'exec')
print('R26 synthetic self-test and infer.py compile PASS')
'''


def build(source_dir=r23.SOURCE_DIR, out=OUT):
    source_dir, out = Path(source_dir), Path(out)
    nb = json.loads((source_dir / 'ntl-we-back-0-666.ipynb').read_text(encoding='utf-8'))
    original = copy.deepcopy(nb)
    r23.replace_once(nb, 'import glob, os, subprocess, json, sys', r23.MANIFEST + '\nimport glob, os, subprocess, json, sys')
    r23.replace_once(nb, "_op = _pd.read_parquet(_gl.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)[0], columns=['text', 'model', 'match_weight'])", r23.POOL)
    pip = "if whl: print(subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)"
    r23.replace_once(nb, pip, "if whl:\n    _r23_pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True)\n    print(_r23_pip.stdout, end=''); print(_r23_pip.stderr, end='', file=sys.stderr); print(_r23_pip.returncode)\n    _r23_pip.check_returncode()")
    old_write = "sub.to_parquet(a.out, index=False); log('wrote', a.out, sub.model.value_counts().to_dict())"
    r23.replace_once(nb, old_write, "if a.sim:\n    sub.to_parquet(a.out, index=False); log('wrote', a.out, sub.model.value_counts().to_dict())\nelse:\n    from pathlib import Path as _R26Path\n    _R26Path(a.out).unlink(missing_ok=True)")
    old = "    _safe = _v31_sub(_safe_logits)\n    _safe.to_parquet(_out, index=False)\n    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()\n"
    r23.replace_once(nb, old, RULE)
    r23.replace_once(nb, "'pre_bayes_changed': int((_raw_sub.model.values != _safe.model.values).sum())", "'pre_bayes_changed': int((_raw_sub.model.values != _v31_sub(_safe_logits).model.values).sum())")
    r23.replace_once(nb, "log('V31 Bayesian typography completed; changed from pre-Bayes:', int((_raw_sub.model.values != _safe.model.values).sum()))", "log('V31 Bayesian typography completed; changed from pre-Bayes:', int((_raw_sub.model.values != _v31_sub(_safe_logits).model.values).sum()))")
    b = SOLVER.read_bytes()
    test = SELFTEST.replace('SOLVER_BASE64', repr(base64.b64encode(b).decode())).replace('SOLVER_SHA256', repr(hashlib.sha256(b).hexdigest()))
    idx = next(i for i, c in enumerate(nb['cells']) if ''.join(c.get('source', [])).startswith('%%writefile infer.py'))
    nb['cells'].insert(idx + 1, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': test.splitlines(keepends=True)})
    meta = json.loads((source_dir / 'kernel-metadata.json').read_text(encoding='utf-8'))
    for old_source, new_source in [(r23.OLD_POOL, r23.NEW_POOL), (r23.OLD_GEN, r23.NEW_GEN)]:
        if meta['dataset_sources'].count(old_source) != 1: raise ValueError('metadata source anchor')
        meta['dataset_sources'] = [new_source if s == old_source else s for s in meta['dataset_sources']]
    meta.update(id='zephyrpong/ntl-r26-v15-assign', title='NTL R26 v15 assignment', code_file='ntl-r26-v15-assign.ipynb', is_private=True, enable_internet=False)
    meta.pop('id_no', None)
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    diff = difflib.unified_diff(r23.notebook_text(original).splitlines(keepends=True), r23.notebook_text(nb).splitlines(keepends=True), fromfile='v15/source', tofile='r26/notebook')
    (out / 'source.diff').write_text(''.join(diff), encoding='utf-8', newline='\n')

if __name__ == '__main__':
    build()
