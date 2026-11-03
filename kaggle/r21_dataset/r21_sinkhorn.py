"""Execute the CPU-only blend and Sinkhorn blocks extracted from notebooks."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

SOURCE = Path(__file__).resolve().parent / 'r21_source'


def balance(members, y_train, condition):
    """Return raw blend and balanced log scores in the notebook label order.

    ``members`` contains already weighted 18-column log scores. Pass a single
    validation fold; the prior must use only the matching training fold.
    """
    if condition not in ('r16', 'areeb'):
        raise ValueError('unknown source')
    lp = {name: np.asarray(value, dtype=np.float64) for name, value in members.items()}
    if set(lp) != {'tfidf', 'svc4', 'distort'}:
        raise ValueError('three CPU members required')
    shape = lp['tfidf'].shape
    if len(shape) != 2 or shape[1] != 18 or shape[0] < 18:
        raise ValueError('expected at least 18 rows and 18 classes')
    if any(v.shape != shape or not np.isfinite(v).all() for v in lp.values()):
        raise ValueError('invalid member scores')
    y = np.asarray(y_train)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or not len(y) or np.any((y < 0) | (y >= 18)):
        raise ValueError('invalid training labels')
    env = {'np': np, 'LP': lp, 'y': y, 'test': range(shape[0]),
           '_bundles': [], 'COMP_P': None,
           'a': SimpleNamespace(sinkhorn=.45 if condition == 'r16' else .30,
                                sinkhorn_target_mix=.35, sk_div=1)}
    blend_path = SOURCE / (condition + '_blend.py.txt')
    exec(compile(blend_path.read_text(encoding='utf-8'), str(blend_path), 'exec'), env)
    raw = env['tot'].copy()
    code = (SOURCE / (condition + '_sinkhorn.py.txt')).read_text(encoding='utf-8')
    exec(compile(code, str(SOURCE / (condition + '_sinkhorn.py.txt')), 'exec'), env)
    balanced = env['tot']
    if not np.isfinite(balanced).all():
        raise ValueError('nonfinite balanced scores')
    return raw, balanced
