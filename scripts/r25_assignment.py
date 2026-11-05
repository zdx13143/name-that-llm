"""R25 targeted score correction, followed by the unchanged R24 assignment."""
import numpy as np
from r24_assignment import predict as r24_predict


def biased_scores(z, beta, a, b):
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).all() or not np.isfinite(beta) or beta < 0:
        raise ValueError('invalid bias input')
    if not (0 <= a < z.shape[1] and 0 <= b < z.shape[1] and a != b):
        raise ValueError('invalid class indices')
    eligible = z.argmax(axis=1) == b
    result = z.copy()
    result[eligible, a] += beta
    return result


def predict(z, clusters, alpha, beta, a, b, check=lambda: None):
    return r24_predict(biased_scores(z, beta, a, b), clusters, alpha, check)
