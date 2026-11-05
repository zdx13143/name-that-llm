"""Exact soft duplicate penalty and deterministic lexicographic optimum."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def scale_scores(scores, folds):
    scores = np.asarray(scores, dtype=float)
    z = np.empty_like(scores)
    scales = {}
    for f in np.unique(folds):
        mask = folds == f
        s = scores[mask]
        top = np.partition(s, -2, axis=1)[:, -2:]
        gaps = top[:, 1] - top[:, 0]
        positive = gaps[gaps > 0]
        if not len(positive):
            raise ValueError('no positive top-two gaps')
        scale = float(np.median(positive))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('invalid score scale')
        z[mask] = (s - s.max(axis=1, keepdims=True)) / scale
        scales[int(f)] = scale
    return z, scales


def objective(z, labels, alpha):
    counts = np.bincount(labels, minlength=z.shape[1])
    return float(z[np.arange(len(z)), labels].sum() - alpha * np.maximum(counts - 1, 0).sum())


def solve(z, alpha):
    z = np.asarray(z, dtype=float)
    n, classes = z.shape
    if not np.isfinite(z).all() or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('invalid assignment input')
    if not n:
        return np.empty(0, dtype=int)
    if alpha == 0:
        return z.argmax(axis=1)
    # Class-major slots; first occurrence is free, every later occurrence costs alpha.
    slots = np.repeat(np.arange(classes), n)
    costs = np.tile(np.r_[0., np.full(n-1, alpha)], classes)
    weights = z[:, slots] - costs
    rows, cols = linear_sum_assignment(-weights)
    optimum = float(weights[rows, cols].sum())
    chosen = []; used = np.zeros(classes, dtype=int); prefix = 0.
    # Fix rows in source order to the smallest class retaining the optimum.
    # Allow only floating-point summation error when comparing objective values.
    tol = 8 * np.finfo(float).eps * max(1., abs(optimum), n * np.max(np.abs(z)), n * alpha)
    for i in range(n):
        for c in range(classes):
            increment = z[i, c] - (alpha if used[c] else 0.)
            trial = used.copy(); trial[c] += 1
            remain = n-i-1
            if remain:
                penalty = np.concatenate([np.full(remain, alpha) if k else
                                          np.r_[0., np.full(remain-1, alpha)] for k in trial])
                w = z[i+1:, np.repeat(np.arange(classes), remain)] - penalty
                rr, cc = linear_sum_assignment(-w)
                value = prefix + increment + float(w[rr, cc].sum())
            else:
                value = prefix + increment
            if value >= optimum - tol:
                chosen.append(c); used[c] += 1; prefix += increment
                break
        else:
            raise RuntimeError('assignment tie resolution failed')
    return np.array(chosen, dtype=int)


def predict(z, groups, alpha, check=lambda: None):
    result = z.argmax(axis=1)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if 12 <= len(idx) <= 18:
            check()
            result[idx] = solve(z[idx], alpha)
    return result
