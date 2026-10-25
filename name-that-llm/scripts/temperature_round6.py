"""Training-only scalar temperature fitting; no data access or candidate selection."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


def valid_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 3 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid multiclass scores")
    return scores - scores.max(axis=1, keepdims=True)


def nll(scores, truth, inverse_temperature=1.0):
    scores = valid_scores(scores)
    truth = np.asarray(truth)
    if truth.shape != (len(scores),) or truth.dtype.kind not in "iu" or np.any(truth < 0) or np.any(truth >= scores.shape[1]):
        raise ValueError("Invalid calibration labels")
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = scores * inverse_temperature
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(truth)), truth]))


def fit_temperature(scores, truth, settings):
    scores = valid_scores(scores)
    baseline = nll(scores, truth)
    evaluations = 0
    def objective(beta):
        nonlocal evaluations
        evaluations += 1
        if evaluations > settings["max_objective_evaluations"]:
            raise RuntimeError("Scalar calibration evaluation budget exhausted")
        return nll(scores, truth, beta)
    result = minimize_scalar(objective, bounds=tuple(settings["inverse_temperature_bounds"]), method="bounded",
                             options={"maxiter": settings["max_objective_evaluations"], "xatol": settings["xatol"]})
    if not result.success or not np.isfinite(result.x) or result.x <= 0 or result.fun > baseline + 1e-8:
        raise ValueError("Temperature optimization failed; no fallback")
    return {"temperature": float(1/result.x), "inverse_temperature": float(result.x),
            "training_nll_before": baseline, "training_nll_after": float(result.fun),
            "objective_evaluations": evaluations, "training_rows":len(scores)}


def calibrated_probabilities(scores, calibration):
    scores = valid_scores(scores)
    beta = calibration["inverse_temperature"]
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Invalid saved inverse temperature")
    result = softmax(scores * beta, axis=1)
    if not np.array_equal(result.argmax(1), scores.argmax(1)):
        raise ValueError("Positive scalar calibration changed class ranking")
    return result


def gaussian_scores(features, model):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]) or not np.isfinite(x).all():
        raise ValueError("Invalid Gaussian features")
    return valid_scores(((x-model["mean"])/model["scale"]) @ model["coefficients"] + model["intercept"])


def nested_folds(data, outer_train, outer_valid, splits=3):
    from sklearn.model_selection import GroupKFold
    outer_train, outer_valid = np.asarray(outer_train), np.asarray(outer_valid)
    groups, truth = data["groups"], data["truth"]
    if set(outer_train) & set(outer_valid) or set(groups[outer_train]) & set(groups[outer_valid]):
        raise ValueError("Outer split overlap")
    seen = np.zeros(len(outer_train), dtype=int)
    folds, identity = [], np.full(len(outer_train), -1, dtype=np.int64)
    for number, (tr, va) in enumerate(GroupKFold(n_splits=splits).split(outer_train, truth[outer_train], groups[outer_train])):
        train, valid = outer_train[tr], outer_train[va]
        if (set(groups[train]) & set(groups[valid]) or set(groups[train]) & set(groups[outer_valid])
                or set(groups[valid]) & set(groups[outer_valid]) or set(truth[train]) != set(range(len(data["labels"])))):
            raise ValueError("Invalid nested group isolation or class coverage")
        seen[va] += 1
        identity[va] = number
        folds.append((train, valid, va))
    if not np.all(seen == 1):
        raise ValueError("Incomplete nested OOF")
    return folds, identity
