"""Fixed label-free probability balancing, inspired by ntl-e642 v2.

Reference: https://www.kaggle.com/code/zdx13143/ntl-e642?scriptVersionId=362382146
Only the final matrix-scaling operation is reproduced, not the full ensemble.
"""
from __future__ import annotations

import numpy as np


def balance(probabilities):
    """Balance an entire deployment batch; no labels, IDs or groups are inputs."""
    p = np.array(probabilities, dtype=np.float64, copy=True)
    if p.ndim != 2 or p.shape[1] != 18 or not len(p) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected a finite nonnegative nonempty N by 18 matrix")
    p = np.maximum(p, 1e-300)
    if len(p) < 50:
        return p / p.sum(1, keepdims=True)
    for _ in range(300):
        p *= (len(p) / 18) / p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return p


def metrics(probabilities, truth, labels):
    y = np.asarray(truth)
    pred = np.asarray(probabilities).argmax(1)
    if y.shape != pred.shape or y.dtype.kind not in "iu" or (y < 0).any() or (y >= 18).any():
        raise ValueError("Invalid development truth")
    confusion = np.bincount(y * 18 + pred, minlength=18 * 18).reshape(18, 18)
    counts = confusion.sum(1)
    if (counts == 0).any():
        raise ValueError("All 18 classes must be represented in evaluation")
    recall = confusion.diagonal() / counts
    return {"balanced_accuracy": float(recall.mean()), "accuracy": float((pred == y).mean()),
            "correct": int((pred == y).sum()), "rows": len(y),
            "per_class": [{"label": str(label), "support": int(counts[i]),
                           "correct": int(confusion[i, i]), "recall": float(recall[i])}
                          for i, label in enumerate(labels)],
            "confusion_matrix": confusion.tolist(), "predicted_counts": confusion.sum(0).tolist()}


def paired_comparison(candidate, baseline, truth, groups, *, resamples, seed):
    """Descriptive prompt bootstrap conditional on already fixed predictions."""
    y = np.asarray(truth)
    current, previous = candidate.argmax(1), baseline.argmax(1)
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = inverse.max() + 1
    counts = np.zeros((n_groups, 18), dtype=np.int64)
    current_correct, previous_correct = counts.copy(), counts.copy()
    np.add.at(counts, (inverse, y), 1)
    np.add.at(current_correct, (inverse, y), current == y)
    np.add.at(previous_correct, (inverse, y), previous == y)
    difference = current_correct - previous_correct
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        selected = rng.integers(0, n_groups, n_groups)
        denominator = counts[selected].sum(0)
        if (denominator == 0).any():
            raise ValueError("Bootstrap sample omitted a class")
        samples[i] = np.mean(difference[selected].sum(0) / denominator)
    return {"gain": float(np.mean(difference.sum(0) / counts.sum(0))),
            "descriptive_prompt_95ci": np.quantile(samples, [.025, .975]).tolist(),
            "changed_predictions": int((current != previous).sum()),
            "corrected": int(((current == y) & (previous != y)).sum()),
            "harmed": int(((current != y) & (previous == y)).sum()),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed}


def gate(candidate, folds, comparison, settings):
    checks = {
        "pooled": candidate["balanced_accuracy"] >= settings["min_pooled_balanced_accuracy"],
        "each_fold": all(f["candidate_ba"] >= settings["min_each_fold_balanced_accuracy"] for f in folds),
        "gain": comparison["gain"] >= settings["min_gain_over_r7"],
        "paired_lower_bound": comparison["descriptive_prompt_95ci"][0] > settings["paired_prompt_95ci_lower_exceeds"],
        "improved_folds": sum(f["candidate_ba"] > f["baseline_ba"] for f in folds) >= settings["min_improved_folds"],
    }
    return {"passed": all(checks.values()), "checks": checks}
