"""Within-prompt PCA and a single-row Gaussian nuisance head.

Mathematical design inspired by the shared-prompt Gaussian attributed to
prvsiyan in the repository's two-stage LoRA Gaussian reference notebook.
This implementation has no clustering, assignment, group inference or encoder.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.special import softmax
from sklearn.covariance import ledoit_wolf_shrinkage


def grouped_means(values, inverse, counts):
    result = np.zeros((len(counts), values.shape[1]), dtype=np.float64)
    np.add.at(result, inverse, values)
    return result / counts[:, None]


def shrink(covariance, strength, target_trace=None):
    trace = float(np.trace(covariance)) if target_trace is None else target_trace
    return (1 - strength) * covariance + strength * trace / len(covariance) * np.eye(len(covariance))


def fit_head(features, truth, groups, settings, classes=18):
    x = np.asarray(features, dtype=np.float64)
    y, groups = np.asarray(truth), np.asarray(groups)
    if (x.ndim != 2 or y.shape != (len(x),) or groups.shape != y.shape
            or y.dtype.kind not in "iu" or set(y) != set(range(classes))
            or not np.isfinite(x).all()):
        raise ValueError("Invalid training-only feature/label/group inputs")
    _, gi, ng = np.unique(groups, return_inverse=True, return_counts=True)
    n, width = x.shape
    g = len(ng)
    dof = n - g - classes + 1
    dimensions = min(settings["dimensions"], n - g - 1, width)
    if g < 2 or dof < 1 or dimensions < 1:
        raise ValueError("Insufficient training groups/residual degrees of freedom")

    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    residual_for_pca = z - grouped_means(z, gi, ng)[gi]
    _, singular, vt = linalg.svd(residual_for_pca, full_matrices=False, check_finite=False)
    if singular[dimensions - 1] <= singular[0] * 1e-10:
        raise ValueError("Registered PCA dimensions exceed numerical within-prompt rank")
    basis = vt[:dimensions].T
    reduced = z @ basis
    class_counts = np.bincount(y, minlength=classes)
    prompt_effect = np.zeros((g, dimensions))
    for _ in range(settings["two_way_iterations"]):
        class_mean = grouped_means(reduced - prompt_effect[gi], y, class_counts)
        prompt_effect = grouped_means(reduced - class_mean[y], gi, ng)
    shift = np.average(prompt_effect, axis=0, weights=ng)
    prompt_effect -= shift
    class_mean += shift
    error = reduced - class_mean[y] - prompt_effect[gi]

    residual_raw = error.T @ error / dof
    residual_shrinkage = float(ledoit_wolf_shrinkage(error, assume_centered=True))
    residual_cov = shrink(residual_raw, residual_shrinkage)
    average_variance = float(np.trace(residual_cov) / dimensions)
    if not np.isfinite(average_variance) or average_variance <= 0:
        raise ValueError("Degenerate residual covariance")
    residual_cov += settings["relative_residual_jitter"] * average_variance * np.eye(dimensions)

    centered_prompt = prompt_effect - prompt_effect.mean(0)
    prompt_raw = centered_prompt.T @ centered_prompt / (g - 1) - np.mean(1.0 / ng) * residual_raw
    prompt_shrinkage = float(ledoit_wolf_shrinkage(centered_prompt, assume_centered=True))
    target_trace = max(float(np.trace(prompt_raw)), 1e-8 * float(np.trace(residual_cov)))
    prompt_cov = shrink(prompt_raw, prompt_shrinkage, target_trace)
    lower = linalg.cholesky(residual_cov, lower=True, check_finite=False)
    inv_lower = linalg.solve_triangular(lower, np.eye(dimensions), lower=True, check_finite=False)
    relative_prompt = inv_lower @ prompt_cov @ inv_lower.T
    eigenvalues, rotation = linalg.eigh((relative_prompt + relative_prompt.T) / 2, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, settings["prompt_eigenvalue_floor"])
    whitening = inv_lower.T @ rotation
    prototypes = class_mean @ whitening
    transform = basis @ whitening
    # Marginalize the unknown new prompt. The common quadratic term cancels
    # between classes, leaving a batch-independent linear score per row.
    weighted_prototypes = prototypes / (1 + eigenvalues)
    model = {"mean": mean, "scale": scale,
             "coefficients": transform @ weighted_prototypes.T,
             "intercept": -0.5 * (prototypes * weighted_prototypes).sum(1),
             "classes": np.arange(classes, dtype=np.int64)}
    if any(not np.isfinite(value).all() for value in model.values()):
        raise ValueError("Non-finite Gaussian head")
    info = {"training_rows": n, "training_groups": g, "dimensions": dimensions,
            "residual_degrees_of_freedom": dof,
            "residual_shrinkage": residual_shrinkage, "prompt_shrinkage": prompt_shrinkage,
            "retained_within_prompt_variance": float(np.square(singular[:dimensions]).sum() / np.square(singular).sum()),
            "prompt_relative_variance_quantiles": np.quantile(eigenvalues, [.05,.5,.95,1]).tolist()}
    return model, info


def predict_probabilities(features, model):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]) or not np.isfinite(x).all():
        raise ValueError("Invalid prediction features")
    scores = ((x - model["mean"]) / model["scale"]) @ model["coefficients"] + model["intercept"]
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores")
    return softmax(scores, axis=1)
