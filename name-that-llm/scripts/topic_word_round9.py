"""Fixed text-only grouping and partial centering of the word view."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def capped_average(similarity, cap=18, threshold=0.15):
    """Size-weighted average linkage, with deterministic input-order tie breaking."""
    a = np.asarray(similarity, dtype=np.float32).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("Invalid similarity matrix")
    n = len(a)
    np.fill_diagonal(a, -np.inf)
    sizes, alive, labels = np.ones(n, dtype=np.int32), np.ones(n, bool), np.arange(n)
    while n > 1:
        permitted = (sizes[:, None] + sizes[None, :] <= cap) & alive[:, None] & alive[None, :]
        m = np.where(permitted, a, -np.inf)
        i, j = divmod(int(np.argmax(m)), n)
        if not np.isfinite(m[i, j]) or m[i, j] < threshold:
            break
        # Matches the reference Lance-Williams update; dead/diagonal entries stay excluded.
        with np.errstate(invalid="ignore"):
            new = (sizes[i] * a[i] + sizes[j] * a[j]) / (sizes[i] + sizes[j])
        a[i] = new
        a[:, i] = new
        a[i, i] = -np.inf
        a[j] = -np.inf
        a[:, j] = -np.inf
        sizes[i] += sizes[j]
        alive[j] = False
        labels[labels == j] = i
    return labels


def infer_groups(texts, ids, settings):
    """Never accepts model labels, prompt ids, or known group counts."""
    texts, ids = np.asarray(texts), np.asarray(ids)
    if texts.ndim != 1 or ids.shape != texts.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Invalid text/id axes")
    order = np.argsort(ids, kind="stable")
    groups = np.empty(len(texts), dtype=np.int64)
    groups[order] = np.arange(len(texts))
    if len(texts) < 40:
        return groups
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                          min_df=settings["min_df"], max_df=settings["max_df"])
    try:
        matrix = vec.fit_transform(texts[order].tolist())
    except ValueError as error:
        if not any(s in str(error) for s in ("empty vocabulary", "After pruning, no terms remain")):
            raise
        return groups
    groups[order] = capped_average((matrix @ matrix.T).toarray(), settings["cap"], settings["threshold"])
    return groups


def center_word(matrix, groups, alpha=0.5, lo=2, hi=18):
    """Subtract alpha times group mean only for eligible groups; retain other rows."""
    x = csr_matrix(matrix, dtype=np.float32)
    groups = np.asarray(groups)
    if groups.shape != (x.shape[0],) or not 0 <= alpha <= 1 or not np.isfinite(x.data).all():
        raise ValueError("Invalid centering inputs")
    _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    eligible = (sizes[inverse] >= lo) & (sizes[inverse] <= hi)
    if not eligible.any() or alpha == 0:
        return x.copy()
    # Ineligible groups have zero membership, so large external groups cannot densify rows.
    rows = np.flatnonzero(eligible)
    membership = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, inverse[rows])),
                            shape=(len(groups), len(sizes)))
    means = membership.T @ x
    means = means.multiply((1 / sizes).astype(np.float32)[:, None]).tocsr()
    correction = membership @ means
    correction.data *= np.float32(alpha)
    result = (x - correction).tocsr()
    result.eliminate_zeros()
    return result


def transform_views(views, texts, groups, alpha):
    parts, word_counts = [], None
    for name, vec, weight in views:
        part = (vec.transform(texts) * weight).tocsr()
        if name == "word":
            before = part.nnz
            part = center_word(part, groups, alpha)
            word_counts = {"before_nnz": before, "after_nnz": part.nnz}
        parts.append(part)
    if word_counts is None:
        raise ValueError("Frozen views lack word features")
    return hstack(parts, format="csr"), word_counts


def group_diagnostics(groups, truth_groups=None):
    groups = np.asarray(groups)
    _, sizes = np.unique(groups, return_counts=True)
    result = {"rows": len(groups), "clusters": len(sizes), "singleton_rows": int(sizes[sizes == 1].sum()),
              "eligible_rows": int(sizes[(sizes >= 2) & (sizes <= 18)].sum()),
              "oversize_rows": int(sizes[sizes > 18].sum()),
              "size_histogram": {str(n): int((sizes == n).sum()) for n in sorted(set(sizes.tolist()))}}
    if truth_groups is not None:
        table = contingency_matrix(truth_groups, groups)
        choose2 = lambda values: int(np.sum(values * (values - 1) // 2))
        tp = choose2(table)
        inferred_pairs, true_pairs = choose2(table.sum(0)), choose2(table.sum(1))
        result.update(adjusted_rand_index=float(adjusted_rand_score(truth_groups, groups)),
                      pair_true_positives=tp, inferred_pairs=inferred_pairs, true_pairs=true_pairs,
                      pair_precision=tp / inferred_pairs if inferred_pairs else None,
                      pair_recall=tp / true_pairs if true_pairs else None)
    return result


def metric_gate(metrics, comparison, half_comparison, single_comparison, settings):
    values = [metrics["balanced_accuracy"], *metrics["fold_balanced_accuracy"], comparison["pooled_ba_gain"],
              *comparison["prompt_paired_gain_95ci"], half_comparison["pooled_ba_gain"], single_comparison["pooled_ba_gain"]]
    checks = {"finite_complete": bool(np.isfinite(values).all() and len(metrics["fold_balanced_accuracy"]) == 5),
              "pooled": metrics["balanced_accuracy"] >= settings["min_pooled_balanced_accuracy"],
              "each_fold": min(metrics["fold_balanced_accuracy"]) >= settings["min_each_fold_balanced_accuracy"],
              "paired_gain": comparison["pooled_ba_gain"] >= settings["min_gain"],
              "paired_interval": comparison["prompt_paired_gain_95ci"][0] > 0,
              "improved_folds": comparison["improved_folds"] >= settings["min_improved_folds"],
              "half_batch_robustness": half_comparison["pooled_ba_gain"] >= settings["min_half_batch_gain"],
              "singleton_robustness": single_comparison["pooled_ba_gain"] >= settings["min_singleton_gain"]}
    return {"metric_gate_passed": all(checks.values()), "checks": checks}
