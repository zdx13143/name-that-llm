"""Fixed group-level margin shrinkage; no labels or fitting."""
import numpy as np


def center_margins(scores, groups, alpha=0.5):
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups)
    if (scores.ndim != 2 or scores.shape[1] != 18 or groups.shape != (len(scores),)
            or not np.isfinite(scores).all() or alpha != 0.5):
        raise ValueError("Invalid fixed margin-centering inputs")
    result = scores.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if 2 <= len(rows) <= 18:
            result[rows] -= alpha * scores[rows].mean(axis=0)
    return result


def predict(artifact, texts, ids):
    """One whole unlabeled batch, identical grouping for features and margins."""
    from topic_word_round9 import infer_groups, transform_views
    if not 1 <= len(texts) <= 4096 or len(texts) != len(ids):
        raise ValueError("Expected 1..4096 test rows")
    groups = infer_groups(texts, ids, artifact["grouping"])
    matrix, _ = transform_views(artifact["views"], texts, groups, artifact["word_alpha"])
    model = artifact["classifier"]
    if not np.array_equal(model.classes_, np.arange(18)):
        raise ValueError("Class order changed")
    scores = np.asarray(model.decision_function(matrix), dtype=np.float64)
    if artifact["margin_correction"]:
        scores = center_margins(scores, groups)
    if scores.shape != (len(texts), 18) or not np.isfinite(scores).all():
        raise ValueError("Invalid decision scores")
    return scores, groups


def synthetic_inputs():
    # 40 distinct subjects, each repeated twice: min_df=2 / max_df=.05 retains them.
    nouns = "orchid telescope copper meadow violin glacier pottery lantern dolphin orchard volcano compass velvet fossil walnut bicycle bamboo river marble satellite cricket cinnamon cathedral penguin engine canvas dragonfly cedar museum ocean robot sandstone harpsichord eclipse feather emerald pumpkin comet willow prism".split()
    texts = [f"The {noun} appeared in a detailed observation of the {noun}. "
             + ("Careful measurements explained its structure." if j == 0 else "We recorded the evidence, then discussed its limits.")
             for noun in nouns for j in range(2)]
    ids = [f"artificial-{i:03d}" for i in range(len(texts))]
    return texts, ids
