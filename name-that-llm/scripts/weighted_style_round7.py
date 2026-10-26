"""Four-view SVC branch matching the public reference architecture."""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from stylegram_round5 import shape_text, function_text


def vectorizer(view):
    name = view["name"]
    analyzer = "char" if name == "shape" else "word" if name == "function" else name
    options = dict(analyzer=analyzer, ngram_range=tuple(view["ngram_range"]), lowercase=False,
                   sublinear_tf=True, min_df=2, max_features=view["max_features"], dtype=np.float32)
    if name in ("shape", "function"):
        options["preprocessor"] = shape_text if name == "shape" else function_text
    if analyzer == "word":
        options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    return TfidfVectorizer(**options)


def sample_weights(official_rows, external_rows, official_weight):
    if official_rows <= 0 or external_rows <= 0 or not np.isfinite(official_weight) or official_weight <= 0:
        raise ValueError("Invalid fixed sample weights")
    return np.concatenate([np.full(official_rows, official_weight), np.ones(external_rows)])


def transform(views, texts):
    return hstack([vec.transform(texts) * weight for _, vec, weight in views], format="csr")


def classifier(settings):
    return LinearSVC(**settings)


def decision(views, model, texts):
    if not np.array_equal(model.classes_, np.arange(18)):
        raise ValueError("SVC label order changed")
    scores = np.asarray(model.decision_function(transform(views, texts)), dtype=np.float64)
    if scores.shape != (len(texts), 18) or not np.isfinite(scores).all():
        raise ValueError("Invalid classifier scores")
    return scores
