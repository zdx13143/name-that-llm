"""Four text-only views, trained within each fold; scores are not calibrated."""
from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

FUNCTION_WORDS = frozenset("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_text(text):
    return " ".join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
                    for t in re.findall(r"\w+|[^\w\s]", text))


def make_model(method):
    views = []
    for view in method["views"]:
        name = view["name"]
        analyzer = "char" if name == "shape" else "word" if name == "function" else name
        options = dict(analyzer=analyzer, ngram_range=tuple(view["ngram_range"]),
                       lowercase=False, sublinear_tf=True, min_df=2,
                       max_features=view["max_features"], dtype=np.float64)
        if name in ("shape", "function"):
            options["preprocessor"] = shape_text if name == "shape" else function_text
        if analyzer == "word":
            options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        views.append((name, TfidfVectorizer(**options)))
    return Pipeline([
        ("views", FeatureUnion(views, transformer_weights={v["name"]: v["weight"] for v in method["views"]}, n_jobs=1)),
        ("classifier", LinearSVC(**method["classifier"]))])


def predict_scores(model, texts, classes=18):
    if not np.array_equal(model.classes_, np.arange(classes)) or classes < 3:
        raise ValueError("Expected contiguous multiclass label order")
    scores = np.asarray(model.decision_function(texts), dtype=np.float64)
    if scores.shape != (len(texts), classes) or not np.isfinite(scores).all():
        raise ValueError("Invalid SVC decision scores")
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    return scores / scores.sum(axis=1, keepdims=True)


def blend_scores(style, semantic, weight):
    if not 0 <= weight <= 1 or style.shape != semantic.shape:
        raise ValueError("Invalid fixed fusion")
    from windows_modernbert_dev import probabilities
    a = probabilities(style, *style.shape)
    b = probabilities(semantic, *semantic.shape)
    return probabilities(weight * a + (1 - weight) * b, *style.shape)
