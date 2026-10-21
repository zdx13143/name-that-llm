"""Reusable, offline TF-IDF classification and strict submission checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def feature_config(config):
    return {key: value for key, value in config.items()
            if key not in {"classifier", "C", "class_weight"}}


def make_features(config):
    common = dict(lowercase=config.get("lowercase", False),
                  sublinear_tf=config.get("sublinear_tf", True),
                  min_df=config.get("min_df", 2), dtype=np.float32, norm="l2")
    representations = []
    if config["representation"] in {"char", "combined"}:
        representations.append(("char", TfidfVectorizer(
            analyzer="char", ngram_range=tuple(config.get("char_ngram", [3, 5])),
            max_features=100000, **common)))
    if config["representation"] in {"word", "combined"}:
        representations.append(("word", TfidfVectorizer(
            analyzer="word", ngram_range=tuple(config.get("word_ngram", [1, 2])),
            token_pattern=r"(?u)\b\w+\b", max_features=50000, **common)))
    if not representations:
        raise ValueError("Unknown representation")
    return FeatureUnion(representations, n_jobs=1)


def make_classifier(config, seed=20261018):
    common = dict(C=config.get("C", 1.0), class_weight=config.get("class_weight"),
                  random_state=seed, tol=1e-4)
    if config.get("classifier", "svm") == "svm":
        return LinearSVC(dual=True, max_iter=10000, **common)
    if config["classifier"] == "logistic":
        return LogisticRegression(solver="lbfgs", max_iter=1000, **common)
    raise ValueError("Unknown classifier")


def make_pipeline(config, seed=20261018):
    return Pipeline([("features", make_features(config)),
                     ("classifier", make_classifier(config, seed))])


def checked_submission(test_rows, predictions, labels, output):
    """Preserve every test ID and actual row count; never infer hidden size."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = [row["id"] for row in test_rows]
    predictions = [str(value) for value in predictions]
    if not ids or any(not str(value).strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Test IDs must be nonblank and unique")
    if len(predictions) != len(ids) or any(value not in labels for value in predictions):
        raise ValueError("Predictions must cover every test row with a reference label")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id": ids, "model": predictions}), output)
    result = pq.read_table(output).to_pydict()
    if result["id"] != ids or result["model"] != predictions:
        raise RuntimeError("Parquet round-trip validation failed")
    return {"rows": len(ids), "unique_ids": len(set(ids)), "labels_valid": True,
            "sha256": file_hash(output)}
