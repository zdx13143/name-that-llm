"""Second CPU round: interpretable style features and independent score fusion."""
from __future__ import annotations

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

import model_cpu as original


class StylometryFeatures(TransformerMixin, BaseEstimator):
    """Text-only counts and rates. No IDs, prompts, labels or external resources."""
    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        rows = []
        for text in texts:
            words = re.findall(r"\w+", text, flags=re.UNICODE)
            lengths = np.array([len(word) for word in words], dtype=float)
            n, count = max(len(text), 1), max(len(words), 1)
            sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)
            values = [np.log1p(len(text)), np.log1p(len(words)), np.log1p(text.count("\n")),
                      float(lengths.mean()) if len(lengths) else 0.,
                      float(lengths.std()) if len(lengths) else 0.,
                      len(set(word.lower() for word in words)) / count,
                      count / sentence_count, sum(char.isupper() for char in text) / n,
                      sum(char.isdigit() for char in text) / n,
                      sum(char.isspace() for char in text) / n,
                      sum(ord(char) > 127 for char in text) / n]
            values.extend(text.count(char) / n for char in
                          ("\n", ",", ";", ":", "!", "?", "(", "[", "-", "—", "'", '"', "`", "*", "/"))
            rows.append(values)
        return np.asarray(rows, dtype=np.float32)


def make_features(config):
    features = original.make_features(config)
    if config.get("char_analyzer") == "char_wb":
        features.set_params(char__analyzer="char_wb")
    if config.get("style_weight", 0):
        style = Pipeline([("counts", StylometryFeatures()), ("scale", StandardScaler()),
                          ("normalize", Normalizer())])
        features = FeatureUnion([("text", features), ("style", style)], n_jobs=1,
                                transformer_weights={"text": 1., "style": config["style_weight"]})
    return features


def normalize_scores(scores):
    return scores / np.maximum(np.linalg.norm(scores, axis=1, keepdims=True), 1e-12)


class ScoreFusion(ClassifierMixin, BaseEstimator):
    def __init__(self, char_config, word_config, char_weight=.5):
        self.char_config = char_config
        self.word_config = word_config
        self.char_weight = char_weight

    def fit(self, texts, y):
        self.char_model_ = make_pipeline(self.char_config)
        self.word_model_ = make_pipeline(self.word_config)
        self.char_model_.fit(texts, y)
        self.word_model_.fit(texts, y)
        self.classes_ = self.char_model_.classes_
        if not np.array_equal(self.classes_, self.word_model_.classes_):
            raise ValueError("Fusion component classes differ")
        return self

    def decision_function(self, texts):
        return (self.char_weight * normalize_scores(self.char_model_.decision_function(texts))
                + (1 - self.char_weight) * normalize_scores(self.word_model_.decision_function(texts)))

    def predict(self, texts):
        return self.classes_[self.decision_function(texts).argmax(axis=1)]


def make_pipeline(config, seed=20261018):
    if config.get("classifier") == "score_fusion":
        return ScoreFusion(config["char_config"], config["word_config"], config["char_weight"])
    return Pipeline([("features", make_features(config)),
                     ("classifier", original.make_classifier(config, seed))])
