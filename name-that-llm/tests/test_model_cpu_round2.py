"""Synthetic checks for the additional CPU feature and score-fusion paths."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_cpu_round2 import ScoreFusion, StylometryFeatures, make_features, normalize_scores


class AdditionalFeatureTests(unittest.TestCase):
    def test_style_features_are_finite_and_fold_scaler_does_not_refit_on_validation(self):
        raw = StylometryFeatures().transform(["", " ", "Short TEXT!", "Two lines.\nAnother line."])
        self.assertEqual(raw.shape, (4, 26))
        self.assertTrue(np.isfinite(raw).all())
        features = make_features({"representation": "char", "char_ngram": [1, 2],
                                  "min_df": 1, "style_weight": .25})
        features.fit_transform(["Short training text!", "Another training example."])
        scaler = dict(features.transformer_list)["style"].named_steps["scale"]
        frozen = scaler.mean_.copy()
        features.transform(["UNSEEN VALIDATION " * 100])
        np.testing.assert_array_equal(scaler.mean_, frozen)

    def test_boundary_analyzer_is_used_in_the_new_feature_path(self):
        features = make_features({"representation": "char", "char_analyzer": "char_wb"})
        self.assertEqual(dict(features.transformer_list)["char"].analyzer, "char_wb")

    def test_fusion_uses_normalized_independent_margins_and_preserves_class_mapping(self):
        class FixedScores:
            def __init__(self, scores):
                self.scores = np.array(scores, dtype=float)
            def decision_function(self, texts):
                return self.scores
        fusion = ScoreFusion({}, {}, char_weight=.75)
        fusion.char_model_ = FixedScores([[100., 0., 0.], [0., 4., 3.]])
        fusion.word_model_ = FixedScores([[0., 1., 0.], [0., 0., 100.]])
        fusion.classes_ = np.array([10, 20, 30])
        np.testing.assert_allclose(fusion.decision_function(["a", "b"]),
                                   [[.75, .25, 0.], [0., .6, .7]])
        np.testing.assert_array_equal(fusion.predict(["a", "b"]), [10, 30])
        self.assertTrue(np.isfinite(normalize_scores(np.zeros((2, 3)))).all())


if __name__ == "__main__":
    unittest.main()
