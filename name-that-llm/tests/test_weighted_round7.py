"""Synthetic admission checks; these tests perform no real-data model fit."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import windows_external_round7 as run
from weighted_style_round7 import sample_weights


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(run.CONFIG.read_bytes())
        self.metrics = {"balanced_accuracy": .45, "fold_balanced_accuracy": [.45]*5}
        self.control = {"pooled_ba_gain": .02, "prompt_paired_gain_95ci": [.001, .04],
                        "fold_ba_gains": [.02]*5, "improved_folds": 5}
        self.prior = {"pooled_ba_gain": .08}

    def test_exact_gate_rejects_nearly_one_point_gain(self):
        self.control["pooled_ba_gain"] = np.nextafter(.01, 0)
        self.assertFalse(run.metric_gate(self.metrics, self.control, self.prior, self.config)["metric_gate_passed"])

    def test_no_fallback_when_interval_crosses_zero(self):
        self.control["prompt_paired_gain_95ci"] = [-.001, .04]
        self.assertFalse(run.metric_gate(self.metrics, self.control, self.prior, self.config)["metric_gate_passed"])

    def test_nonfinite_or_incomplete_results_do_not_pass(self):
        self.metrics["balanced_accuracy"] = float("nan")
        self.assertFalse(run.metric_gate(self.metrics, self.control, self.prior, self.config)["metric_gate_passed"])
        self.metrics["balanced_accuracy"] = .45
        self.metrics["fold_balanced_accuracy"] = [.45]*4
        self.assertFalse(run.metric_gate(self.metrics, self.control, self.prior, self.config)["metric_gate_passed"])

    def test_weight_array_matches_official_then_external_order(self):
        np.testing.assert_array_equal(sample_weights(3, 5, 4), [4, 4, 4, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            sample_weights(3, 5, float("nan"))


if __name__ == "__main__":
    unittest.main()
