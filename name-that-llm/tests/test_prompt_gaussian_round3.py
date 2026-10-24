"""Synthetic Gaussian invariance, paired statistics and independent budget tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prompt_gaussian_head as head
import windows_prompt_gaussian_round3 as run


class GaussianTests(unittest.TestCase):
    def setUp(self):
        self.config, self.base = run.configuration()
        self.settings = {**self.config["method"], "dimensions": 6}
        rng = np.random.default_rng(51)
        self.groups = np.repeat(np.arange(20), 3)
        self.truth = np.tile(np.arange(3), 20)
        self.features = (rng.normal(size=(20, 8))[self.groups] * 2
                         + np.eye(3, 8)[self.truth] * 4 + rng.normal(size=(60, 8)) * .3)

    def test_single_row_batch_order_and_export_preserve_predictions(self):
        model, info = head.fit_head(self.features, self.truth, self.groups, self.settings, classes=3)
        before = {key: value.copy() for key, value in model.items()}
        prediction = head.predict_probabilities(self.features, model)
        np.testing.assert_allclose(prediction.sum(1), 1, atol=1e-14)
        np.testing.assert_allclose(head.predict_probabilities(self.features[::-1], model)[::-1], prediction, atol=1e-12)
        np.testing.assert_allclose(head.predict_probabilities(self.features[17:18], model), prediction[17:18], atol=1e-12)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(info["training_groups"], 20)
        for key in model:
            np.testing.assert_array_equal(model[key], before[key])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "head.npz"
            np.savez_compressed(path, **model)
            with np.load(path, allow_pickle=False) as artifact:
                loaded = {key: artifact[key] for key in artifact.files}
            np.testing.assert_allclose(head.predict_probabilities(self.features, loaded), prediction, atol=0, rtol=0)

    def test_training_affine_scaling_changes_no_predictions(self):
        first, _ = head.fit_head(self.features, self.truth, self.groups, self.settings, classes=3)
        scaled = self.features * np.arange(1, 9) + np.arange(8) * 13
        second, _ = head.fit_head(scaled, self.truth, self.groups, self.settings, classes=3)
        np.testing.assert_allclose(head.predict_probabilities(self.features, first),
                                   head.predict_probabilities(scaled, second), atol=1e-10, rtol=1e-10)

    def test_nonfinite_missing_classes_and_single_prompt_are_rejected(self):
        changed = self.features.copy()
        changed[0, 0] = np.nan
        for features, truth, groups in ((changed, self.truth, self.groups),
                                        (self.features, np.zeros(60, dtype=int), self.groups),
                                        (self.features, self.truth, np.zeros(60, dtype=int))):
            with self.assertRaises(ValueError):
                head.fit_head(features, truth, groups, self.settings, classes=3)

    def test_paired_group_interval_has_known_constant_gain(self):
        truth, groups = np.tile(np.arange(3), 10), np.repeat(np.arange(10), 3)
        data = {"truth": truth, "groups": groups, "labels": np.asarray(["a", "b", "c"])}
        baseline = np.tile([1., 0., 0.], (30, 1))
        candidate = np.eye(3)[truth]
        result = run.paired_comparison(candidate, baseline, data, groups // 2, self.config["evaluation"])
        self.assertAlmostEqual(result["pooled_ba_gain"], 2 / 3)
        np.testing.assert_allclose(result["prompt_paired_gain_95ci"], [2 / 3, 2 / 3])
        self.assertEqual(result["improved_folds"], 5)
        metrics = {"balanced_accuracy": .4, "fold_balanced_accuracy": [.4] * 5}
        self.assertTrue(run.metric_gate(metrics, result, self.config)["metric_gate_passed"])
        bad = {**result, "prompt_paired_gain_95ci": [0, .9]}
        self.assertFalse(run.metric_gate(metrics, bad, self.config)["metric_gate_passed"])
        bad = {**result, "improved_folds": 3}
        self.assertFalse(run.metric_gate(metrics, bad, self.config)["metric_gate_passed"])

    def test_fit_failure_is_charged_to_only_the_new_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            claim_path = Path(folder) / "new-claim.json"
            value = {"synthetic": True}
            claim = {"state": "started", "attempted_fits": 0, "completed_fits": 0,
                     "contract_sha256": run.dev.object_hash(value)}
            run.write_json(claim_path, claim)
            data = {"truth": self.truth, "groups": self.groups, "labels": np.arange(18)}
            folds = [(np.arange(48), np.arange(48,60))]
            inputs = (data, self.features, np.zeros((60,18)), folds, np.zeros(60), {})
            with patch.object(run, "CLAIM", claim_path), patch.object(run.dev, "require_live_supervisor"):
                with patch.object(run, "load_inputs", return_value=inputs), patch.object(run, "contract", return_value=value):
                    with patch.object(run, "fit_head", side_effect=RuntimeError("synthetic failure")):
                        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                            run.execute(self.config, self.base)
            result = json.loads(claim_path.read_bytes())
            self.assertEqual(result["attempted_fits"], 1)
            self.assertEqual(result["completed_fits"], 0)

    def test_existing_output_failure_is_durable_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            claim_path, output = root / "claims/new.json", root / "existing"
            output.mkdir()
            with patch.object(run, "CLAIM", claim_path), patch.object(run, "OUTPUT", output):
                with self.assertRaises(FileExistsError):
                    run.supervise(self.config, {"synthetic": True})
                first = claim_path.read_bytes()
                self.assertEqual(json.loads(first)["state"], "failed")
                self.assertEqual(json.loads(first)["attempted_fits"], 0)
                with self.assertRaises(FileExistsError):
                    run.supervise(self.config, {"synthetic": True})
                self.assertEqual(first, claim_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
