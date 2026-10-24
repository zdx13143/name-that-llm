"""Synthetic leakage, export and admission checks; no competition input or claim."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stylegram_round5 as style
import windows_stylegram_round5 as runner


class StylegramTests(unittest.TestCase):
    def setUp(self):
        self.config, self.base = runner.configuration()
        self.texts = ["The CAT and the dog!", "The CAT and the dog!",
                      "We can write a small story.", "We can write a small story.",
                      "Why? You ask why?", "Why? You ask why?"]
        self.y = np.repeat(np.arange(3), 2)

    def test_text_only_views(self):
        self.assertEqual(style.shape_text("HELLO World 123!"), "X x 123!")
        self.assertEqual(style.function_text("The dragon and a kettle!"), "the CONTENT and a CONTENT !")

    def test_training_vocabulary_and_inference_invariance(self):
        model = style.make_model(self.config["method"]).fit(self.texts, self.y)
        before = {name: dict(v.vocabulary_) for name, v in model[0].transformer_list}
        valid = ["validationonlyquasar nevertrainword!", "We can write."]
        p = style.predict_scores(model, valid, 3)
        np.testing.assert_allclose(p[::-1], style.predict_scores(model, valid[::-1], 3), atol=1e-12)
        np.testing.assert_allclose(p[:1], style.predict_scores(model, valid[:1], 3), atol=1e-12)
        self.assertEqual(before, {name: dict(v.vocabulary_) for name, v in model[0].transformer_list})
        self.assertNotIn("validationonlyquasar", before["word"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(p, style.predict_scores(joblib.load(path), valid, 3))

    def test_label_order_rejected(self):
        model = style.make_model(self.config["method"]).fit(self.texts, self.y + 1)
        with self.assertRaises(ValueError):
            style.predict_scores(model, self.texts, 3)

    def test_fixed_fusion_and_invalid_inputs(self):
        a, b = np.array([[.7, .2, .1]]), np.array([[.1, .3, .6]])
        np.testing.assert_allclose(style.blend_scores(a, b, .8), [[.58, .22, .20]])
        with self.assertRaises(ValueError):
            style.blend_scores(a, b * 2, .8)

    def test_guard_requires_both_absolute_and_comparative_success(self):
        metrics = {"balanced_accuracy":.4, "fold_balanced_accuracy":[.4]*5}
        comparison = {"pooled_ba_gain":.1, "prompt_paired_gain_95ci":[.05,.15], "fold_ba_gains":[.1]*5, "improved_folds":5}
        control = {"pooled_ba_gain":.01}
        self.assertTrue(runner.metric_gate(metrics, comparison, control, self.config)["metric_gate_passed"])
        for key, value in (("balanced_accuracy", .35), ("fold_balanced_accuracy", [.4]*4+[.29])):
            changed = copy.deepcopy(metrics)
            changed[key] = value
            self.assertFalse(runner.metric_gate(changed, comparison, control, self.config)["metric_gate_passed"])
        self.assertFalse(runner.metric_gate(metrics, comparison, {"pooled_ba_gain":-.001}, self.config)["metric_gate_passed"])
        comparison["prompt_paired_gain_95ci"][0] = 0
        self.assertFalse(runner.metric_gate(metrics, comparison, control, self.config)["metric_gate_passed"])

    def test_existing_claim_blocks_before_loading_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claimed.json"
            path.write_text("{}")
            with patch.object(runner, "CLAIM", path), patch.object(runner, "load_inputs", side_effect=AssertionError("must not load")):
                result, value = runner.readiness(self.config, self.base)
            self.assertEqual(result["state"], "blocked")
            self.assertIsNone(value)

    def test_sixth_fit_refused_without_mutating_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claim.json"
            claim = {"attempted_fits": 4}
            runner.dev.consume(claim, "attempted_fits", 5, path)
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                runner.dev.consume(claim, "attempted_fits", 5, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(before)["attempted_fits"], 5)

    def test_worker_guard_parent_time_and_memory(self):
        parent = MagicMock()
        parent.create_time.return_value = 100
        parent.is_running.return_value = True
        parent.memory_info.return_value.rss = 1024
        parent.children.return_value = []
        claim = {"supervisor_pid":123, "supervisor_create_time":100, "started_unix_seconds":100}
        budget = {"development_timeout_seconds":10, "max_host_rss_gib":1}
        with patch.object(runner.psutil, "Process", return_value=parent), patch.object(runner.time, "time", return_value=105):
            self.assertIsNone(runner.guard_violation(claim, budget))
            parent.create_time.return_value = 101
            self.assertIn("identity", runner.guard_violation(claim, budget))
            parent.create_time.return_value = 100
            parent.memory_info.return_value.rss = 2**31
            self.assertIn("RSS", runner.guard_violation(claim, budget))
        with patch.object(runner.psutil, "Process", return_value=parent), patch.object(runner.time, "time", return_value=111):
            self.assertIn("wall-clock", runner.guard_violation(claim, budget))


if __name__ == "__main__":
    unittest.main()
