"""Synthetic-only Windows development isolation, identity and budget regressions."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import windows_modernbert_dev as dev


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = self.root / "labels.json"
        self.names = [f"model-{i:02}" for i in range(18)]
        self.labels.write_text(json.dumps({"labels": self.names}), encoding="utf-8")
        groups = np.asarray([f"prompt-{i:03}" for i in range(150)])
        protected = set(np.random.default_rng(20261018).permutation(groups)[:30])
        missing = next(group for group in groups if group not in protected)
        rows = []
        for group in groups:
            for label in range(18):
                if group == missing and label == 17:
                    continue
                held = group in protected
                rows.append({"id": f"id-{len(rows)}", "prompt_id": group,
                             "text": "PROTECTED text" if held else f"development {group} {label}",
                             "model": "PROTECTED label" if held else self.names[label]})
        self.train = self.root / "train.csv"
        with self.train.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "prompt_id", "text", "model"])
            writer.writeheader()
            writer.writerows(rows)
        metadata = [{key: row[key] for key in ("id", "prompt_id")} for row in rows]
        self.input = {"train_sha256": dev.digest(self.train), "labels_sha256": dev.digest(self.labels),
                      "split_sha256": hashlib.sha256(dev.historical_manifest_bytes(metadata)).hexdigest(),
                      "development_rows": 2159, "protected_rows": 540,
                      "development_groups": 120, "protected_groups": 30}

    def data(self):
        return dev.load_development(self.train, self.input, self.labels)

    def test_protected_text_and_invalid_protected_label_never_enter_development(self):
        data = self.data()
        with self.train.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = json.loads(dev.historical_manifest_bytes(rows))["development_indices"]
        np.testing.assert_array_equal(data["row_index"], expected)
        np.testing.assert_array_equal(data["ids"], [rows[i]["id"] for i in expected])
        self.assertEqual(len(data["truth"]), 2159)
        self.assertEqual(len(set(data["groups"])), 120)
        self.assertFalse(any("PROTECTED" in text for text in data["texts"]))
        folds, ids = dev.grouped_folds(data)
        self.assertEqual(set(ids), set(range(5)))
        for train, valid in folds:
            self.assertFalse(set(data["groups"][train]) & set(data["groups"][valid]))

    def test_changed_train_and_split_identity_are_rejected(self):
        for key in ("train_sha256", "split_sha256", "labels_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                dev.load_development(self.train, {**self.input, key: "0" * 64}, self.labels)

    def test_same_shape_cache_with_wrong_row_order_or_contract_is_rejected(self):
        data = self.data()
        _, folds = dev.grouped_folds(data)
        features = np.ones((2159, 3), dtype=np.float32)
        path = self.root / "features.npz"
        dev.save_array(path, "features", features, data, folds, "fixed")
        np.testing.assert_array_equal(dev.load_features(path, data, folds, "fixed", 3), features)
        for changed, identity in (({**data, "ids": data["ids"][::-1]}, "fixed"), (data, "changed")):
            with self.assertRaisesRegex(ValueError, "identity"):
                dev.load_features(path, changed, folds, identity, 3)
        with self.assertRaises(FileExistsError):
            dev.save_array(path, "features", features, data, folds, "fixed")

    def test_nonfinite_wrong_dtype_and_wrong_dimensions_cache_are_rejected(self):
        data = self.data()
        _, folds = dev.grouped_folds(data)
        values = (np.full((2159, 3), np.nan, dtype=np.float32), np.ones((2159, 3), dtype=np.float64),
                  np.ones((2159, 2), dtype=np.float32))
        for number, value in enumerate(values):
            path = self.root / f"invalid-{number}.npz"
            dev.save_array(path, "features", value, data, folds, "fixed")
            with self.assertRaisesRegex(ValueError, "Invalid feature matrix"):
                dev.load_features(path, data, folds, "fixed", 3)

    def test_bad_model_returns_blocked_without_consuming_claim(self):
        config, base = dev.configuration()
        claim = self.root / "unused.json"
        with patch.object(dev, "CLAIM", claim), patch.object(dev, "OUTPUT", self.root / "no-output"):
            with patch.object(dev, "validate_files", side_effect=ValueError("synthetic bad model")):
                state, value = dev.readiness(self.train, config, base)
        self.assertEqual(state["state"], "blocked")
        self.assertTrue(any("synthetic bad model" in reason for reason in state["blockers"]))
        self.assertIsNone(value)
        self.assertFalse(claim.exists())

    def test_float32_roundoff_is_accepted_but_nan_and_uncovered_rows_fail(self):
        valid = np.full((2, 18), 1 / 18, dtype=np.float32)
        actual = dev.probabilities(valid, 2, 18)
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_allclose(actual.sum(1), 1, atol=1e-14)
        for bad in (np.full((2, 18), np.nan), np.zeros((2, 18)), np.ones((2, 18))):
            with self.assertRaises(ValueError):
                dev.probabilities(bad, 2, 18)

    def test_fixed_head_produces_complete_oof_and_descriptive_metrics(self):
        full = self.data()
        selected = np.isin(full["groups"], np.unique(full["groups"])[:10])
        data = {key: value if key == "labels" else value[selected] for key, value in full.items()}
        features = np.eye(18, dtype=np.float32)[data["truth"]]
        folds, ids = dev.grouped_folds(data)
        evaluation = dev.configuration()[0]["evaluation"]
        claim = {"attempted_fits": 0, "completed_fits": 0}
        claim_path = self.root / "claim.json"
        dev.write_json(claim_path, claim)
        oof, times = dev.fit_head(features, data, folds, evaluation, claim, 5, claim_path)
        self.assertEqual(claim, {"attempted_fits": 5, "completed_fits": 5})
        self.assertEqual(len(times), 5)
        metrics = dev.metrics(oof, data, ids, evaluation)
        self.assertEqual(metrics["balanced_accuracy"], 1)
        self.assertEqual(metrics["prompt_bootstrap_ba_95ci"], [1, 1])
        self.assertEqual(len(metrics["confusion_matrix"]), 18)

    def test_fit_exception_consumes_attempt_before_operation(self):
        data = self.data()
        folds, _ = dev.grouped_folds(data)
        claim = {"attempted_fits": 0, "completed_fits": 0}
        claim_path = self.root / "failed.json"
        dev.write_json(claim_path, claim)
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                dev.fit_head(np.ones((2159, 2)), data, folds, dev.configuration()[0]["evaluation"], claim, 5, claim_path)
        self.assertEqual(json.loads(claim_path.read_bytes()), {"attempted_fits": 1, "completed_fits": 0})
        with self.assertRaisesRegex(RuntimeError, "Budget exhausted"):
            dev.consume(claim, "attempted_fits", 1, claim_path)

    def test_forward_budget_uses_current_claim_after_path_switch(self):
        current = self.root / "active-recovery.json"
        claim = {"attempted_forward_calls": 0}
        with patch.object(dev, "CLAIM", current), patch.object(dev, "write_json") as writer:
            dev.consume(claim, "attempted_forward_calls", 5)
        self.assertEqual(writer.call_args.args[0], current)
        self.assertEqual(claim["attempted_forward_calls"], 1)

    def test_fit_budget_uses_current_claim_after_path_switch(self):
        current = self.root / "active-recovery.json"
        data = self.data()
        folds, _ = dev.grouped_folds(data)
        claim = {"attempted_fits": 0, "completed_fits": 0}
        with patch.object(dev, "CLAIM", current), patch.object(dev, "write_json") as writer:
            with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=RuntimeError("synthetic stop")):
                with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                    dev.fit_head(np.ones((2159, 2)), data, folds, dev.configuration()[0]["evaluation"], claim, 5)
        self.assertEqual(writer.call_args.args[0], current)
        self.assertEqual(claim["attempted_fits"], 1)

    def test_supervisor_kills_overdue_child_and_preserves_fixed_claim(self):
        claim, output = self.root / "claims/run.json", self.root / "run"
        config = {"config": {"experiment_id": "synthetic-only"}}
        base = {"budget": {"development_timeout_seconds": 0.05, "max_host_rss_gib": 8}}
        real_popen = subprocess.Popen
        def sleeper(*args, **kwargs):
            return real_popen([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs)
        with patch.object(dev, "CLAIM", claim), patch.object(dev, "OUTPUT", output), patch.object(dev, "SOURCES", ()):
            with patch.object(dev.subprocess, "Popen", side_effect=sleeper):
                with self.assertRaises(TimeoutError):
                    dev.supervise(self.train, config, base)
            state = json.loads(claim.read_bytes())
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["attempted_fits"], 0)
            before = claim.read_bytes()
            with self.assertRaises(FileExistsError):
                dev.supervise(self.train, config, base)
            self.assertEqual(before, claim.read_bytes())

    def test_termination_exception_still_records_failure(self):
        claim, output = self.root / "claims/cleanup.json", self.root / "cleanup"
        value = {"config": {"experiment_id": "synthetic-only"}}
        base = {"budget": {"development_timeout_seconds": 2, "max_host_rss_gib": 8}}
        real_popen = subprocess.Popen
        def failed_worker(*args, **kwargs):
            return real_popen([sys.executable, "-c", "raise SystemExit(3)"], **kwargs)
        with patch.object(dev, "CLAIM", claim), patch.object(dev, "OUTPUT", output), patch.object(dev, "SOURCES", ()):
            with patch.object(dev.subprocess, "Popen", side_effect=failed_worker):
                with patch.object(dev, "terminate_worker", side_effect=subprocess.TimeoutExpired("synthetic", 30)):
                    with self.assertRaisesRegex(RuntimeError, "Worker failed"):
                        dev.supervise(self.train, value, base)
        state = json.loads(claim.read_bytes())
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["termination"]["cleanup_exception"], "TimeoutExpired")
        self.assertEqual(state["attempted_fits"], 0)

    def test_pre_spawn_failure_has_a_durable_reason(self):
        claim, output = self.root / "claims/pre-spawn.json", self.root / "existing"
        output.mkdir()
        with patch.object(dev, "CLAIM", claim), patch.object(dev, "OUTPUT", output):
            with self.assertRaises(FileExistsError):
                dev.supervise(self.train, {"config": {"experiment_id": "synthetic-only"}}, {})
        state = json.loads(claim.read_bytes())
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["phase"], "pre_spawn")
        self.assertEqual(state["exception_type"], "FileExistsError")


if __name__ == "__main__":
    unittest.main()
