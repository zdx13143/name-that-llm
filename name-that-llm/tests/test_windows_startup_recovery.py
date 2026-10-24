"""Startup recovery is bound to a preserved zero-work failure, not any failed run."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import windows_modernbert_dev as dev


class StartupRecoveryTests(unittest.TestCase):
    def test_registered_recovery_config_matches_its_fixed_digest(self):
        self.assertEqual(dev.digest(dev.RECOVERY_CONFIG), dev.RECOVERY_CONFIG_SHA256)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "failed"
        self.output.mkdir()
        (self.output / "source_snapshot").mkdir()
        self.claim_path = self.root / "claim.json"
        self.previous = {"sources": {}, "fixed_protocol": "synthetic"}
        self.claim = {"state": "failed", "phase": "reconcile", "experiment_id": "synthetic",
                      "attempted_forward_calls": 0, "completed_forward_calls": 0,
                      "attempted_fits": 0, "completed_fits": 0,
                      "final_holdout_evaluations": 0, "full_data_final_fits": 0,
                      "termination": {"worker_still_running": False, "returncode": 1},
                      "contract_sha256": dev.object_hash(self.previous)}
        self.claim_path.write_text(json.dumps(self.claim), encoding="utf-8")
        (self.output / "contract.json").write_text(json.dumps(self.previous), encoding="utf-8")
        (self.output / "stderr.log").write_text("synthetic startup failure", encoding="utf-8")
        (self.output / "stdout.log").write_text("", encoding="utf-8")
        paths = {"claim": self.claim_path, "contract": self.output / "contract.json",
                 "stderr": self.output / "stderr.log", "stdout": self.output / "stdout.log"}
        self.amendment = {"experiment_id": "synthetic",
                          "original_artifact_sha256": {key: dev.digest(path) for key, path in paths.items()}}

    def test_original_failure_is_verified_without_mutation(self):
        before = self.claim_path.read_bytes()
        self.assertEqual(dev.verify_zero_work_startup(self.amendment, self.claim_path, self.output), self.previous)
        self.assertEqual(before, self.claim_path.read_bytes())
        with (self.output / "features.npz").open("xb") as handle:
            handle.write(b"unexpected")
        with self.assertRaisesRegex(ValueError, "Unexpected artifacts"):
            dev.verify_zero_work_startup(self.amendment, self.claim_path, self.output)

    def test_changed_amendment_is_rejected_before_touching_original_claim(self):
        path = self.root / "changed-recovery.json"
        path.write_text(json.dumps(self.amendment), encoding="utf-8")
        with patch.object(dev, "RECOVERY_CONFIG", path):
            with self.assertRaisesRegex(ValueError, "amendment identity"):
                dev.activate_startup_recovery()

    def test_identity_change_or_any_computation_rejects_recovery(self):
        self.claim["attempted_forward_calls"] = 1
        self.claim_path.write_text(json.dumps(self.claim), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "preserved failed attempt"):
            dev.verify_zero_work_startup(self.amendment, self.claim_path, self.output)
        self.amendment["original_artifact_sha256"]["claim"] = dev.digest(self.claim_path)
        with self.assertRaisesRegex(ValueError, "zero-work"):
            dev.verify_zero_work_startup(self.amendment, self.claim_path, self.output)
        self.claim["attempted_forward_calls"] = 0
        self.claim["attempted_fits"] = 1
        self.claim_path.write_text(json.dumps(self.claim), encoding="utf-8")
        self.amendment["original_artifact_sha256"]["claim"] = dev.digest(self.claim_path)
        with self.assertRaisesRegex(ValueError, "zero-work"):
            dev.verify_zero_work_startup(self.amendment, self.claim_path, self.output)

    def test_recovery_contract_cannot_change_data_or_parameters(self):
        data = {key: np.asarray([0, 1]) for key in ("row_index", "ids", "groups", "truth", "labels")}
        data["texts"] = np.asarray(["synthetic a", "synthetic b"])
        config = {"runtime": {}, "evaluation": {"folds": 5}, "fixed_parameter": 0.1}
        with patch.object(dev, "grouped_folds", return_value=([], np.asarray([0, 1]))):
            with patch.object(dev, "RECOVERY", None):
                original = dev.contract(data, config, {}, [])
            recovery = {"original_contract": original, "amendment": self.amendment, "amendment_sha256": "synthetic"}
            with patch.object(dev, "RECOVERY", recovery):
                self.assertIn("startup_recovery", dev.contract(data, config, {}, []))
                with self.assertRaisesRegex(ValueError, "registered experiment"):
                    dev.contract(data, {**config, "fixed_parameter": 0.2}, {}, [])
                with self.assertRaisesRegex(ValueError, "registered experiment"):
                    dev.contract({**data, "texts": data["texts"][::-1]}, config, {}, [])


if __name__ == "__main__":
    unittest.main()
