"""Synthetic final-stage admission tests: no competition data or model execution."""
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import windows_modernbert_final_round4 as final


class ConditionalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.base = final.cv.configuration()
        self.claim = {"state": "completed", "contract_sha256": "hash", "supervisor_sha256": "hash", "summary_sha256": "hash"}
        self.supervisor = {"state": "completed", "exit_code": 0, "candidate_development_gate_passed": True}
        self.summary = {"metrics": {}, "paired_round3_comparison": {}}
        self.verified = {"state": "verified", "development_gate_passed": True, "contract_sha256": "hash", "oof_sha256": "hash"}
        self.documents = {final.cv.CLAIM: self.claim, final.cv.OUTPUT / "supervisor.json": self.supervisor,
                          final.cv.OUTPUT / "summary.json": self.summary, final.RECONCILIATION: self.verified,
                          final.cv.OUTPUT / "contract.json": {"sources": {}}}

    def admission(self, data=None):
        def read(path):
            return json.dumps(self.documents[path]).encode()
        with mock.patch.object(final.cv, "configuration", return_value=(self.config, self.base)), \
             mock.patch.object(Path, "read_bytes", read), mock.patch.object(final, "digest", return_value="hash"), \
             mock.patch.object(final.cv, "metric_gate", return_value={"metric_gate_passed": True}), \
             mock.patch.object(final.cv.dev, "object_hash", return_value="hash"), \
             mock.patch.object(final.importlib.metadata, "version", side_effect=lambda p: self.config["runtime"][p]), \
             mock.patch.object(final.cv, "load_inputs", return_value=(data, None, None, None)) as loader:
            with self.assertRaises(ValueError):
                final.inputs_and_contract()
            return loader.call_count

    def test_unfinished_cv_never_loads_training_data(self):
        self.claim["state"] = "started"
        self.assertEqual(self.admission(), 0)

    def test_failed_gate_never_loads_training_data(self):
        self.supervisor["candidate_development_gate_passed"] = False
        self.assertEqual(self.admission(), 0)

    def test_unverified_results_never_load_training_data(self):
        self.verified["state"] = "pending"
        self.assertEqual(self.admission(), 0)

    def test_wrong_final_training_partition_rejected(self):
        self.assertEqual(self.admission({"truth": np.zeros(2158, dtype=int), "groups": np.arange(2158) % 120}), 1)

    def test_existing_final_claim_blocks_before_admission(self):
        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(final, "inputs_and_contract") as admission, \
             mock.patch.object(sys, "argv", ["final", "--run"]), mock.patch("builtins.print"):
            self.assertEqual(final.main(), 2)
            admission.assert_not_called()


if __name__ == "__main__":
    unittest.main()
