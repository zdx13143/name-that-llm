"""Synthetic-only round selection, prior-record preservation and gate tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import windows_modernbert_dev as dev


class Round2Tests(unittest.TestCase):
    def test_new_protocol_switches_contract_source_and_worker_without_old_claim_write(self):
        old_claim, old_output, old_config = dev.CLAIM, dev.OUTPUT, dev.CONFIG
        with patch.multiple(dev, CONFIG=old_config, CLAIM=old_claim, OUTPUT=old_output,
                            SOURCES=dev.SOURCES, ROUND2=False, RECOVERY=None):
            with patch.object(dev, "verify_preserved_claims") as verify:
                dev.activate_round2()
            self.assertEqual(verify.call_count, 1)
            self.assertNotEqual(dev.CLAIM, old_claim)
            self.assertNotEqual(dev.OUTPUT, old_output)
            self.assertIn(dev.CONFIG, dev.SOURCES)
            self.assertNotIn(old_config, dev.SOURCES)
            config, base = dev.configuration()
            self.assertEqual(config["parent_experiment_id"], base["experiment_id"])
            self.assertEqual(config["budget"]["max_classifier_fit_attempts"], 5)
            self.assertIn("--round2", dev.worker_command(Path("synthetic.csv")))
            self.assertNotIn("--recover-zero-work-startup", dev.worker_command(Path("synthetic.csv")))

    def test_changed_old_claim_fails_without_repair_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            claim = root / "outputs/continuation_claims/old.json"
            claim.parent.mkdir(parents=True)
            claim.write_bytes(b'{"state":"failed"}')
            config = {"preserved_claims": {str(claim.relative_to(root)): dev.digest(claim)}}
            with patch.object(dev, "MODULE", root):
                dev.verify_preserved_claims(config)
                claim.write_bytes(b'{"state":"changed"}')
                before = claim.read_bytes()
                with self.assertRaisesRegex(ValueError, "prior-round"):
                    dev.verify_preserved_claims(config)
                self.assertEqual(claim.read_bytes(), before)

    def test_metric_gate_requires_all_five_folds_and_both_thresholds(self):
        config = json.loads(dev.ROUND2_CONFIG.read_bytes())
        good = {"balanced_accuracy": 0.36, "fold_balanced_accuracy": [0.30] * 5}
        self.assertTrue(dev.candidate_gate(good, config)["metric_gate_passed"])
        for bad in ({**good, "balanced_accuracy": .35999},
                    {**good, "fold_balanced_accuracy": [.29999] + [.4] * 4},
                    {**good, "fold_balanced_accuracy": [.4] * 4},
                    {**good, "balanced_accuracy": float("nan")},
                    {**good, "fold_balanced_accuracy": [1.01] * 5}):
            with self.subTest(bad=bad):
                self.assertFalse(dev.candidate_gate(bad, config)["metric_gate_passed"])


if __name__ == "__main__":
    unittest.main()
