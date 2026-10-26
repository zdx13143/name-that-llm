import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import windows_requested_final_round7 as requested


class RequestedSubmissionTests(unittest.TestCase):
    def test_failed_gate_is_preserved_in_authorization(self):
        config = json.loads(requested.REQUEST.read_bytes())
        digests = {
            requested.REQUEST: requested.REQUEST_SHA256,
            requested.DEVELOPMENT_CLAIM: config["development_claim_sha256"],
            requested.DEVELOPMENT / "summary.json": config["development_summary_sha256"],
            requested.DEVELOPMENT / "verification.json": config["development_verification_sha256"],
        }
        with patch.object(requested.final.run, "digest", side_effect=digests.__getitem__):
            auth = requested.authorization({"state": "verified", "development_gate_passed": False})
        self.assertIs(auth["development_gate_passed"], False)
        self.assertEqual(auth["submission_budget"]["kaggle_formal_submissions"], 1)
        self.assertEqual(auth["final_budget"]["max_total_fit_attempts"], 5)

    def test_cannot_relabel_gate_as_passed(self):
        with self.assertRaisesRegex(ValueError, "verified failed-gate"):
            requested.authorization({"state": "verified", "development_gate_passed": True})

    def test_config_mutation_rejected(self):
        with patch.object(requested.final.run, "digest", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "config changed"):
                requested.authorization({"state": "verified", "development_gate_passed": False})


if __name__ == "__main__":
    unittest.main()
