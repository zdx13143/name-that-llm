"""Final-stage rejection checks; no raw data and no model fit."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import windows_external_final_round7 as final


class FinalAdmissionTests(unittest.TestCase):
    def test_failed_gate_stops_before_input_load(self):
        with patch.object(final, "CLAIM", Mock(exists=lambda: False)), \
             patch.object(final, "OUTPUT", Mock(exists=lambda: False)), \
             patch.object(final, "check", return_value={"state": "verified", "development_gate_passed": False}), \
             patch.object(final.run, "load_inputs") as inputs:
            with self.assertRaisesRegex(ValueError, "gate did not pass"):
                final.admission()
            inputs.assert_not_called()

    def test_existing_claim_stops_before_verification(self):
        with patch.object(final, "CLAIM", Mock(exists=lambda: True)), patch.object(final, "check") as verify:
            with self.assertRaisesRegex(ValueError, "no retry/resume"):
                final.admission()
            verify.assert_not_called()

    def test_total_budget_blocks_before_another_fit_is_claimed(self):
        claim = {"attempted_fits": 1, "attempted_vectorizer_fits": 4}
        with patch.object(final.run.dev, "consume") as consume:
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                final.consume(claim, "attempted_fits", {"max_total_fit_attempts": 5})
            consume.assert_not_called()


if __name__ == "__main__":
    unittest.main()
