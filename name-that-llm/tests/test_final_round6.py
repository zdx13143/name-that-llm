import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import windows_calibrated_final_round6 as final


class AdmissionTests(unittest.TestCase):
    def test_failed_development_gate_blocks_before_data(self):
        with patch.object(final,"check",return_value={"state":"verified","development_gate_passed":False}),patch.object(final.devrun,"load_inputs",side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError,"gate"):
                final.admission()

    def test_existing_final_claim_prevents_read_or_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"claim.json"
            path.write_text("{}")
            with patch.object(final,"CLAIM",path),patch.object(final,"check",side_effect=AssertionError("must not verify")):
                with self.assertRaisesRegex(ValueError,"already exists"):
                    final.admission()

    def test_total_final_fit_limit(self):
        config,_=final.devrun.configuration()
        with patch.object(final.devrun.dev,"consume",side_effect=AssertionError("must not consume")):
            with self.assertRaises(RuntimeError):
                final.consume({"attempted_fits":8,"attempted_temperature_fits":2},"attempted_fits",config["conditional_deployment_budget"])


if __name__=="__main__": unittest.main()
