"""One frozen startup correction; the failed original claim remains closed."""
import json
from pathlib import Path

import windows_external_round7 as experiment
from supervisor_round7 import require_live_supervisor


def configure():
    original_config = experiment.CONFIG
    config = experiment.MODULE / "config/windows_external_round7_startup_20261026.json"
    expected = "193e7e26c33357fa69e26ea3bd3cc85fe58323c8dfe1291e30fd3c3879d085ca"
    if experiment.digest(config) != expected:
        raise ValueError("Startup correction config identity changed")
    value = json.loads(config.read_bytes())
    original = json.loads(original_config.read_bytes())
    correction = value["startup_correction"]
    if experiment.digest(original_config) != correction["original_config_sha256"]:
        raise ValueError("Original config changed")
    old_claim = experiment.CLAIM
    if experiment.digest(old_claim) != correction["failed_claim_sha256"]:
        raise ValueError("Failed startup evidence changed")
    failed = json.loads(old_claim.read_bytes())
    if (failed["state"] != "failed" or failed["contract_sha256"] != correction["failed_contract_sha256"]
            or any(failed[k] != 0 for k in ("attempted_fits", "completed_fits", "attempted_vectorizer_fits",
                                           "completed_vectorizer_fits", "attempted_forward_calls", "completed_forward_calls"))):
        raise ValueError("Only a zero-fit startup failure is eligible")
    for key in original:
        if key not in ("experiment_id", "preserved_claims", "budget") and value[key] != original[key]:
            raise ValueError(f"Scientific recipe changed: {key}")
    if ({k: v for k, v in value["budget"].items() if k != "development_timeout_seconds"}
            != {k: v for k, v in original["budget"].items() if k != "development_timeout_seconds"}
            or value["budget"]["development_timeout_seconds"] + failed["wall_seconds"] > original["budget"]["development_timeout_seconds"]):
        raise ValueError("Combined budget changed")
    experiment.CONFIG = config
    experiment.CONFIG_SHA256 = expected
    experiment.CLAIM = experiment.MODULE / "outputs/continuation_claims/windows-external-weighting-dev-20261026-r7-startup-1.json"
    experiment.OUTPUT = experiment.MODULE / "outputs/windows_external_weighting_20261026_r7_startup1"
    experiment.ENTRYPOINT = Path(__file__).resolve()
    experiment.SUPERVISOR_CHECK = require_live_supervisor
    experiment.SOURCES = (*experiment.SOURCES, config, Path(__file__).resolve(),
                          Path(__file__).with_name("supervisor_round7.py"))


if __name__ == "__main__":
    configure()
    raise SystemExit(experiment.main())
