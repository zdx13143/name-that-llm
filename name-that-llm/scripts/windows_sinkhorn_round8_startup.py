"""One zero-transform startup correction; original R8 evidence stays closed."""
import json
from pathlib import Path

import windows_sinkhorn_round8 as experiment
from supervisor_round7 import require_live_supervisor


def require_supervisor(claim):
    if any(claim.get(key) != 0 for key in ("attempted_transforms", "completed_transforms",
            "model_fit_calls", "model_prediction_calls", "new_encoder_forward_calls", "old_holdout_evaluations")):
        raise RuntimeError("R8 startup requires an unused claim")
    # Reuse the proven native Windows ancestry guard. Its R7 operation counters
    # are zero here: this experiment has no fitting or encoder operations.
    require_live_supervisor({**claim, **{key: 0 for key in (
        "attempted_forward_calls", "completed_forward_calls", "attempted_fits", "completed_fits",
        "attempted_vectorizer_fits", "completed_vectorizer_fits")}})


def configure():
    original_config = experiment.CONFIG
    config = experiment.MODULE / "config/windows_sinkhorn_round8_startup_20261026.json"
    expected = "36816890fe1e6621fa466a7c2d90b1dee32fd40f3b82d96715297080b9fb6b4b"
    if experiment.digest(config) != expected:
        raise ValueError("Startup correction config identity changed")
    value = json.loads(config.read_bytes())
    original = json.loads(original_config.read_bytes())
    correction = value["startup_correction"]
    if experiment.digest(original_config) != correction["original_config_sha256"]:
        raise ValueError("Original R8 config changed")
    if experiment.digest(experiment.CLAIM) != correction["failed_claim_sha256"]:
        raise ValueError("Failed R8 startup evidence changed")
    failed = json.loads(experiment.CLAIM.read_bytes())
    if (failed["state"] != "failed" or failed["contract_sha256"] != correction["failed_contract_sha256"]
            or any(failed[key] != 0 for key in ("attempted_transforms", "completed_transforms",
                "model_fit_calls", "model_prediction_calls", "new_encoder_forward_calls", "old_holdout_evaluations"))):
        raise ValueError("Only a zero-transform startup failure is eligible")
    for key in original:
        if key not in ("experiment_id", "preserved_claims", "budget") and value[key] != original[key]:
            raise ValueError(f"Scientific recipe changed: {key}")
    if ({k: v for k, v in value["budget"].items() if k != "max_wall_seconds"}
            != {k: v for k, v in original["budget"].items() if k != "max_wall_seconds"}
            or value["budget"]["max_wall_seconds"] + failed["elapsed_seconds"] > original["budget"]["max_wall_seconds"]):
        raise ValueError("Combined budget changed")
    experiment.CONFIG = config
    experiment.CONFIG_SHA256 = expected
    experiment.CLAIM = experiment.MODULE / "outputs/continuation_claims/windows-frozen-svc-sinkhorn-dev-20261026-r8-startup-1.json"
    experiment.OUTPUT = experiment.MODULE / "outputs/windows_sinkhorn_20261026_r8_startup1"
    experiment.ENTRYPOINT = Path(__file__).resolve()
    experiment.SUPERVISOR_CHECK = require_supervisor
    experiment.SOURCES = (*experiment.SOURCES, config, Path(__file__).resolve(),
                          Path(__file__).with_name("supervisor_round7.py"),
                          experiment.MODULE / "docs/WINDOWS_ROUND8_STARTUP_20261026.md")


if __name__ == "__main__":
    configure()
    experiment.main()
