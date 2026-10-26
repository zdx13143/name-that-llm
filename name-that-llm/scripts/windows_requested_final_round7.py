"""One explicitly requested R7 submission; preserve its failed development gate."""
import json
from pathlib import Path

import windows_external_final_round7 as final

REQUEST = final.MODULE / "config/round7_requested_submission_20261026.json"
REQUEST_SHA256 = "746582b1577a6da0837789f17bac56c28c2250bfdbb9e24bcdf006f04b490799"
DEVELOPMENT = final.MODULE / "outputs/windows_external_weighting_20261026_r7_startup1"
DEVELOPMENT_CLAIM = final.MODULE / "outputs/continuation_claims/windows-external-weighting-dev-20261026-r7-startup-1.json"


def authorization(verified):
    if final.run.digest(REQUEST) != REQUEST_SHA256:
        raise ValueError("Explicit submission request config changed")
    request = json.loads(REQUEST.read_bytes())
    if (verified["state"] != "verified" or verified["development_gate_passed"] is not False
            or request["development_gate_passed"] is not False
            or request["admission_mode"] != "explicit_user_request_after_failed_development_gate"):
        raise ValueError("This request only admits the verified failed-gate R7 candidate")
    protected = {
        DEVELOPMENT_CLAIM: request["development_claim_sha256"],
        DEVELOPMENT / "summary.json": request["development_summary_sha256"],
        DEVELOPMENT / "verification.json": request["development_verification_sha256"],
    }
    if any(final.run.digest(path) != expected for path, expected in protected.items()):
        raise ValueError("Completed R7 evidence changed")
    return {"mode": request["admission_mode"], "request_config_sha256": REQUEST_SHA256,
            "user_request": request["user_request"], "development_gate_passed": False,
            "final_budget": request["final_budget"], "submission_budget": request["submission_budget"]}


def configure_entrypoint():
    request = json.loads(REQUEST.read_bytes())
    if final.run.digest(REQUEST) != REQUEST_SHA256:
        raise ValueError("Explicit submission request config changed")
    final.CLAIM = final.MODULE / "outputs/continuation_claims" / (request["action_id"] + ".json")
    final.OUTPUT = final.MODULE / "outputs/windows_external_requested_final_20261026_r7"
    final.ENTRYPOINT = Path(__file__).resolve()
    final.EXTRA_SOURCES = (REQUEST, Path(__file__).resolve())
    final.authorize = authorization
    final.configure()
    config, _ = final.run.configuration()
    if any(config["conditional_final_budget"][key] != value for key, value in request["final_budget"].items()):
        raise ValueError("Requested final budget differs from the frozen recipe")
    if (config["method"]["candidate_official_weight"] != request["candidate"]["official_sample_weight"]
            or config["method"]["external_weight"] != request["candidate"]["external_sample_weight"]):
        raise ValueError("Requested candidate weights differ from the frozen recipe")


if __name__ == "__main__":
    configure_entrypoint()
    raise SystemExit(final.main())
