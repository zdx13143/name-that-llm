"""Build an allowlisted R7 frozen model bundle after the requested final fit."""
import json
from pathlib import Path
import shutil

from inference_round7 import digest, write_json

MODULE = Path(__file__).resolve().parents[1]
FINAL = MODULE / "outputs/windows_external_requested_final_20261026_r7"
ASSETS = MODULE / "outputs/kaggle_r7_assets/ntl-r7-assets"


def main():
    claim_path = MODULE / "outputs/continuation_claims/windows-external-weighting-requested-final-20261026-r7.json"
    claim = json.loads(claim_path.read_bytes())
    summary = json.loads((FINAL / "summary.json").read_bytes())
    receipt = json.loads((FINAL / "supervisor.json").read_bytes())
    contract = json.loads((FINAL / "contract.json").read_bytes())
    if (claim["state"] != "completed" or receipt["state"] != "completed" or receipt["exit_code"] != 0
            or claim["summary_sha256"] != digest(FINAL / "summary.json")
            or claim["supervisor_sha256"] != digest(FINAL / "supervisor.json")
            or summary["development_gate_passed"] is not False
            or summary["authorization"]["mode"] != "explicit_user_request_after_failed_development_gate"
            or claim["attempted_fits"] != 1 or claim["completed_fits"] != 1
            or claim["attempted_vectorizer_fits"] != 4 or claim["completed_vectorizer_fits"] != 4):
        raise ValueError("Requested final fit is not complete or evidence changed")
    for name in ("weighted_svc.joblib", "synthetic_scores.npz"):
        key = "model_sha256" if name.endswith("joblib") else "synthetic_scores_sha256"
        if digest(FINAL / name) != summary[key]:
            raise ValueError("Frozen export bytes changed")
    source_hashes = {name.replace("\\", "/"): value for name, value in contract["sources"].items()}
    for name in ("stylegram_round5.py", "weighted_style_round7.py"):
        if digest(MODULE / "scripts" / name) != source_hashes["scripts/" + name]:
            raise ValueError("Training preprocessing source changed")
    ASSETS.mkdir(parents=True, exist_ok=False)
    for name in ("weighted_svc.joblib", "synthetic_scores.npz"):
        shutil.copyfile(FINAL / name, ASSETS / name)
    for name in ("stylegram_round5.py", "weighted_style_round7.py", "inference_round7.py"):
        shutil.copyfile(MODULE / "scripts" / name, ASSETS / name)
    shutil.copyfile(MODULE / "config/labels.json", ASSETS / "labels.json")
    shutil.copyfile(MODULE / "config/round7_external_sources_20261026.json", ASSETS / "external_source_manifest.json")
    (ASSETS / "README.txt").write_text(
        "Private frozen R7 four-view SVC, official sample weight 4.\n"
        "Final training uses 2159 development rows plus the fixed public external pool.\n"
        "External data source/version/file hashes are recorded in external_source_manifest.json; no raw passages are included.\n"
        "Hailey and Michael sources were excluded. Method reference: https://www.kaggle.com/code/zdx13143/ntl-ensemble-600?scriptVersionId=360258830\n"
        "This is a standalone SVC component, not the full 0.601 ensemble. The development incremental gate failed; the user explicitly requested one public-score observation.\n"
        "No original competition records, protected holdout, OOF, credentials, encoder or test-time training are included.\n", encoding="utf-8")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(ASSETS.iterdir())}
    manifest = {"candidate": claim["experiment_id"], "official_weight": 4, "development_gate_passed": False,
                "final_contract_sha256": claim["contract_sha256"], "final_claim_sha256": digest(claim_path),
                "final_summary_sha256": digest(FINAL / "summary.json"), "files": files}
    write_json(ASSETS / "manifest.json", manifest, True)
    print(json.dumps({"state": "assets_built", "manifest_sha256": digest(ASSETS / "manifest.json"), "files": len(files)}))


if __name__ == "__main__":
    main()
