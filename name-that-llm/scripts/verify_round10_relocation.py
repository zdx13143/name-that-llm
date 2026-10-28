"""Validate the exact R10 payload in Linux on eighty artificial texts only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from inference_round10 import digest, write_json


def main():
    if sys.platform != "linux":
        raise RuntimeError("Linux relocation validation required")
    module = Path(__file__).resolve().parents[1]
    package = module / "outputs/kaggle_r10_package"
    target = package / "relocation_verification.json"
    claim_path = package / "relocation_claim.json"
    if target.exists() or claim_path.exists():
        raise FileExistsError("Preserve prior relocation attempt")
    receipt = json.loads((package / "package_receipt.json").read_bytes())
    payload = package / receipt["payload_file"]
    if digest(payload) != receipt["payload_sha256"]:
        raise ValueError("Payload changed")
    destination = Path(tempfile.mkdtemp(prefix="ntl-r10-relocate-"))
    claim = {"state": "started", "payload_sha256": receipt["payload_sha256"], "temporary_directory": str(destination),
             "model_fits": 0, "old_holdout_evaluations": 0}
    write_json(claim_path, claim, True)
    try:
        with tarfile.open(payload, "r:gz") as archive:
            members = archive.getmembers()
            if {Path(member.name).parts[0] for member in members} != {"ntl-r10-assets", "cpython-3.11.17-linux-x86_64-gnu"}:
                raise ValueError("Unexpected archive roots")
            archive.extractall(destination, filter="data")
        python = destination / "cpython-3.11.17-linux-x86_64-gnu/bin/python3.11"
        assets = destination / "ntl-r10-assets"
        result_dir = destination / "synthetic-check"
        env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with (package / "relocation_stdout.log").open("x", encoding="utf-8") as stdout, \
             (package / "relocation_stderr.log").open("x", encoding="utf-8") as stderr:
            result = subprocess.run([str(python), str(assets / "inference_round10.py"), "synthetic", "--assets", str(assets),
                                     "--output", str(result_dir), "--manifest-sha256", receipt["manifest_sha256"]],
                                    cwd=assets, env=env, stdout=stdout, stderr=stderr, text=True, timeout=900)
        if result.returncode != 0:
            raise RuntimeError(f"Relocated inference failed: exit {result.returncode}")
        proof = json.loads((result_dir / "claim.json").read_bytes())
        if (proof["state"] != "completed" or proof["rows"] != 80 or proof["completed_prediction_calls"] != 3
                or proof["attempted_prediction_calls"] != 3 or proof["grouping_vectorizer_fit_attempts"] != 2
                or proof["synthetic_max_abs_error"] > 1e-7 or proof["model_fits"] != 0):
            raise ValueError("Incomplete relocation proof")
        proof.update(state="verified", payload_sha256=receipt["payload_sha256"], archive_members=len(members), temporary_directory=str(destination))
        write_json(target, proof, True)
        claim.update(state="completed", verification_sha256=digest(target))
        write_json(claim_path, claim)
        print(json.dumps(proof), flush=True)
    except BaseException as error:
        claim.update(state="failed", error_type=type(error).__name__, error=str(error))
        write_json(claim_path, claim)
        raise


if __name__ == "__main__":
    main()
