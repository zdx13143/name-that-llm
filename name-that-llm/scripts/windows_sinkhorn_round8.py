"""One supervised R8 evaluation of frozen OOF; defaults to read-only readiness."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

from probe_modernbert import canonical, digest, write_json
from sinkhorn_round8 import balance, metrics, paired_comparison, gate

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_sinkhorn_round8_20261026.json"
CONFIG_SHA256 = "6d6b54b5fbf557150424ba2a5ac3473e1e448adb5233a1214d176d49f17e5389"
OUTPUT = MODULE / "outputs/windows_sinkhorn_20261026_r8"
CLAIM = MODULE / "outputs/continuation_claims/windows-frozen-svc-sinkhorn-dev-20261026-r8.json"
SOURCES = (CONFIG, Path(__file__), Path(__file__).with_name("sinkhorn_round8.py"),
           Path(__file__).with_name("probe_modernbert.py"),
           MODULE / "docs/WINDOWS_ROUND8_PLAN_20261026.md")
IDENTITY_KEYS = ("row_index", "ids", "groups", "truth", "labels", "fold_id")
ENTRYPOINT = Path(__file__).resolve()


def require_supervisor(claim):
    parent = psutil.Process(os.getppid())
    if (claim["state"] != "started" or parent.pid != claim["supervisor_pid"]
            or abs(parent.create_time() - claim["supervisor_create_time"]) > .01):
        raise RuntimeError("R8 execution requires its live supervisor")


SUPERVISOR_CHECK = require_supervisor


def object_hash(value):
    import hashlib
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def configuration():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("R8 configuration identity changed")
    value = json.loads(CONFIG.read_bytes())
    if value["experiment_id"] != CLAIM.stem:
        raise ValueError("R8 experiment identity changed")
    for name, version in value["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Runtime version changed: {name}")
    if sys.platform != "win32":
        raise ValueError("This registered run requires native Windows")
    return value


def verify_files(config):
    for name, expected in {**config["input_sha256"], **config["preserved_claims"]}.items():
        path = (MODULE / name).resolve()
        if not path.is_relative_to(MODULE) or digest(path) != expected:
            raise ValueError(f"Frozen file changed: {name}")


def validate_arrays(data, reference):
    for key in IDENTITY_KEYS:
        np.testing.assert_array_equal(data[key], reference[key], err_msg=f"Identity changed: {key}")
    if (len(data["ids"]) != 2159 or len(np.unique(data["ids"])) != 2159
            or len(np.unique(data["groups"])) != 120 or set(data["fold_id"]) != set(range(5))
            or len(data["labels"]) != 18 or len(set(data["labels"].tolist())) != 18):
        raise ValueError("Original development axes changed")
    for group in np.unique(data["groups"]):
        if len(np.unique(data["fold_id"][data["groups"] == group])) != 1:
            raise ValueError("Prompt group crosses validation folds")
    p = data["probabilities"]
    if p.shape != (2159, 18) or p.dtype != np.float64 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid frozen R7 probabilities")
    if not np.allclose(p.sum(1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Frozen R7 probabilities are not row-normalized")


def load_inputs(config):
    verify_files(config)
    old_claim = json.loads((MODULE / "outputs/continuation_claims/windows-external-weighting-dev-20261026-r7-startup-1.json").read_bytes())
    folder = MODULE / config["baseline_output"]
    old_contract = json.loads((folder / "contract.json").read_bytes())
    if (old_claim["state"] != "completed" or old_claim["contract_sha256"] != config["baseline_contract_sha256"]
            or object_hash(old_contract) != config["baseline_contract_sha256"]
            or digest(folder / "summary.json") != old_claim["summary_sha256"]
            or digest(folder / "supervisor.json") != old_claim["supervisor_sha256"]):
        raise ValueError("R7 completed evidence is inconsistent")
    with np.load(folder / "candidate_oof.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    ref_folder = MODULE / config["reference_output"]
    reference_contract = json.loads((ref_folder / "contract.json").read_bytes())
    if object_hash(reference_contract) != config["reference_contract_sha256"]:
        raise ValueError("Original fold reference contract changed")
    with np.load(ref_folder / "oof.npz", allow_pickle=False) as saved:
        reference = {key: saved[key].copy() for key in IDENTITY_KEYS}
        if str(saved["contract_sha256"]) != config["reference_contract_sha256"]:
            raise ValueError("Original fold reference OOF identity changed")
    if str(data["contract_sha256"]) != config["baseline_contract_sha256"]:
        raise ValueError("Frozen R7 OOF contract changed")
    validate_arrays(data, reference)
    return data


def contract(config, data):
    return {"config": config, "sources": {str(p.relative_to(MODULE)).replace("\\", "/"): digest(p) for p in SOURCES},
            "identity_sha256": object_hash({key: data[key].tolist() for key in IDENTITY_KEYS}),
            "python": sys.version, "platform": sys.platform,
            "runtime": {name: importlib.metadata.version(name) for name in config["runtime"]}}


def readiness():
    config = configuration()
    if CLAIM.exists() or OUTPUT.exists():
        raise FileExistsError("R8 claim/output already exists; retry and resume are forbidden")
    data = load_inputs(config)
    return config, data, contract(config, data)


def execute():
    config = configuration()
    claim = json.loads(CLAIM.read_bytes())
    SUPERVISOR_CHECK(claim)
    data = load_inputs(config)
    value = contract(config, data)
    if claim["contract_sha256"] != object_hash(value):
        raise ValueError("R8 execution contract changed")
    baseline = data["probabilities"]
    baseline_metrics = metrics(baseline, data["truth"], data["labels"])
    if abs(baseline_metrics["balanced_accuracy"] - config["evaluation"]["baseline_balanced_accuracy"]) > 1e-12:
        raise ValueError("R7 baseline metric identity changed")
    candidate = np.empty_like(baseline)
    folds = []
    for number in range(5):
        rows = np.flatnonzero(data["fold_id"] == number)
        claim["attempted_transforms"] += 1
        write_json(CLAIM, claim, exclusive=False)
        transformed = balance(baseline[rows])
        claim["completed_transforms"] += 1
        claim["attempted_transforms"] += 1
        write_json(CLAIM, claim, exclusive=False)
        reversed_result = balance(baseline[rows][::-1])[::-1]
        claim["completed_transforms"] += 1
        error = float(np.max(np.abs(transformed - reversed_result)))
        if error > 1e-10 or not np.array_equal(transformed.argmax(1), reversed_result.argmax(1)):
            raise ValueError("Row permutation changed predictions")
        candidate[rows] = transformed
        current = metrics(transformed, data["truth"][rows], data["labels"])
        previous = metrics(baseline[rows], data["truth"][rows], data["labels"])
        record = {"fold": number, "rows": len(rows), "groups": len(np.unique(data["groups"][rows])),
                  "candidate_ba": current["balanced_accuracy"], "baseline_ba": previous["balanced_accuracy"],
                  "permutation_max_abs_error": error,
                  "before_class_probability_mass": baseline[rows].sum(0).tolist(),
                  "after_class_probability_mass": transformed.sum(0).tolist(),
                  "candidate_predicted_counts": current["predicted_counts"]}
        folds.append(record)
        write_json(CLAIM, claim, exclusive=False)
        print(json.dumps(record), flush=True)
    candidate_metrics = metrics(candidate, data["truth"], data["labels"])
    comparison = paired_comparison(candidate, baseline, data["truth"], data["groups"],
                                   resamples=config["evaluation"]["bootstrap_resamples"],
                                   seed=config["evaluation"]["bootstrap_seed"])
    with (OUTPUT / "candidate_oof.npz").open("xb") as handle:
        np.savez_compressed(handle, probabilities=candidate, **{key: data[key] for key in IDENTITY_KEYS},
                            contract_sha256=np.asarray(claim["contract_sha256"]))
    verify_files(config)
    summary = {"experiment_id": config["experiment_id"], "contract_sha256": claim["contract_sha256"],
               "baseline_metrics": baseline_metrics, "candidate_metrics": candidate_metrics,
               "comparison": comparison, "folds": folds,
               "candidate_gate": gate(candidate_metrics, folds, comparison, config["candidate_gate"]),
               "candidate_oof_sha256": digest(OUTPUT / "candidate_oof.npz"),
               "attempted_transforms": claim["attempted_transforms"], "completed_transforms": claim["completed_transforms"],
               "model_fit_calls": 0, "model_prediction_calls": 0, "new_encoder_forward_calls": 0,
               "old_holdout_evaluations": 0, "new_kaggle_submissions": 0,
               "limitations": [config["evaluation"]["bootstrap_scope"],
                               "Uniform class probability mass is an assumed deployment prior, not verified hidden-test class counts.",
                               "Reuses adaptively selected R7 OOF and its upstream external-source curation/decontamination; not a blind holdout or full 0.642 ensemble."]}
    write_json(OUTPUT / "summary.json", summary)
    claim.update(state="worker_completed", summary_sha256=digest(OUTPUT / "summary.json"))
    write_json(CLAIM, claim, exclusive=False)


def supervise(config, value):
    claim = {"state": "started", "experiment_id": config["experiment_id"], "supervisor_pid": os.getpid(),
             "supervisor_create_time": psutil.Process().create_time(), "started_unix_seconds": time.time(),
             "contract_sha256": object_hash(value), "attempted_transforms": 0, "completed_transforms": 0,
             "model_fit_calls": 0, "model_prediction_calls": 0, "new_encoder_forward_calls": 0, "old_holdout_evaluations": 0}
    write_json(CLAIM, claim)
    started, peak, process = time.monotonic(), 0, None
    try:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        write_json(OUTPUT / "contract.json", value)
        for source in SOURCES:
            target = OUTPUT / "source_snapshot" / source.relative_to(MODULE)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               **{name: str(config["budget"]["threads"]) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-u", str(ENTRYPOINT), "--execute"],
                                       stdout=stdout, stderr=stderr, env=env)
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.monotonic() - started > config["budget"]["max_wall_seconds"]:
                    raise TimeoutError("R8 wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss + sum(p.memory_info().rss for p in monitored.children(recursive=True))
                    peak = max(peak, rss)
                    if rss > config["budget"]["max_host_rss_gib"] * 2**30:
                        raise MemoryError("R8 RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.1)
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed"
                or claim["attempted_transforms"] != config["budget"]["max_transform_attempts"]
                or claim["completed_transforms"] != config["budget"]["max_transform_attempts"]):
            raise RuntimeError("R8 worker did not complete its exact transform budget")
        verify_files(config)
        summary = json.loads((OUTPUT / "summary.json").read_bytes())
        receipt = {"state": "completed", "exit_code": process.returncode, "wall_seconds": time.monotonic() - started,
                   "sampled_peak_process_tree_rss_bytes": peak, "summary_sha256": digest(OUTPUT / "summary.json"),
                   "candidate_development_gate_passed": summary["candidate_gate"]["passed"]}
        write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(OUTPUT / "supervisor.json"))
        write_json(CLAIM, claim, exclusive=False)
        print(json.dumps(receipt, indent=2))
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        claim = json.loads(CLAIM.read_bytes())
        claim.update(state="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        write_json(CLAIM, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        execute()
        return
    try:
        config, data, value = readiness()
    except Exception as exc:
        print(json.dumps({"state": "blocked", "reason": str(exc)}))
        raise SystemExit(2)
    if args.run:
        supervise(config, value)
    else:
        print(json.dumps({"state": "ready", "rows": len(data["ids"]), "groups": len(np.unique(data["groups"])),
                          "candidate_evaluated": False, "contract_sha256": object_hash(value)}, indent=2))


if __name__ == "__main__":
    main()
