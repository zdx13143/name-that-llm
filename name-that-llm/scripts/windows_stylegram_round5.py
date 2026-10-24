"""One registered CPU stylegram/frozen-semantic fusion experiment. Default: readiness."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import warnings

import joblib
import numpy as np
import psutil
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import windows_modernbert_dev as dev
from probe_modernbert import digest, write_json
from windows_prompt_gaussian_round3 import paired_comparison
from stylegram_round5 import make_model, predict_scores, blend_scores

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_stylegram_round5_20261025.json"
CONFIG_SHA256 = "c83166e0cf7cdbeb72732dfc1d34a158902b45e54a8c804b2696437fb5f93888"
BASE = MODULE / "config/windows_modernbert_20261023.json"
OUTPUT = MODULE / "outputs/windows_stylegram_20261025_r5"
CLAIM = MODULE / "outputs/continuation_claims/windows-stylegram-fusion-dev-20261025-r5.json"
SOURCES = (Path(__file__), Path(__file__).with_name("stylegram_round5.py"),
           Path(__file__).with_name("windows_modernbert_dev.py"), Path(__file__).with_name("probe_modernbert.py"),
           Path(__file__).with_name("restore_team_split.py"), Path(__file__).with_name("windows_prompt_gaussian_round3.py"),
           Path(__file__).with_name("prompt_gaussian_head.py"), CONFIG, BASE, MODULE / "config/labels.json")


def configuration():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("Registered Round5 config identity differs")
    config, base = json.loads(CONFIG.read_bytes()), json.loads(BASE.read_bytes())
    if config["experiment_id"] != "windows-stylegram-fusion-dev-20261025-r5":
        raise ValueError("Unexpected experiment")
    return config, base


def load_inputs(config, base):
    dev.verify_preserved_claims(config)
    data = dev.load_development(MODULE / "data/raw/train.csv", base["input"], MODULE / "config/labels.json")
    folds, fold_id = dev.grouped_folds(data, config["evaluation"]["folds"])
    reference = json.loads((MODULE / config["fold_reference_output"] / "contract.json").read_bytes())
    if (dev.object_hash(reference) != config["fold_reference_contract_sha256"]
            or dev.object_hash(fold_id.tolist()) != reference["fold_sha256"]
            or dev.object_hash(data["texts"].tolist()) != reference["text_sha256"]):
        raise ValueError("Original development text/fold identity differs")
    previous = MODULE / config["baseline_output"]
    prior_contract = json.loads((previous / "contract.json").read_bytes())
    prior_claim = json.loads((MODULE / "outputs/continuation_claims/windows-prompt-gaussian-dev-20261024-r3.json").read_bytes())
    if (dev.object_hash(prior_contract) != config["baseline_contract_sha256"]
            or prior_claim["state"] != "completed"
            or prior_claim["contract_sha256"] != config["baseline_contract_sha256"]
            or digest(previous / "summary.json") != prior_claim["summary_sha256"]
            or digest(previous / "supervisor.json") != prior_claim["supervisor_sha256"]
            or digest(previous / "oof.npz") != config["baseline_oof_sha256"]):
        raise ValueError("Completed Round3 artifact identity differs")
    with np.load(previous / "oof.npz", allow_pickle=False) as artifact:
        if set(artifact.files) != {"probabilities", "row_index", "ids", "groups", "truth", "labels", "fold_id", "contract_sha256"}:
            raise ValueError("Baseline OOF schema differs")
        for key in ("row_index", "ids", "groups", "truth", "labels"):
            if not np.array_equal(artifact[key], data[key]):
                raise ValueError("Baseline OOF row/label identity differs")
        if (not np.array_equal(artifact["fold_id"], fold_id)
                or str(artifact["contract_sha256"]) != config["baseline_contract_sha256"]):
            raise ValueError("Baseline OOF fold/contract identity differs")
        baseline = dev.probabilities(artifact["probabilities"], len(fold_id), len(data["labels"]))
    return data, baseline, folds, fold_id


def contract(config, base, data, fold_id):
    return {"config": config, "base_config": base,
            "sources": {str(path.relative_to(MODULE)): digest(path) for path in SOURCES},
            "text_sha256": dev.object_hash(data["texts"].tolist()),
            "row_identity_sha256": dev.object_hash({key: data[key].tolist() for key in ("row_index", "ids", "groups", "truth", "labels")}),
            "fold_sha256": dev.object_hash(fold_id.tolist()),
            "package_versions": {name: importlib.metadata.version(name) for name in config["runtime"]},
            "python": sys.version, "platform": sys.platform}


def readiness(config, base):
    blockers = []
    if sys.platform != "win32":
        blockers.append("Registered runtime is native Windows")
    if CLAIM.exists() or OUTPUT.exists():
        blockers.append("Fixed claim/output already exists; no retry or resume")
    for name, expected in config["runtime"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            blockers.append(f"Runtime {name}: {actual}; expected {expected}")
    if blockers:
        return {"state": "blocked", "blockers": blockers}, None
    try:
        data, _, _, fold_id = load_inputs(config, base)
        value = contract(config, base, data, fold_id)
    except (OSError, ValueError) as error:
        return {"state": "blocked", "blockers": [f"{type(error).__name__}: {error}"]}, None
    return {"state": "ready", "rows": len(fold_id), "contract_sha256": dev.object_hash(value)}, value


def metric_gate(metrics, comparison, control_comparison, config):
    gate = config["candidate_gate"]
    folds = metrics["fold_balanced_accuracy"]
    values = [metrics["balanced_accuracy"], *folds, comparison["pooled_ba_gain"],
              *comparison["prompt_paired_gain_95ci"], *comparison["fold_ba_gains"], control_comparison["pooled_ba_gain"]]
    checks = {
        "finite_complete": bool(np.isfinite(values).all() and len(folds) == 5 and len(comparison["fold_ba_gains"]) == 5
                                and all(0 <= v <= 1 for v in [metrics["balanced_accuracy"], *folds])),
        "pooled": metrics["balanced_accuracy"] >= gate["min_pooled_balanced_accuracy"],
        "each_fold": min(folds, default=-1) >= gate["min_each_fold_balanced_accuracy"],
        "paired_gain": comparison["pooled_ba_gain"] >= gate["min_paired_gain_over_round3"],
        "paired_interval": comparison["prompt_paired_gain_95ci"][0] > gate["paired_prompt_95ci_lower_exceeds"],
        "improved_folds": comparison["improved_folds"] >= gate["min_improved_folds_over_round3"],
        "not_worse_than_style_control": control_comparison["pooled_ba_gain"] >= gate["min_paired_gain_over_style_control"]}
    return {"metric_gate_passed": all(checks.values()), "checks": checks,
            "requires_completed_supervisor": True, "deployment_validation_complete": False}


def guard_violation(claim, budget):
    """Worker-side backstop; parent supervision remains the primary enforcer."""
    try:
        parent = psutil.Process(claim["supervisor_pid"])
        if parent.create_time() != claim["supervisor_create_time"] or not parent.is_running():
            return "Supervisor identity/liveness lost"
        if time.time() - claim["started_unix_seconds"] > budget["development_timeout_seconds"]:
            return "Worker wall-clock budget exceeded"
        rss = parent.memory_info().rss + sum(p.memory_info().rss for p in parent.children(recursive=True))
        if rss > budget["max_host_rss_gib"] * 2**30:
            return "Worker process-tree RSS budget exceeded"
    except psutil.Error:
        return "Unable to verify live supervisor/resource budget"
    return None


def start_worker_guard(claim, budget):
    def watch():
        while True:
            reason = guard_violation(claim, budget)
            if reason:
                print(reason, file=sys.stderr, flush=True)
                # This worker uses n_jobs=1 and creates no child processes.
                os._exit(3)
            time.sleep(.2)
    threading.Thread(target=watch, daemon=True, name="round5-budget-guard").start()


def terminate_tree(process):
    """Kill and wait for every observed descendant, including the venv redirector."""
    try:
        root = psutil.Process(process.pid)
        processes = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        processes = []
    for item in reversed(processes):
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    process.poll()
    return {"observed_pids": [p.pid for p in processes], "remaining_pids": [p.pid for p in alive],
            "worker_still_running": process.poll() is None, "returncode": process.returncode}


def execute(config, base):
    claim = json.loads(CLAIM.read_bytes())
    dev.require_live_supervisor(claim)
    start_worker_guard(claim, config["budget"])
    data, baseline, folds, fold_id = load_inputs(config, base)
    if dev.object_hash(contract(config, base, data, fold_id)) != claim["contract_sha256"]:
        raise ValueError("Contract changed after readiness")
    style = np.full_like(baseline, np.nan)
    records = []
    with threadpool_limits(limits=config["budget"]["threads"]):
        for number, (train, valid) in enumerate(folds):
            model = make_model(config["method"])
            dev.consume(claim, "attempted_fits", config["budget"]["max_classifier_fit_attempts"], CLAIM)
            started = time.perf_counter()
            print(f"fold={number} fit_started train={len(train)} valid={len(valid)}", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(data["texts"][train].tolist(), data["truth"][train])
            claim["completed_fits"] += 1
            write_json(CLAIM, claim, exclusive=False)
            probabilities = predict_scores(model, data["texts"][valid].tolist())
            reverse = predict_scores(model, data["texts"][valid[::-1]].tolist())[::-1]
            single = predict_scores(model, data["texts"][valid[:1]].tolist())
            error = max(float(np.max(np.abs(reverse - probabilities))), float(np.max(np.abs(single - probabilities[:1]))))
            if error > 1e-10:
                raise ValueError("Inference batch/order invariance failed")
            path = OUTPUT / f"style_fold{number}.joblib"
            with path.open("xb") as handle:
                joblib.dump(model, handle, compress=3)
            restored = joblib.load(path)
            export_error = float(np.max(np.abs(predict_scores(restored, data["texts"][valid[:3]].tolist()) - probabilities[:3])))
            if export_error > 1e-10:
                raise ValueError("Exported style model changed")
            style[valid] = probabilities
            records.append({"fold": number, "fit_and_predict_seconds": time.perf_counter() - started,
                            "inference_invariance_max_abs_error": error, "export_max_abs_error": export_error,
                            "model_sha256": digest(path), "svc_iterations": int(model[-1].n_iter_),
                            "vocabulary_sizes": {name: len(view.vocabulary_) for name, view in model[0].transformer_list}})
            print(f"fold={number} completed seconds={records[-1]['fit_and_predict_seconds']:.2f}", flush=True)
            del model, restored
    style = dev.probabilities(style, len(fold_id), 18)
    candidate = blend_scores(style, baseline, config["method"]["style_weight"])
    dev.save_array(OUTPUT / "style_oof.npz", "probabilities", style, data, fold_id, claim["contract_sha256"])
    dev.save_array(OUTPUT / "oof.npz", "probabilities", candidate, data, fold_id, claim["contract_sha256"])
    metrics = dev.metrics(candidate, data, fold_id, config["evaluation"])
    comparison = paired_comparison(candidate, baseline, data, fold_id, config["evaluation"])
    control = paired_comparison(candidate, style, data, fold_id, config["evaluation"])
    report = {"scope": config["scope"], "contract_sha256": claim["contract_sha256"],
              "metrics": metrics, "style_control_metrics": dev.metrics(style, data, fold_id, config["evaluation"]),
              "paired_round3_comparison": comparison, "paired_style_control_comparison": control,
              "candidate_gate": metric_gate(metrics, comparison, control, config), "folds": records,
              "oof_sha256": digest(OUTPUT / "oof.npz"), "style_oof_sha256": digest(OUTPUT / "style_oof.npz"),
              "attempted_fits": claim["attempted_fits"], "completed_fits": claim["completed_fits"],
              "new_encoder_forward_calls": 0, "final_holdout_evaluations": 0, "full_data_final_fits": 0,
              "conditional_final_fits": 0, "new_kaggle_submissions": 0,
              "limitations": "Repeated adaptive development on original 120 prompts. Descriptive interval, no independent holdout. SVC softmax is uncalibrated. No fallback selection."}
    dev.verify_preserved_claims(config)
    write_json(OUTPUT / "summary.json", report)
    claim.update(state="worker_completed", summary_sha256=digest(OUTPUT / "summary.json"))
    write_json(CLAIM, claim, exclusive=False)


def supervise(config, value):
    claim = {"state": "started", "experiment_id": config["experiment_id"],
             "supervisor_pid": os.getpid(), "supervisor_create_time": psutil.Process().create_time(),
             "contract_sha256": dev.object_hash(value), "attempted_forward_calls": 0, "completed_forward_calls": 0,
             "attempted_fits": 0, "completed_fits": 0, "final_holdout_evaluations": 0,
             "full_data_final_fits": 0, "started_unix_seconds": time.time()}
    write_json(CLAIM, claim)
    process, phase = None, "pre_spawn"
    started, peak_rss = time.perf_counter(), 0
    try:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        write_json(OUTPUT / "contract.json", value)
        for source in SOURCES:
            target = OUTPUT / "source_snapshot" / source.relative_to(MODULE)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
        env = {**os.environ, **{key: str(config["budget"]["threads"]) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--execute"], env=env, stdout=stdout, stderr=stderr)
            phase = "worker"
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter() - started > config["budget"]["development_timeout_seconds"]:
                    raise TimeoutError("Round5 wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss
                    rss += sum(child.memory_info().rss for child in monitored.children(recursive=True))
                    peak_rss = max(peak_rss, rss)
                    if rss > config["budget"]["max_host_rss_gib"] * 2**30:
                        raise MemoryError("Round5 process-tree RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed"
                or claim["attempted_fits"] != 5 or claim["completed_fits"] != 5
                or claim["attempted_forward_calls"] != 0 or claim["completed_forward_calls"] != 0
                or digest(OUTPUT / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Round5 worker failed or returned incomplete artifacts")
        dev.verify_preserved_claims(config)
        if any(digest(MODULE / relative) != expected for relative, expected in value["sources"].items()):
            raise ValueError("Execution sources changed during this round")
        summary = json.loads((OUTPUT / "summary.json").read_bytes())
        gate = metric_gate(summary["metrics"], summary["paired_round3_comparison"], summary["paired_style_control_comparison"], config)
        receipt = {"state": "completed", "exit_code": 0, "wall_seconds": time.perf_counter() - started,
                   "sampled_peak_process_tree_rss_bytes": peak_rss, "summary_sha256": claim["summary_sha256"],
                   "candidate_development_gate_passed": gate["metric_gate_passed"], "candidate_deployment_validation_complete": False}
        write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(OUTPUT / "supervisor.json"))
        write_json(CLAIM, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = terminate_tree(process)
        except BaseException as cleanup_error:
            termination = {"cleanup_exception": type(cleanup_error).__name__}
        finally:
            claim = json.loads(CLAIM.read_bytes())
            claim.update(state="failed", exception_type=type(error).__name__, exception_message=str(error),
                         phase=phase, termination=termination, wall_seconds=time.perf_counter() - started,
                         sampled_peak_process_tree_rss_bytes=peak_rss)
            write_json(CLAIM, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config, base = configuration()
    if args.execute:
        execute(config, base)
        return 0
    state, value = readiness(config, base)
    if not args.run or state["state"] != "ready":
        print(json.dumps(state))
        return 0 if state["state"] == "ready" else 2
    print(json.dumps(supervise(config, value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
