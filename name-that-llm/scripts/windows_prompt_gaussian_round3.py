"""One registered CPU head experiment on immutable Round2 development features."""
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
from threadpoolctl import threadpool_limits

import windows_modernbert_dev as dev
from prompt_gaussian_head import fit_head, predict_probabilities
from probe_modernbert import digest, write_json

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_prompt_gaussian_round3_20261024.json"
CONFIG_SHA256 = "be521933587ae0fd99086a2d1a52cb0dc67bea396ad81bdedfc9f9caa12f7091"
BASE = MODULE / "config/windows_modernbert_20261023.json"
OUTPUT = MODULE / "outputs/windows_prompt_gaussian_20261024_r3"
CLAIM = MODULE / "outputs/continuation_claims/windows-prompt-gaussian-dev-20261024-r3.json"
SOURCES = (Path(__file__), Path(__file__).with_name("prompt_gaussian_head.py"),
           Path(__file__).with_name("windows_modernbert_dev.py"), Path(__file__).with_name("probe_modernbert.py"),
           Path(__file__).with_name("restore_team_split.py"), CONFIG, BASE, MODULE / "config/labels.json",
           MODULE / "requirements-windows-dev.txt")


def configuration():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("Registered Round3 config identity differs")
    config, base = json.loads(CONFIG.read_bytes()), json.loads(BASE.read_bytes())
    if config["experiment_id"] != "windows-prompt-gaussian-dev-20261024-r3":
        raise ValueError("Unregistered experiment")
    return config, base


def preserved(config):
    for relative, expected in config["preserved_claims"].items():
        target = (MODULE / relative).resolve()
        if not target.is_relative_to((MODULE / "outputs/continuation_claims").resolve()) or digest(target) != expected:
            raise ValueError("Prior claim changed")


def load_inputs(config, base):
    preserved(config)
    prior = MODULE / config["prior_output"]
    prior_contract = json.loads((prior / "contract.json").read_bytes())
    prior_claim = json.loads((MODULE / "outputs/continuation_claims/windows-modernbert-frozen-dev-20261024-r2.json").read_bytes())
    if (dev.object_hash(prior_contract) != config["prior_contract_sha256"]
            or prior_claim["state"] != "completed"
            or prior_claim["contract_sha256"] != config["prior_contract_sha256"]
            or digest(prior / "summary.json") != prior_claim["summary_sha256"]
            or digest(prior / "supervisor.json") != prior_claim["supervisor_sha256"]
            or digest(prior / "features.npz") != config["features_sha256"]
            or digest(prior / "oof.npz") != config["baseline_oof_sha256"]):
        raise ValueError("Completed Round2 source identity differs")
    data = dev.load_development(MODULE / "data/raw/train.csv", base["input"])
    folds, fold_id = dev.grouped_folds(data)
    if (dev.object_hash(fold_id.tolist()) != prior_contract["fold_sha256"]
            or dev.object_hash(data["texts"].tolist()) != prior_contract["text_sha256"]):
        raise ValueError("Original development split/text identity differs")
    features = dev.load_features(prior / "features.npz", data, fold_id,
                                 config["prior_contract_sha256"], base["feature"]["dimensions"])
    with np.load(prior / "oof.npz", allow_pickle=False) as artifact:
        if set(artifact.files) != {"probabilities", "row_index", "ids", "groups", "truth", "labels", "fold_id", "contract_sha256"}:
            raise ValueError("Baseline OOF schema differs")
        for key in ("row_index", "ids", "groups", "truth", "labels"):
            if not np.array_equal(artifact[key], data[key]):
                raise ValueError("Baseline OOF row/label identity differs")
        if (not np.array_equal(artifact["fold_id"], fold_id)
                or str(artifact["contract_sha256"]) != config["prior_contract_sha256"]):
            raise ValueError("Baseline OOF fold/contract identity differs")
        baseline = dev.probabilities(artifact["probabilities"], len(features), len(data["labels"]))
    return data, features, baseline, folds, fold_id, prior_contract


def contract(config, base, data, fold_id):
    return {"config": config, "base_config": base,
            "sources": {str(path.relative_to(MODULE)): digest(path) for path in SOURCES},
            "row_identity_sha256": dev.object_hash({key: data[key].tolist() for key in
                                                   ("row_index", "ids", "groups", "truth", "labels")}),
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
        data, _, _, _, fold_id, _ = load_inputs(config, base)
        value = contract(config, base, data, fold_id)
    except (OSError, ValueError) as error:
        return {"state": "blocked", "blockers": [f"{type(error).__name__}: {error}"]}, None
    return {"state": "ready", "rows": len(fold_id), "contract_sha256": dev.object_hash(value)}, value


def paired_comparison(candidate, baseline, data, fold_id, evaluation):
    truth = data["truth"]
    difference = (candidate.argmax(1) == truth).astype(np.int64) - (baseline.argmax(1) == truth)
    groups, gi = np.unique(data["groups"], return_inverse=True)
    counts = np.zeros((len(groups), len(data["labels"])), dtype=np.int64)
    hits = np.zeros_like(counts)
    np.add.at(counts, (gi, truth), 1)
    np.add.at(hits, (gi, truth), difference)
    draws = np.random.default_rng(evaluation["bootstrap_seed"]).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=evaluation["bootstrap_resamples"])
    denominators = draws @ counts
    if np.any(denominators == 0):
        raise ValueError("Paired bootstrap draw missing a class")
    fold_gains = []
    for number in range(evaluation["folds"]):
        selected = fold_id == number
        fold_gains.append(float(np.mean([difference[selected & (truth == label)].mean()
                                        for label in range(len(data["labels"]))])))
    return {"pooled_ba_gain": float(np.mean([difference[truth == label].mean() for label in range(len(data["labels"]))])),
            "fold_ba_gains": fold_gains,
            "improved_folds": sum(value > 1e-12 for value in fold_gains),
            "prompt_paired_gain_95ci": np.quantile(((draws @ hits) / denominators).mean(1), [.025,.975]).tolist()}


def metric_gate(metrics, comparison, config):
    gate = config["candidate_gate"]
    folds = metrics["fold_balanced_accuracy"]
    all_values = [metrics["balanced_accuracy"], *folds, comparison["pooled_ba_gain"],
                  *comparison["prompt_paired_gain_95ci"], *comparison["fold_ba_gains"]]
    checks = {
        "finite_complete": bool(np.isfinite(all_values).all() and len(folds) == 5
                                and len(comparison["fold_ba_gains"]) == 5
                                and all(0 <= value <= 1 for value in [metrics["balanced_accuracy"], *folds])),
        "pooled": metrics["balanced_accuracy"] >= gate["min_pooled_balanced_accuracy"],
        "each_fold": min(folds, default=-1) >= gate["min_each_fold_balanced_accuracy"],
        "paired_gain": comparison["pooled_ba_gain"] >= gate["min_paired_gain_over_round2"],
        "paired_interval": comparison["prompt_paired_gain_95ci"][0] > gate["paired_prompt_95ci_lower_exceeds"],
        "improved_folds": comparison["improved_folds"] >= gate["min_improved_folds_over_round2"]}
    return {"metric_gate_passed": all(checks.values()), "checks": checks,
            "requires_completed_supervisor": True, "deployment_validation_complete": False}


def execute(config, base):
    claim = json.loads(CLAIM.read_bytes())
    dev.require_live_supervisor(claim)
    data, features, baseline, folds, fold_id, _ = load_inputs(config, base)
    if dev.object_hash(contract(config, base, data, fold_id)) != claim["contract_sha256"]:
        raise ValueError("Contract changed after readiness")
    oof = np.full((len(features), len(data["labels"])), np.nan)
    records = []
    with threadpool_limits(limits=config["budget"]["threads"]):
        for number, (train, valid) in enumerate(folds):
            dev.consume(claim, "attempted_fits", config["budget"]["max_classifier_fit_attempts"], CLAIM)
            started = time.perf_counter()
            model, info = fit_head(features[train], data["truth"][train], data["groups"][train], config["method"])
            claim["completed_fits"] += 1
            write_json(CLAIM, claim, exclusive=False)
            probabilities = dev.probabilities(predict_probabilities(features[valid], model), len(valid), 18)
            reverse = predict_probabilities(features[valid[::-1]], model)[::-1]
            single = predict_probabilities(features[valid[:1]], model)
            error = max(float(np.max(np.abs(reverse - probabilities))),
                        float(np.max(np.abs(single - probabilities[:1]))))
            if error > 1e-10:
                raise ValueError("Batch/order invariant inference check failed")
            path = OUTPUT / f"head_fold{number}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, **model)
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(model) or any(not np.array_equal(saved[key], value) for key, value in model.items()):
                    raise ValueError("Exported numerical head changed")
            oof[valid] = probabilities
            records.append({"fold": number, "fit_and_predict_seconds": time.perf_counter() - started,
                            "inference_invariance_max_abs_error": error, "model_sha256": digest(path), **info})
    oof = dev.probabilities(oof, len(features), 18)
    dev.save_array(OUTPUT / "oof.npz", "probabilities", oof, data, fold_id, claim["contract_sha256"])
    metrics = dev.metrics(oof, data, fold_id, config["evaluation"])
    comparison = paired_comparison(oof, baseline, data, fold_id, config["evaluation"])
    report = {"scope": config["scope"], "contract_sha256": claim["contract_sha256"],
              "metrics": metrics, "paired_round2_comparison": comparison,
              "candidate_gate": metric_gate(metrics, comparison, config), "folds": records,
              "oof_sha256": digest(OUTPUT / "oof.npz"), "attempted_fits": claim["attempted_fits"],
              "completed_fits": claim["completed_fits"], "new_encoder_forward_calls": 0,
              "final_holdout_evaluations": 0, "full_data_final_fits": 0,
              "conditional_final_head_fits": 0, "new_kaggle_submissions": 0,
              "limitations": "Post-diagnostic adaptive development; paired interval is descriptive, no new independent holdout or Kaggle score."}
    preserved(config)
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
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--execute"],
                                       env=env, stdout=stdout, stderr=stderr)
            phase = "worker"
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter() - started > config["budget"]["development_timeout_seconds"]:
                    raise TimeoutError("Round3 wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss
                    rss += sum(child.memory_info().rss for child in monitored.children(recursive=True))
                    peak_rss = max(peak_rss, rss)
                    if rss > config["budget"]["max_host_rss_gib"] * 2**30:
                        raise MemoryError("Round3 process-tree RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed"
                or claim["attempted_fits"] != 5 or claim["completed_fits"] != 5
                or claim["attempted_forward_calls"] != 0 or claim["completed_forward_calls"] != 0
                or digest(OUTPUT / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Round3 worker failed or returned incomplete artifacts")
        preserved(config)
        if any(digest(MODULE / relative) != expected for relative, expected in value["sources"].items()):
            raise ValueError("Execution sources changed during this round")
        summary = json.loads((OUTPUT / "summary.json").read_bytes())
        gate = metric_gate(summary["metrics"], summary["paired_round2_comparison"], config)
        receipt = {"state": "completed", "exit_code": 0, "wall_seconds": time.perf_counter() - started,
                   "sampled_peak_process_tree_rss_bytes": peak_rss, "summary_sha256": claim["summary_sha256"],
                   "candidate_development_gate_passed": gate["metric_gate_passed"],
                   "candidate_deployment_validation_complete": False}
        write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(OUTPUT / "supervisor.json"))
        write_json(CLAIM, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = dev.terminate_worker(process)
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
