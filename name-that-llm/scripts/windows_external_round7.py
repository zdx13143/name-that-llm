"""One preregistered paired SVC experiment; readiness is the default."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import psutil
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import windows_modernbert_dev as dev
import windows_stylegram_round5 as guards
from probe_modernbert import digest, write_json
from windows_prompt_gaussian_round3 import paired_comparison
from external_pool_round7 import load_pool, fold_pool, MARKS
from weighted_style_round7 import vectorizer, sample_weights, classifier, decision

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_external_round7_20261026.json"
CONFIG_SHA256 = "b7b32e3f1a47aae941844b8797a868ecc322a7943a77f31d0784c72ccffaf98c"
BASE = MODULE / "config/windows_modernbert_20261023.json"
OUTPUT = MODULE / "outputs/windows_external_weighting_20261026_r7"
CLAIM = MODULE / "outputs/continuation_claims/windows-external-weighting-dev-20261026-r7.json"
ENTRYPOINT = Path(__file__).resolve()
SUPERVISOR_CHECK = dev.require_live_supervisor
SOURCES = tuple(dict.fromkeys((*guards.SOURCES, Path(__file__), CONFIG,
    Path(__file__).with_name("external_pool_round7.py"), Path(__file__).with_name("weighted_style_round7.py"),
    MODULE / "config/round7_external_sources_20261026.json")))


def configuration():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("Round7 config identity changed")
    config = json.loads(CONFIG.read_bytes())
    if config["experiment_id"] != CLAIM.stem or tuple(config["method"]["punctuation_marks"]) != MARKS:
        raise ValueError("Unexpected Round7 recipe")
    if config["method"]["external_weight"] != 1.0:
        raise ValueError("External sample weight must remain one")
    return config, json.loads(BASE.read_bytes())


def load_inputs(config, base):
    dev.verify_preserved_claims(config)
    data = dev.load_development(MODULE / "data/raw/train.csv", base["input"], MODULE / "config/labels.json")
    folds, fold_id = dev.grouped_folds(data, config["evaluation"]["folds"])
    reference = json.loads((MODULE / config["fold_reference_output"] / "contract.json").read_bytes())
    if (dev.object_hash(reference) != config["fold_reference_contract_sha256"]
            or dev.object_hash(fold_id.tolist()) != reference["fold_sha256"]
            or dev.object_hash(data["texts"].tolist()) != reference["text_sha256"]):
        raise ValueError("Original development identity changed")
    folder = MODULE / config["round6_output"]
    prior = json.loads((MODULE / "outputs/continuation_claims/windows-calibrated-fusion-dev-20261025-r6.json").read_bytes())
    if (prior["state"] != "completed" or prior["contract_sha256"] != config["round6_contract_sha256"]
            or dev.object_hash(json.loads((folder / "contract.json").read_bytes())) != config["round6_contract_sha256"]
            or digest(folder / "oof.npz") != config["round6_oof_sha256"]
            or digest(folder / "summary.json") != prior["summary_sha256"]
            or digest(folder / "supervisor.json") != prior["supervisor_sha256"]):
        raise ValueError("Frozen Round6 evidence changed")
    with np.load(folder / "oof.npz", allow_pickle=False) as saved:
        for key in ("row_index", "ids", "groups", "truth", "labels"):
            np.testing.assert_array_equal(saved[key], data[key])
        np.testing.assert_array_equal(saved["fold_id"], fold_id)
        if str(saved["contract_sha256"]) != config["round6_contract_sha256"]:
            raise ValueError("Round6 OOF contract differs")
        round6 = dev.probabilities(saved["probabilities"], len(fold_id), 18)
    pool = load_pool(MODULE / config["pool_path"], config["pool_sha256"], data["labels"].tolist())
    receipt = json.loads((MODULE / config["pool_receipt_path"]).read_bytes())
    if (len(pool["truth"]) != config["pool_rows"] or len(set(pool["ids"])) != len(pool["ids"])
            or receipt["prepared_csv_sha256"] != config["pool_sha256"]
            or receipt["preparer_sha256"] != digest(Path(__file__).with_name("external_pool_round7.py"))
            or receipt["raw_manifest_sha256"] != config["pool_raw_manifest_sha256"]):
        raise ValueError("External curation identity differs")
    return data, pool, round6, folds, fold_id


def contract(config, base, data, pool, folds, fold_id):
    selections = []
    for train, _ in folds:
        selected, present = fold_pool(pool, data["texts"][train], data["truth"][train])
        selections.append({"rows": len(selected), "ids_sha256": dev.object_hash(pool["ids"][selected].tolist()),
                           "punctuation_present": present.tolist()})
    return {"config": config, "base_config": base,
            "sources": {str(p.relative_to(MODULE)): digest(p) for p in SOURCES},
            "row_identity_sha256": dev.object_hash({k: data[k].tolist() for k in ("row_index", "ids", "groups", "truth", "labels")}),
            "text_sha256": dev.object_hash(data["texts"].tolist()), "fold_sha256": dev.object_hash(fold_id.tolist()),
            "pool_receipt_sha256": digest(MODULE / config["pool_receipt_path"]), "fold_external_selections": selections,
            "python": sys.version, "platform": sys.platform,
            "package_versions": {name: importlib.metadata.version(name) for name in config["runtime"]}}


def readiness(config, base):
    blockers = []
    if CLAIM.exists() or OUTPUT.exists():
        blockers.append("Round7 claim/output already exists; retry/resume forbidden")
    if sys.platform != "win32":
        blockers.append("Native Windows required")
    for name, expected in config["runtime"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            blockers.append(f"Runtime package missing: {name}")
            continue
        if actual != expected:
            blockers.append(f"Runtime differs: {name}")
    if blockers:
        return {"state": "blocked", "blockers": blockers}, None
    try:
        data, pool, _, folds, fold_id = load_inputs(config, base)
        value = contract(config, base, data, pool, folds, fold_id)
    except (OSError, ValueError, AssertionError) as error:
        return {"state": "blocked", "blockers": [f"{type(error).__name__}: {error}"]}, None
    return {"state": "ready", "official_rows": len(fold_id), "pool_rows": len(pool["truth"]),
            "fold_external_rows": [v["rows"] for v in value["fold_external_selections"]],
            "contract_sha256": dev.object_hash(value)}, value


def metric_gate(metrics, control, round6, config):
    threshold = config["candidate_gate"]
    values = [metrics["balanced_accuracy"], *metrics["fold_balanced_accuracy"], control["pooled_ba_gain"],
              *control["prompt_paired_gain_95ci"], *control["fold_ba_gains"], round6["pooled_ba_gain"]]
    checks = {"finite_complete": bool(np.isfinite(values).all() and len(metrics["fold_balanced_accuracy"]) == 5),
              "pooled": metrics["balanced_accuracy"] >= threshold["min_pooled_balanced_accuracy"],
              "each_fold": min(metrics["fold_balanced_accuracy"], default=-1) >= threshold["min_each_fold_balanced_accuracy"],
              "gain_over_pool_control": control["pooled_ba_gain"] >= threshold["min_gain_over_pool_control"],
              "paired_interval": control["prompt_paired_gain_95ci"][0] > threshold["paired_prompt_95ci_lower_exceeds"],
              "improved_folds": control["improved_folds"] >= threshold["min_improved_folds"],
              "gain_over_round6": round6["pooled_ba_gain"] >= threshold["min_gain_over_round6"]}
    return {"metric_gate_passed": all(checks.values()), "checks": checks,
            "requires_completed_supervisor": True, "deployment_validation_complete": False}


def consume(claim, kind, config):
    if claim["attempted_fits"] + claim["attempted_vectorizer_fits"] >= config["budget"]["max_total_fit_attempts"]:
        raise RuntimeError("Total fit budget exhausted")
    key = "max_classifier_fit_attempts" if kind == "attempted_fits" else "max_vectorizer_fit_attempts"
    dev.consume(claim, kind, config["budget"][key], CLAIM)


def execute(config, base):
    claim = json.loads(CLAIM.read_bytes())
    SUPERVISOR_CHECK(claim)
    guards.start_worker_guard(claim, config["budget"])
    data, pool, round6, folds, fold_id = load_inputs(config, base)
    value = contract(config, base, data, pool, folds, fold_id)
    if dev.object_hash(value) != claim["contract_sha256"]:
        raise ValueError("Execution contract changed")
    outputs = {name: np.full((len(fold_id), 18), np.nan) for name in ("control", "candidate")}
    records = []
    with threadpool_limits(limits=config["budget"]["threads"]):
        for number, (train, valid) in enumerate(folds):
            started = time.perf_counter()
            folder = OUTPUT / f"fold{number}"
            folder.mkdir(exist_ok=False)
            selected, present = fold_pool(pool, data["texts"][train], data["truth"][train])
            fit_texts = data["texts"][train].tolist() + pool["texts"][selected].tolist()
            fit_y = np.concatenate([data["truth"][train], pool["truth"][selected]])
            matrices, validation, views = [], [], []
            for setting in config["method"]["views"]:
                consume(claim, "attempted_vectorizer_fits", config)
                print(f"fold={number} view={setting['name']} fit_started rows={len(fit_y)}", flush=True)
                vec = vectorizer(setting)
                matrix = vec.fit_transform(fit_texts) * setting["weight"]
                claim["completed_vectorizer_fits"] += 1
                write_json(CLAIM, claim, exclusive=False)
                matrices.append(matrix)
                validation.append(vec.transform(data["texts"][valid].tolist()) * setting["weight"])
                views.append((setting["name"], vec, setting["weight"]))
            matrix, valid_matrix = hstack(matrices, format="csr"), hstack(validation, format="csr")
            del matrices, validation, fit_texts
            with (folder / "views.joblib").open("xb") as handle:
                joblib.dump(views, handle, compress=3)
            record = {"fold": number, "official_train_rows": len(train), "valid_rows": len(valid),
                      "external_rows": len(selected), "external_ids_sha256": dev.object_hash(pool["ids"][selected].tolist()),
                      "punctuation_present": present.tolist(), "views_sha256": digest(folder / "views.joblib"),
                      "features": matrix.shape[1], "vocabulary_sizes": {name: len(vec.vocabulary_) for name, vec, _ in views}, "models": {}}
            for name in ("control", "candidate"):
                consume(claim, "attempted_fits", config)
                weight = config["method"][name + "_official_weight"]
                print(f"fold={number} {name} classifier_fit_started official_weight={weight}", flush=True)
                model = classifier(config["method"]["classifier"])
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(matrix, fit_y, sample_weight=sample_weights(len(train), len(selected), weight))
                claim["completed_fits"] += 1
                write_json(CLAIM, claim, exclusive=False)
                if not np.array_equal(model.classes_, np.arange(18)):
                    raise ValueError("Classifier class order differs")
                scores = np.asarray(model.decision_function(valid_matrix), dtype=np.float64)
                if scores.shape != (len(valid), 18) or not np.isfinite(scores).all():
                    raise ValueError("Invalid validation scores")
                reverse = np.asarray(model.decision_function(valid_matrix[::-1]))[::-1]
                singleton = decision(views, model, data["texts"][valid[:1]].tolist())
                error = max(float(np.abs(scores-reverse).max()), float(np.abs(scores[:1]-singleton).max()))
                if error > 1e-7:
                    raise ValueError("Inference order/batch invariance failed")
                outputs[name][valid] = softmax(scores, axis=1)
                with (folder / f"{name}.joblib").open("xb") as handle:
                    joblib.dump(model, handle, compress=3)
                with (folder / f"{name}_scores.npz").open("xb") as handle:
                    np.savez_compressed(handle, scores=scores, valid_positions=valid, labels=data["labels"],
                                        contract_sha256=claim["contract_sha256"])
                record["models"][name] = {"weight": weight, "iterations": int(model.n_iter_),
                    "model_sha256": digest(folder / f"{name}.joblib"), "scores_sha256": digest(folder / f"{name}_scores.npz"),
                    "inference_invariance_max_abs_error": error}
                del model
            record["seconds"] = time.perf_counter()-started
            records.append(record)
            write_json(folder / "receipt.json", record)
            print(f"fold={number} completed seconds={record['seconds']:.2f}", flush=True)
            del matrix, valid_matrix, views, fit_y
    for name, array in outputs.items():
        dev.probabilities(array, len(fold_id), 18)
        dev.save_array(OUTPUT / f"{name}_oof.npz", "probabilities", array, data, fold_id, claim["contract_sha256"])
    metrics = dev.metrics(outputs["candidate"], data, fold_id, config["evaluation"])
    control = paired_comparison(outputs["candidate"], outputs["control"], data, fold_id, config["evaluation"])
    prior = paired_comparison(outputs["candidate"], round6, data, fold_id, config["evaluation"])
    summary = {"scope": config["scope"], "contract_sha256": claim["contract_sha256"], "metrics": metrics,
        "pool_control_metrics": dev.metrics(outputs["control"], data, fold_id, config["evaluation"]),
        "paired_pool_control_comparison": control, "paired_round6_comparison": prior,
        "candidate_gate": metric_gate(metrics, control, prior, config), "folds": records,
        "candidate_oof_sha256": digest(OUTPUT / "candidate_oof.npz"), "control_oof_sha256": digest(OUTPUT / "control_oof.npz"),
        **{k: claim[k] for k in ("attempted_fits", "completed_fits", "attempted_vectorizer_fits", "completed_vectorizer_fits")},
        "new_encoder_forward_calls": 0, "conditional_final_fits": 0, "old_holdout_evaluations": 0, "new_kaggle_submissions": 0,
        "limitations": "Adaptive development on the original120 prompts with externally curated public data. All development texts, including each validation fold, were used without labels solely to exclude overlapping external groups before the frozen pool was built. CV is conditional on this fixed decontamination and upstream source curation, not a blind independent holdout or an evaluation of the complete 0.601 ensemble."}
    dev.verify_preserved_claims(config)
    write_json(OUTPUT / "summary.json", summary)
    claim.update(state="worker_completed", summary_sha256=digest(OUTPUT / "summary.json"))
    write_json(CLAIM, claim, exclusive=False)


def supervise(config, value):
    claim = {"state": "started", "experiment_id": config["experiment_id"], "supervisor_pid": os.getpid(),
             "supervisor_create_time": psutil.Process().create_time(), "started_unix_seconds": time.time(),
             "contract_sha256": dev.object_hash(value), "attempted_fits": 0, "completed_fits": 0,
             "attempted_vectorizer_fits": 0, "completed_vectorizer_fits": 0, "attempted_forward_calls": 0,
             "completed_forward_calls": 0, "old_holdout_evaluations": 0, "full_data_final_fits": 0}
    write_json(CLAIM, claim)
    process, phase, peak = None, "pre_spawn", 0
    started = time.perf_counter()
    try:
        OUTPUT.mkdir(exist_ok=False, parents=True)
        write_json(OUTPUT / "contract.json", value)
        for source in SOURCES:
            target = OUTPUT / "source_snapshot" / source.relative_to(MODULE)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
        env = {**os.environ, **{k: str(config["budget"]["threads"]) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-u", str(ENTRYPOINT), "--execute"], env=env, stdout=stdout, stderr=stderr)
            phase, monitored = "worker", psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter()-started > config["budget"]["development_timeout_seconds"]:
                    raise TimeoutError("Round7 wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss + sum(p.memory_info().rss for p in monitored.children(recursive=True))
                    peak = max(peak, rss)
                    if rss > config["budget"]["max_host_rss_gib"] * 2**30:
                        raise MemoryError("Round7 process-tree RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed"
                or claim["attempted_fits"] != 10 or claim["completed_fits"] != 10
                or claim["attempted_vectorizer_fits"] != 20 or claim["completed_vectorizer_fits"] != 20
                or claim["attempted_forward_calls"] != 0 or claim["completed_forward_calls"] != 0
                or digest(OUTPUT / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Round7 worker failed or returned incomplete work")
        dev.verify_preserved_claims(config)
        if any(digest(MODULE / p) != sha for p, sha in value["sources"].items()) or digest(MODULE / config["pool_path"]) != config["pool_sha256"]:
            raise ValueError("Execution source/data identity changed")
        summary = json.loads((OUTPUT / "summary.json").read_bytes())
        gate = metric_gate(summary["metrics"], summary["paired_pool_control_comparison"], summary["paired_round6_comparison"], config)
        receipt = {"state": "completed", "exit_code": 0, "wall_seconds": time.perf_counter()-started,
                   "sampled_peak_process_tree_rss_bytes": peak, "summary_sha256": claim["summary_sha256"],
                   "candidate_development_gate_passed": gate["metric_gate_passed"], "deployment_validation_complete": False}
        write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(OUTPUT / "supervisor.json"))
        write_json(CLAIM, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = guards.terminate_tree(process)
        finally:
            claim = json.loads(CLAIM.read_bytes())
            claim.update(state="failed", exception_type=type(error).__name__, exception_message=str(error), phase=phase,
                         termination=termination, wall_seconds=time.perf_counter()-started, sampled_peak_process_tree_rss_bytes=peak)
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
