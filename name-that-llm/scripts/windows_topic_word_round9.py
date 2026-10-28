"""Single supervised R9 word-centering experiment. Default is readiness only."""
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
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import windows_external_round7 as r7
import windows_modernbert_dev as dev
import windows_stylegram_round5 as guards
from probe_modernbert import digest, write_json
from supervisor_round7 import require_live_supervisor
from topic_word_round9 import infer_groups, transform_views, group_diagnostics, metric_gate
from weighted_style_round7 import classifier, sample_weights
from windows_prompt_gaussian_round3 import paired_comparison

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_topic_word_round9_20261028.json"
CONFIG_SHA256 = "001fadd21809bb7ac4fbba133b4714210db310066feecd63689b84a3ee53d700"
CLAIM = MODULE / "outputs/continuation_claims/windows-topic-word-dev-20261028-r9.json"
OUTPUT = MODULE / "outputs/windows_topic_word_20261028_r9"
SOURCES = tuple(dict.fromkeys((*r7.SOURCES, Path(__file__), CONFIG,
    Path(__file__).with_name("topic_word_round9.py"), Path(__file__).with_name("supervisor_round7.py"),
    MODULE / "docs/WINDOWS_ROUND9_PLAN_20261028.md")))


def configuration():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("R9 config identity changed")
    config = json.loads(CONFIG.read_bytes())
    if config["experiment_id"] != CLAIM.stem or sys.platform != "win32":
        raise ValueError("Unexpected experiment or platform")
    for name, version in config["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Runtime identity changed: {name}")
    return config


def verify_files(config):
    for name, expected in {**config["input_sha256"], **config["preserved_claims"]}.items():
        path = (MODULE / name).resolve()
        if not path.is_relative_to(MODULE) or digest(path) != expected:
            raise ValueError(f"Frozen file changed: {name}")


def load_inputs(config):
    verify_files(config)
    parent, base = r7.configuration()
    data, pool, _, folds, fold_id = r7.load_inputs(parent, base)
    folder = MODULE / config["baseline_output"]
    prior = json.loads((folder / "contract.json").read_bytes())
    old_claim = json.loads((MODULE / "outputs/continuation_claims/windows-external-weighting-dev-20261026-r7-startup-1.json").read_bytes())
    if (dev.object_hash(prior) != config["baseline_contract_sha256"] or old_claim["state"] != "completed"
            or old_claim["contract_sha256"] != config["baseline_contract_sha256"]
            or digest(folder / "summary.json") != old_claim["summary_sha256"]
            or digest(folder / "supervisor.json") != old_claim["supervisor_sha256"]):
        raise ValueError("R7 baseline evidence changed")
    with np.load(folder / "candidate_oof.npz", allow_pickle=False) as saved:
        for key in ("row_index", "ids", "groups", "truth", "labels"):
            np.testing.assert_array_equal(saved[key], data[key])
        np.testing.assert_array_equal(saved["fold_id"], fold_id)
        if str(saved["contract_sha256"]) != config["baseline_contract_sha256"]:
            raise ValueError("R7 OOF contract differs")
        baseline = dev.probabilities(saved["probabilities"], len(fold_id), 18).copy()
    return data, pool, baseline, folds, fold_id, prior


def contract(config, data, fold_id):
    return {"config": config, "sources": {p.relative_to(MODULE).as_posix(): digest(p) for p in SOURCES},
            "text_sha256": dev.object_hash(data["texts"].tolist()),
            "row_identity_sha256": dev.object_hash({k: data[k].tolist() for k in ("row_index", "ids", "groups", "truth", "labels")}),
            "fold_sha256": dev.object_hash(fold_id.tolist()), "python": sys.version, "platform": sys.platform}


def readiness(config):
    if CLAIM.exists() or OUTPUT.exists():
        return {"state": "blocked", "reason": "R9 claim/output exists; retry and resume forbidden"}, None
    try:
        data, _, _, _, fold_id, _ = load_inputs(config)
        value = contract(config, data, fold_id)
        return {"state": "ready", "rows": len(fold_id), "contract_sha256": dev.object_hash(value)}, value
    except (OSError, ValueError, AssertionError) as error:
        return {"state": "blocked", "reason": f"{type(error).__name__}: {error}"}, None


def grouping(claim, config, texts, ids):
    dev.consume(claim, "attempted_groupings", config["budget"]["max_grouping_attempts"], CLAIM)
    groups = infer_groups(texts, ids, config["method"]["grouping"])
    claim["completed_groupings"] += 1
    write_json(CLAIM, claim, exclusive=False)
    return groups


def validation_scores(views, model, texts, groups, alpha):
    matrix, _ = transform_views(views, texts, groups, alpha)
    scores = np.asarray(model.decision_function(matrix), dtype=np.float64)
    if scores.shape != (len(texts), 18) or not np.isfinite(scores).all():
        raise ValueError("Invalid candidate scores")
    return scores


def execute(config):
    claim = json.loads(CLAIM.read_bytes())
    require_live_supervisor(claim)
    if claim["attempted_groupings"] or claim["completed_groupings"]:
        raise ValueError("Grouping budget already consumed")
    guards.start_worker_guard(claim, config["budget"])
    data, pool, baseline, folds, fold_id, prior = load_inputs(config)
    if dev.object_hash(contract(config, data, fold_id)) != claim["contract_sha256"]:
        raise ValueError("Execution contract changed")
    outputs = {name: np.full_like(baseline, np.nan) for name in ("candidate", "half", "singleton")}
    records = []
    with threadpool_limits(limits=config["budget"]["threads"]):
        for number, (train, valid) in enumerate(folds):
            started = time.perf_counter()
            folder = OUTPUT / f"fold{number}"
            folder.mkdir(exist_ok=False)
            selected, _ = r7.fold_pool(pool, data["texts"][train], data["truth"][train])
            expected = prior["fold_external_selections"][number]
            if len(selected) != expected["rows"] or dev.object_hash(pool["ids"][selected].tolist()) != expected["ids_sha256"]:
                raise ValueError("R7 fold external selection changed")
            views = joblib.load(MODULE / config["baseline_output"] / f"fold{number}/views.joblib")
            if [(name, weight) for name, _, weight in views] != [(s["name"], s["weight"]) for s in config["method"]["views"]]:
                raise ValueError("R7 frozen view order changed")
            train_texts = data["texts"][train].tolist() + pool["texts"][selected].tolist()
            train_groups = np.asarray(["official:" + str(g) for g in data["groups"][train]] + ["external:" + str(g) for g in pool["groups"][selected]])
            y = np.concatenate([data["truth"][train], pool["truth"][selected]])
            print(f"fold={number} transform_started rows={len(y)}", flush=True)
            matrix, nnz = transform_views(views, train_texts, train_groups, config["method"]["alpha"])
            dev.consume(claim, "attempted_fits", config["budget"]["max_classifier_fit_attempts"], CLAIM)
            print(f"fold={number} classifier_fit_started nnz={matrix.nnz}", flush=True)
            model = classifier(config["method"]["classifier"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(matrix, y, sample_weight=sample_weights(len(train), len(selected), config["method"]["official_weight"]))
            claim["completed_fits"] += 1
            write_json(CLAIM, claim, exclusive=False)
            if not np.array_equal(model.classes_, np.arange(18)):
                raise ValueError("Class order changed")
            del matrix, train_texts, y
            texts, ids = data["texts"][valid], data["ids"][valid]
            groups = grouping(claim, config, texts, ids)
            scores = validation_scores(views, model, texts.tolist(), groups, config["method"]["alpha"])
            reverse_groups = grouping(claim, config, texts[::-1], ids[::-1])
            np.testing.assert_array_equal(groups, reverse_groups[::-1])
            reverse = validation_scores(views, model, texts[::-1].tolist(), reverse_groups, config["method"]["alpha"])[::-1]
            error = float(np.abs(scores - reverse).max())
            if error > 1e-7:
                raise ValueError("Permutation invariance failed")
            half_scores = np.empty_like(scores)
            half_groups = np.empty(len(valid), dtype=np.int64)
            for part_number, index in enumerate(np.array_split(np.argsort(ids, kind="stable"), 2)):
                part_groups = grouping(claim, config, texts[index], ids[index])
                half_scores[index] = validation_scores(views, model, texts[index].tolist(), part_groups, config["method"]["alpha"])
                half_groups[index] = part_groups + part_number * len(valid)
            single_scores = validation_scores(views, model, texts.tolist(), np.arange(len(valid)), config["method"]["alpha"])
            for name, item in (("candidate", scores), ("half", half_scores), ("singleton", single_scores)):
                outputs[name][valid] = softmax(item, axis=1)
            with (folder / "scores.npz").open("xb") as handle:
                np.savez_compressed(handle, candidate=scores, half=half_scores, singleton=single_scores,
                                    groups=groups, half_groups=half_groups, valid_positions=valid, labels=data["labels"], contract_sha256=claim["contract_sha256"])
            with (folder / "model.joblib").open("xb") as handle:
                joblib.dump(model, handle, compress=3)
            # True prompt identifiers enter diagnostics only after prediction arrays have been saved.
            record = {"fold": number, "official_train_rows": len(train), "external_rows": len(selected), "word_nnz": nnz,
                      "training_group_diagnostics": group_diagnostics(train_groups),
                      "validation_group_diagnostics": group_diagnostics(groups, data["groups"][valid]),
                      "half_group_diagnostics": group_diagnostics(half_groups, data["groups"][valid]),
                      "permutation_max_abs_error": error, "iterations": int(model.n_iter_), "seconds": time.perf_counter()-started,
                      "model_sha256": digest(folder / "model.joblib"), "scores_sha256": digest(folder / "scores.npz")}
            write_json(folder / "receipt.json", record)
            records.append(record)
            print(f"fold={number} completed seconds={record['seconds']:.2f}", flush=True)
            del views, model, train_groups
    for name, array in outputs.items():
        dev.probabilities(array, len(fold_id), 18)
        dev.save_array(OUTPUT / f"{name}_oof.npz", "probabilities", array, data, fold_id, claim["contract_sha256"])
    metrics = {name: dev.metrics(array, data, fold_id, config["evaluation"]) for name, array in outputs.items()}
    comparisons = {name: paired_comparison(array, baseline, data, fold_id, config["evaluation"]) for name, array in outputs.items()}
    gate = metric_gate(metrics["candidate"], comparisons["candidate"], comparisons["half"], comparisons["singleton"], config["candidate_gate"])
    summary = {"scope": config["scope"], "contract_sha256": claim["contract_sha256"], "metrics": metrics,
               "baseline_metrics": dev.metrics(baseline, data, fold_id, config["evaluation"]), "comparisons": comparisons,
               "candidate_gate": gate, "folds": records,
               "oof_sha256": {name: digest(OUTPUT / f"{name}_oof.npz") for name in outputs},
               **{k: claim[k] for k in ("attempted_fits", "completed_fits", "attempted_groupings", "completed_groupings")},
               "new_supervised_vectorizer_fits": 0, "new_encoder_forward_calls": 0, "old_holdout_evaluations": 0,
               "full_data_final_fits": 0, "new_kaggle_submissions": 0,
               "limitations": "Repeated development on the original 120 prompts. Frozen external pool inherits R7 label-free copy exclusion using all development texts and upstream curation. Text-only transductive groups use the other unlabeled rows in each validation batch. Bootstrap conditions on these fixed predictions and does not regroup each draw. Half/singleton are fixed stress tests, never alternate candidates. Not the full teammate ensemble or an independent holdout."}
    verify_files(config)
    write_json(OUTPUT / "summary.json", summary)
    claim.update(state="worker_completed", summary_sha256=digest(OUTPUT / "summary.json"))
    write_json(CLAIM, claim, exclusive=False)


def supervise(config, value):
    claim = {"state": "started", "experiment_id": config["experiment_id"], "supervisor_pid": os.getpid(),
             "supervisor_create_time": psutil.Process().create_time(), "started_unix_seconds": time.time(),
             "contract_sha256": dev.object_hash(value), "old_holdout_evaluations": 0, "full_data_final_fits": 0}
    claim.update({k: 0 for k in ("attempted_fits", "completed_fits", "attempted_forward_calls", "completed_forward_calls",
                                "attempted_vectorizer_fits", "completed_vectorizer_fits", "attempted_groupings", "completed_groupings")})
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
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--execute"], env=env, stdout=stdout, stderr=stderr)
            phase, monitored = "worker", psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter()-started > config["budget"]["development_timeout_seconds"]:
                    raise TimeoutError("R9 wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss + sum(p.memory_info().rss for p in monitored.children(recursive=True))
                    peak = max(peak, rss)
                    if rss > config["budget"]["max_host_rss_gib"] * 2**30:
                        raise MemoryError("R9 process-tree RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed"
                or any(claim[k] != v for k, v in {"attempted_fits": 5, "completed_fits": 5, "attempted_groupings": 20,
                    "completed_groupings": 20, "attempted_vectorizer_fits": 0, "completed_vectorizer_fits": 0,
                    "attempted_forward_calls": 0, "completed_forward_calls": 0}.items())
                or digest(OUTPUT / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("R9 worker failed or returned incomplete work")
        verify_files(config)
        if any(digest(MODULE / p) != sha for p, sha in value["sources"].items()):
            raise ValueError("Execution source changed")
        summary = json.loads((OUTPUT / "summary.json").read_bytes())
        receipt = {"state": "completed", "exit_code": 0, "wall_seconds": time.perf_counter()-started,
                   "sampled_peak_process_tree_rss_bytes": peak, "summary_sha256": claim["summary_sha256"],
                   "candidate_development_gate_passed": summary["candidate_gate"]["metric_gate_passed"]}
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
    config = configuration()
    if args.execute:
        execute(config)
        return 0
    state, value = readiness(config)
    if not args.run or state["state"] != "ready":
        print(json.dumps(state))
        return 0 if state["state"] == "ready" else 2
    print(json.dumps(supervise(config, value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
