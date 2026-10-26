"""Conditional final weighted SVC fit on 120 development prompts only."""
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
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import windows_external_round7_startup as startup
from verify_round7_results import check
from supervisor_round7 import require_live_supervisor
from weighted_style_round7 import vectorizer, sample_weights, classifier, decision

startup.configure()
run = startup.experiment
MODULE = run.MODULE
CLAIM = MODULE / "outputs/continuation_claims/windows-external-weighting-final-20261026-r7.json"
OUTPUT = MODULE / "outputs/windows_external_final_20261026_r7"
SOURCES = tuple(dict.fromkeys((*run.SOURCES, Path(__file__).resolve(), Path(__file__).with_name("verify_round7_results.py"))))
SYNTHETIC = [
    "A small brass key lay beside the window. Mira turned it over, wondering which door it would open.",
    "The instructions were simple: measure twice, mark the wood, and keep your hands clear of the blade.",
    "Rain gathered on the leaves. By noon, the garden smelled of soil, and the path had vanished beneath shallow water.",
    "An accurate forecast needs observations as well as a model; greater numerical precision cannot repair missing evidence."
]


def admission():
    if CLAIM.exists() or OUTPUT.exists():
        raise ValueError("Final claim/output exists; no retry/resume")
    verified = check()
    if verified["state"] != "verified" or verified["development_gate_passed"] is not True:
        raise ValueError("Round7 development gate did not pass")
    config, base = run.configuration()
    for name, expected in config["runtime"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Final runtime differs: {name}")
    data, pool, _, _, _ = run.load_inputs(config, base)
    if len(data["truth"]) != 2159 or len(set(data["groups"])) != 120:
        raise ValueError("Final development scope changed")
    selected, present = run.fold_pool(pool, data["texts"], data["truth"])
    value = {"config": config, "development_verification": verified,
             "development_claim_sha256": run.digest(run.CLAIM),
             "row_identity_sha256": run.dev.object_hash({k: data[k].tolist() for k in ("row_index", "ids", "groups", "truth", "labels")}),
             "external_rows": len(selected), "external_ids_sha256": run.dev.object_hash(pool["ids"][selected].tolist()),
             "punctuation_present": present.tolist(), "synthetic_text_sha256": run.dev.object_hash(SYNTHETIC),
             "sources": {str(p.relative_to(MODULE)): run.digest(p) for p in SOURCES}}
    return config, base, value


def consume(claim, kind, budget):
    if claim["attempted_fits"] + claim["attempted_vectorizer_fits"] >= budget["max_total_fit_attempts"]:
        raise RuntimeError("Final total fit budget exhausted")
    key = "max_classifier_fit_attempts" if kind == "attempted_fits" else "max_vectorizer_fit_attempts"
    run.dev.consume(claim, kind, budget[key], CLAIM)


def execute():
    claim = json.loads(CLAIM.read_bytes())
    require_live_supervisor(claim)
    config, base = run.configuration()
    budget = config["conditional_final_budget"]
    run.guards.start_worker_guard(claim, budget)
    verified = check()
    value = json.loads((OUTPUT / "contract.json").read_bytes())
    if (verified["development_gate_passed"] is not True or verified != value["development_verification"]
            or run.digest(run.CLAIM) != value["development_claim_sha256"]
            or any(run.digest(MODULE / p) != sha for p, sha in value["sources"].items())
            or run.dev.object_hash(value) != claim["contract_sha256"]):
        raise ValueError("Final admission identity changed")
    data, pool, _, _, _ = run.load_inputs(config, base)
    selected, present = run.fold_pool(pool, data["texts"], data["truth"])
    if (len(selected) != value["external_rows"] or present.tolist() != value["punctuation_present"]
            or run.dev.object_hash(pool["ids"][selected].tolist()) != value["external_ids_sha256"]):
        raise ValueError("Final external selection changed")
    texts = data["texts"].tolist() + pool["texts"][selected].tolist()
    truth = np.concatenate([data["truth"], pool["truth"][selected]])
    matrices, views = [], []
    with threadpool_limits(limits=budget["threads"]):
        for setting in config["method"]["views"]:
            consume(claim, "attempted_vectorizer_fits", budget)
            print(f"final view={setting['name']} fit_started rows={len(texts)}", flush=True)
            vec = vectorizer(setting)
            matrices.append(vec.fit_transform(texts) * setting["weight"])
            views.append((setting["name"], vec, setting["weight"]))
            claim["completed_vectorizer_fits"] += 1
            run.write_json(CLAIM, claim, exclusive=False)
        matrix = hstack(matrices, format="csr")
        del matrices, texts
        consume(claim, "attempted_fits", budget)
        print("final weighted classifier_fit_started", flush=True)
        model = classifier(config["method"]["classifier"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(matrix, truth, sample_weight=sample_weights(2159, len(selected), config["method"]["candidate_official_weight"]))
        claim["completed_fits"] += 1
        run.write_json(CLAIM, claim, exclusive=False)
        scores = decision(views, model, SYNTHETIC)
        artifact = {"views": views, "classifier": model, "labels": data["labels"], "contract_sha256": claim["contract_sha256"]}
        with (OUTPUT / "weighted_svc.joblib").open("xb") as handle:
            joblib.dump(artifact, handle, compress=3)
        restored = joblib.load(OUTPUT / "weighted_svc.joblib")
        np.testing.assert_array_equal(scores, decision(restored["views"], restored["classifier"], SYNTHETIC))
        reverse = decision(views, model, SYNTHETIC[::-1])[::-1]
        singleton = np.concatenate([decision(views, model, [text]) for text in SYNTHETIC])
        error = max(float(np.abs(scores - reverse).max()), float(np.abs(scores - singleton).max()))
        if error > 1e-7:
            raise ValueError("Final inference order/batch invariance failed")
        with (OUTPUT / "synthetic_scores.npz").open("xb") as handle:
            np.savez_compressed(handle, scores=scores, labels=data["labels"], texts=np.asarray(SYNTHETIC),
                                contract_sha256=claim["contract_sha256"])
    summary = {"state": "model_exported", "contract_sha256": claim["contract_sha256"],
               "official_training_rows": 2159, "official_training_groups": 120, "external_training_rows": len(selected),
               "classifier_fits": 1, "vectorizer_fits": 4, "new_encoder_forward_calls": 0,
               "old_holdout_evaluations": 0, "full_2699_row_fits": 0, "new_kaggle_submissions": 0,
               "features": matrix.shape[1], "iterations": int(model.n_iter_), "official_weight": 4,
               "export_max_abs_error": 0.0, "synthetic_order_batch_max_abs_error": error,
               "model_sha256": run.digest(OUTPUT / "weighted_svc.joblib"),
               "synthetic_scores_sha256": run.digest(OUTPUT / "synthetic_scores.npz"),
               "deployment_validation_complete": False}
    run.dev.verify_preserved_claims(config)
    run.write_json(OUTPUT / "summary.json", summary)
    claim.update(state="worker_completed", summary_sha256=run.digest(OUTPUT / "summary.json"))
    run.write_json(CLAIM, claim, exclusive=False)


def supervise(config, value):
    budget = config["conditional_final_budget"]
    claim = {"state": "started", "experiment_id": CLAIM.stem, "supervisor_pid": os.getpid(),
             "supervisor_create_time": psutil.Process().create_time(), "started_unix_seconds": time.time(),
             "contract_sha256": run.dev.object_hash(value), **{key: 0 for key in (
                 "attempted_fits", "completed_fits", "attempted_vectorizer_fits", "completed_vectorizer_fits",
                 "attempted_forward_calls", "completed_forward_calls", "old_holdout_evaluations", "full_data_final_fits")}}
    run.write_json(CLAIM, claim)
    process, phase, peak = None, "pre_spawn", 0
    started = time.perf_counter()
    try:
        OUTPUT.mkdir(exist_ok=False, parents=True)
        run.write_json(OUTPUT / "contract.json", value)
        for source in SOURCES:
            target = OUTPUT / "source_snapshot" / source.relative_to(MODULE)
            target.parent.mkdir(exist_ok=True, parents=True)
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
        env = {**os.environ, **{key: str(budget["threads"]) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--execute"], env=env, stdout=stdout, stderr=stderr)
            monitored, phase = psutil.Process(process.pid), "worker"
            while process.poll() is None:
                if time.perf_counter() - started > budget["development_timeout_seconds"]:
                    raise TimeoutError("Final wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss + sum(p.memory_info().rss for p in monitored.children(recursive=True))
                    peak = max(peak, rss)
                    if rss > budget["max_host_rss_gib"] * 2**30:
                        raise MemoryError("Final process-tree RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed"
                or claim["attempted_fits"] != 1 or claim["completed_fits"] != 1
                or claim["attempted_vectorizer_fits"] != 4 or claim["completed_vectorizer_fits"] != 4
                or run.digest(OUTPUT / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Final worker failed or incomplete")
        if any(run.digest(MODULE / p) != sha for p, sha in value["sources"].items()):
            raise ValueError("Final sources changed")
        if run.digest(run.CLAIM) != value["development_claim_sha256"]:
            raise ValueError("Development claim changed")
        run.dev.verify_preserved_claims(config)
        receipt = {"state": "completed", "exit_code": 0, "wall_seconds": time.perf_counter() - started,
                   "sampled_peak_process_tree_rss_bytes": peak, "summary_sha256": claim["summary_sha256"],
                   "deployment_validation_complete": False}
        run.write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=run.digest(OUTPUT / "supervisor.json"))
        run.write_json(CLAIM, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = run.guards.terminate_tree(process)
        finally:
            claim = json.loads(CLAIM.read_bytes())
            claim.update(state="failed", exception_type=type(error).__name__, exception_message=str(error), phase=phase,
                         termination=termination, wall_seconds=time.perf_counter() - started, sampled_peak_process_tree_rss_bytes=peak)
            run.write_json(CLAIM, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        execute()
        return 0
    try:
        config, _, value = admission()
    except (OSError, ValueError, AssertionError, importlib.metadata.PackageNotFoundError) as error:
        print(json.dumps({"state": "blocked", "reason": f"{type(error).__name__}: {error}"}))
        return 2
    if args.run:
        print(json.dumps(supervise(config, value)))
    else:
        print(json.dumps({"state": "ready", "contract_sha256": run.dev.object_hash(value)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
