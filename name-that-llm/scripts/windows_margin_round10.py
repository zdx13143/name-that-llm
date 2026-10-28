"""One fixed R10 development transform, then one separately claimed final fit."""
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
import windows_topic_word_round9 as r9
import windows_modernbert_dev as dev
import windows_stylegram_round5 as guards
from probe_modernbert import digest, write_json
from supervisor_round7 import require_live_supervisor
from margin_center_round10 import center_margins, predict, synthetic_inputs
from topic_word_round9 import transform_views
from weighted_style_round7 import classifier, sample_weights
from windows_prompt_gaussian_round3 import paired_comparison

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_margin_round10_20261028.json"
CONFIG_SHA256 = "a8848e6aeb4b98eb364cfc3abd3f8c5395e52e564925c7232117e8084c2e05ce"
SOURCES = tuple(dict.fromkeys((*r9.SOURCES, CONFIG, Path(__file__),
    Path(__file__).with_name("margin_center_round10.py"),
    MODULE / "docs/WINDOWS_ROUND10_PLAN_20261028.md")))
IDENTITY = ("row_index", "ids", "groups", "truth", "labels", "fold_id")


def paths(stage):
    stem = "windows-margin-center-dev-20261028-r10" if stage == "development" else "windows-topic-word-final-20261028-r10"
    folder = "windows_margin_20261028_r10" if stage == "development" else "windows_topic_word_final_20261028_r10"
    return MODULE / "outputs/continuation_claims" / (stem + ".json"), MODULE / "outputs" / folder


def configuration():
    if digest(CONFIG) != CONFIG_SHA256 or sys.platform != "win32":
        raise ValueError("Registered configuration/platform changed")
    c = json.loads(CONFIG.read_bytes())
    for name, version in c["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Runtime changed: " + name)
    return c


def verify_files(c):
    for name, expected in {**c["input_sha256"], **c["preserved_claims"]}.items():
        path = (MODULE / name).resolve()
        if not path.is_relative_to(MODULE) or digest(path) != expected:
            raise ValueError("Frozen input changed: " + name)


def load_arrays(c):
    result = {}
    for key, path in [("r7", MODULE / c["r7_output"] / "candidate_oof.npz"),
                      *[(n, MODULE / c["baseline_output"] / f"{n}_oof.npz") for n in ("candidate", "half", "singleton")]]:
        with np.load(path, allow_pickle=False) as z:
            result[key] = {k: z[k].copy() for k in z.files}
    data = result["candidate"]
    if len(data["ids"]) != 2159 or len(np.unique(data["groups"])) != 120 or len(data["labels"]) != 18:
        raise ValueError("Original development axes changed")
    for saved in result.values():
        for key in IDENTITY:
            np.testing.assert_array_equal(saved[key], data[key])
        dev.probabilities(saved["probabilities"], 2159, 18)
    for i in range(5):
        if set(data["truth"][data["fold_id"] == i]) != set(range(18)):
            raise ValueError("Every registered fold must contain every class")
    for g in np.unique(data["groups"]):
        if len(np.unique(data["fold_id"][data["groups"] == g])) != 1:
            raise ValueError("Prompt crosses folds")
    return data, result


def selected_method(c):
    claim_path, folder = paths("development")
    claim = json.loads(claim_path.read_bytes())
    summary = json.loads((folder / "summary.json").read_bytes())
    verified = json.loads((folder / "verification.json").read_bytes())
    if (claim["state"] != "completed" or verified["state"] != "verified"
            or digest(folder / "summary.json") != claim["summary_sha256"]
            or digest(folder / "supervisor.json") != claim["supervisor_sha256"]
            or verified["summary_sha256"] != claim["summary_sha256"]
            or verified["margin_correction"] != summary["margin_correction"]):
        raise ValueError("Development evidence is incomplete")
    return summary["margin_correction"], {p.name: digest(p) for p in (claim_path, folder / "summary.json", folder / "verification.json", folder / "supervisor.json")}


def final_inputs(c):
    data, pool, _, _, fold_id, _ = r9.load_inputs(r9.configuration())
    selected, present = r9.r7.fold_pool(pool, data["texts"], data["truth"])
    old = json.loads((MODULE / c["r7_final_output"] / "contract.json").read_bytes())
    identity = dev.object_hash({k: data[k].tolist() for k in ("row_index", "ids", "groups", "truth", "labels")})
    if (identity != old["row_identity_sha256"] or len(selected) != old["external_rows"] or len(selected) != 38724
            or dev.object_hash(pool["ids"][selected].tolist()) != old["external_ids_sha256"]
            or present.tolist() != old["punctuation_present"]):
        raise ValueError("Final R7 vocabulary training rows changed")
    return data, pool, selected, identity


def contract(c, stage):
    verify_files(c)
    value = {"config": c, "stage": stage, "sources": {p.relative_to(MODULE).as_posix(): digest(p) for p in SOURCES},
             "python": sys.version, "platform": sys.platform}
    if stage == "development":
        data, _ = load_arrays(c)
        value["identity_sha256"] = dev.object_hash({k: data[k].tolist() for k in IDENTITY})
    else:
        enabled, evidence = selected_method(c)
        _, _, _, identity = final_inputs(c)
        value.update(margin_correction=enabled, development_evidence=evidence, identity_sha256=identity)
    return value


def evaluate(c, data, outputs, prior):
    fold_id, e = data["fold_id"], c["evaluation"]
    metrics = {n: dev.metrics(p, data, fold_id, e) for n, p in outputs.items()}
    comparisons = {n: paired_comparison(p, prior[n]["probabilities"], data, fold_id, e) for n, p in outputs.items()}
    r7 = paired_comparison(outputs["candidate"], prior["r7"]["probabilities"], data, fold_id, e)
    m, delta, g = metrics["candidate"], comparisons["candidate"], c["candidate_gate"]
    checks = {"pooled": m["balanced_accuracy"] >= g["min_pooled_balanced_accuracy"],
              "each_fold": min(m["fold_balanced_accuracy"]) >= g["min_each_fold_balanced_accuracy"],
              "gain_over_r9": delta["pooled_ba_gain"] >= g["min_gain_over_r9"],
              "positive_interval": delta["prompt_paired_gain_95ci"][0] > 0,
              "improved_folds": delta["improved_folds"] >= g["min_improved_folds"],
              "total_gain_over_r7": r7["pooled_ba_gain"] >= g["min_gain_over_r7"],
              "half_batch": comparisons["half"]["pooled_ba_gain"] >= g["min_half_gain_over_r9_half"],
              "singleton": comparisons["singleton"]["pooled_ba_gain"] >= g["min_singleton_gain_over_r9_singleton"]}
    changes = {}
    for n, p in outputs.items():
        before, after, y = prior[n]["probabilities"].argmax(1), p.argmax(1), data["truth"]
        changes[n] = {"correct": int((after == y).sum()), "changed": int((before != after).sum()),
                      "corrected": int(((before != y) & (after == y)).sum()), "harmed": int(((before == y) & (after != y)).sum())}
    return {"metrics": metrics, "comparisons_over_r9": comparisons, "comparison_over_r7": r7,
            "changes": changes, "gate": {"passed": all(checks.values()), "checks": checks},
            "margin_correction": all(checks.values()), "selected_method": "R10" if all(checks.values()) else "R9_requested"}


def development(c, claim, cp, out):
    data, prior = load_arrays(c)
    outputs = {n: np.full((2159, 18), np.nan) for n in ("candidate", "half", "singleton")}
    records = []
    def transform(s, g):
        dev.consume(claim, "attempted_transforms", 20, cp)
        result = center_margins(s, g)
        claim["completed_transforms"] += 1
        write_json(cp, claim, exclusive=False)
        return result
    for i in range(5):
        rows = np.flatnonzero(data["fold_id"] == i)
        with np.load(MODULE / c["baseline_output"] / f"fold{i}/scores.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["valid_positions"], rows)
            np.testing.assert_array_equal(z["labels"], data["labels"])
            result = transform(z["candidate"], z["groups"])
            reverse = transform(z["candidate"][::-1], z["groups"][::-1])[::-1]
            np.testing.assert_allclose(result, reverse, atol=1e-10, rtol=0)
            if not np.array_equal(result.argmax(1), reverse.argmax(1)):
                raise ValueError("Permutation changed labels")
            items = {"candidate": result, "half": transform(z["half"], z["half_groups"]),
                     "singleton": transform(z["singleton"], np.arange(len(rows)))}
            for name, score in items.items():
                np.testing.assert_allclose(softmax(z[name], axis=1), prior[name]["probabilities"][rows], atol=1e-12, rtol=0)
                outputs[name][rows] = softmax(score, axis=1)
            with (out / f"fold{i}_scores.npz").open("xb") as f:
                np.savez_compressed(f, **items, valid_positions=rows)
            records.append({"fold": i, "rows": len(rows), "reverse_max_abs_error": float(np.abs(result-reverse).max()),
                            "scores_sha256": digest(out / f"fold{i}_scores.npz")})
    for name, p in outputs.items():
        dev.probabilities(p, 2159, 18)
        dev.save_array(out / f"{name}_oof.npz", "probabilities", p, data, data["fold_id"], claim["contract_sha256"])
    # Ground truth is used only after all candidate arrays have been saved.
    summary = evaluate(c, data, outputs, prior)
    summary.update(folds=records, contract_sha256=claim["contract_sha256"], new_fits=0, new_groupings=0,
                   encoder_forward_calls=0, old_holdout_evaluations=0,
                   oof_sha256={n: digest(out / f"{n}_oof.npz") for n in outputs})
    return summary


def final_fit(c, claim, cp, out, value):
    data, pool, selected, _ = final_inputs(c)
    old = joblib.load(MODULE / c["r7_final_output"] / "weighted_svc.joblib")
    old_contract = json.loads((MODULE / c["r7_final_output"] / "contract.json").read_bytes())
    if old["contract_sha256"] != dev.object_hash(old_contract) or old["labels"].tolist() != data["labels"].tolist():
        raise ValueError("Frozen final vocabulary contract differs")
    views = old["views"]
    if [(n, w) for n, _, w in views] != [(s["name"], s["weight"]) for s in c["method"]["views"]]:
        raise ValueError("View identity changed")
    del old
    texts = data["texts"].tolist() + pool["texts"][selected].tolist()
    groups = np.asarray(["official:"+str(g) for g in data["groups"]] + ["external:"+str(g) for g in pool["groups"][selected]])
    y = np.concatenate([data["truth"], pool["truth"][selected]])
    print("final frozen-view transform started", flush=True)
    matrix, nnz = transform_views(views, texts, groups, c["method"]["alpha"])
    dev.consume(claim, "attempted_fits", 1, cp)
    print("final classifier fit started", flush=True)
    model = classifier(c["method"]["classifier"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(matrix, y, sample_weight=sample_weights(2159, len(selected), 4))
    claim["completed_fits"] += 1
    write_json(cp, claim, exclusive=False)
    del matrix, texts, groups, y
    artifact = {"views": views, "classifier": model, "labels": data["labels"], "contract_sha256": claim["contract_sha256"],
                "word_alpha": .5, "grouping": c["method"]["grouping"], "margin_correction": value["margin_correction"]}
    with (out / "topic_svc.joblib").open("xb") as f:
        joblib.dump(artifact, f, compress=3)
    texts, ids = synthetic_inputs()
    scores, groups = predict(artifact, texts, ids)
    restored, restored_groups = predict(joblib.load(out / "topic_svc.joblib"), texts, ids)
    reversed_scores, reversed_groups = predict(artifact, texts[::-1], ids[::-1])
    np.testing.assert_array_equal(groups, restored_groups)
    np.testing.assert_array_equal(groups, reversed_groups[::-1])
    error = max(float(np.abs(scores-restored).max()), float(np.abs(scores-reversed_scores[::-1]).max()))
    if error > 1e-7:
        raise ValueError("Synthetic export/order mismatch")
    if sorted(np.unique(groups, return_counts=True)[1]) != [2] * 40:
        raise ValueError("Synthetic grouping did not exercise correction")
    with (out / "synthetic_scores.npz").open("xb") as f:
        np.savez_compressed(f, scores=scores, groups=groups, texts=np.asarray(texts), ids=np.asarray(ids),
                            labels=data["labels"], contract_sha256=claim["contract_sha256"])
    return {"state": "model_exported", "contract_sha256": claim["contract_sha256"], "margin_correction": value["margin_correction"],
            "official_training_rows": 2159, "official_training_groups": 120, "external_training_rows": len(selected),
            "classifier_fits": 1, "new_supervised_vectorizer_fits": 0, "old_holdout_evaluations": 0, "full_2699_row_fits": 0,
            "encoder_forward_calls": 0, "iterations": int(model.n_iter_), "word_nnz": nnz,
            "synthetic_rows": 80, "synthetic_prediction_calls": 3, "synthetic_max_abs_error": error,
            "model_sha256": digest(out / "topic_svc.joblib"), "synthetic_scores_sha256": digest(out / "synthetic_scores.npz")}


def execute(c, stage):
    cp, out = paths(stage)
    claim = json.loads(cp.read_bytes())
    require_live_supervisor(claim)
    if claim["attempted_transforms"] or claim["completed_transforms"]:
        raise ValueError("Transforms already attempted")
    guards.start_worker_guard(claim, c[stage+"_budget"])
    value = contract(c, stage)
    if dev.object_hash(value) != claim["contract_sha256"]:
        raise ValueError("Execution contract changed")
    with threadpool_limits(limits=2):
        result = development(c, claim, cp, out) if stage == "development" else final_fit(c, claim, cp, out, value)
    verify_files(c)
    write_json(out / "summary.json", result)
    claim.update(state="worker_completed", summary_sha256=digest(out / "summary.json"))
    write_json(cp, claim, exclusive=False)


def supervise(c, stage, value):
    cp, out = paths(stage)
    budget = c[stage+"_budget"]
    claim = {"state": "started", "experiment_id": cp.stem, "supervisor_pid": os.getpid(),
             "supervisor_create_time": psutil.Process().create_time(), "started_unix_seconds": time.time(),
             "contract_sha256": dev.object_hash(value), "old_holdout_evaluations": 0}
    claim.update({k: 0 for k in ("attempted_fits", "completed_fits", "attempted_forward_calls", "completed_forward_calls",
                                "attempted_vectorizer_fits", "completed_vectorizer_fits", "attempted_transforms", "completed_transforms")})
    write_json(cp, claim)
    process, peak, started = None, 0, time.monotonic()
    try:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "contract.json", value)
        for source in SOURCES:
            target = out / "source_snapshot" / source.relative_to(MODULE)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(source.read_bytes())
        env = {**os.environ, **{k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), stage, "--execute"], env=env, stdout=stdout, stderr=stderr)
            watched = psutil.Process(process.pid)
            while process.poll() is None:
                if time.monotonic()-started > budget["development_timeout_seconds"]:
                    raise TimeoutError("Stage wall clock exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + watched.memory_info().rss + sum(p.memory_info().rss for p in watched.children(recursive=True))
                    peak = max(peak, rss)
                    if rss > budget["max_host_rss_gib"] * 2**30:
                        raise MemoryError("Stage process-tree RSS exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        claim = json.loads(cp.read_bytes())
        expected = {"attempted_fits": budget["attempted_fits"], "completed_fits": budget["attempted_fits"],
                    "attempted_transforms": budget["attempted_transforms"], "completed_transforms": budget["attempted_transforms"],
                    "attempted_vectorizer_fits": 0, "completed_vectorizer_fits": 0, "attempted_forward_calls": 0,
                    "completed_forward_calls": 0, "old_holdout_evaluations": 0}
        if (process.returncode != 0 or claim["state"] != "worker_completed" or any(claim[k] != v for k, v in expected.items())
                or digest(out / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Incomplete stage or operation budget mismatch")
        verify_files(c)
        if any(digest(MODULE / p) != sha for p, sha in value["sources"].items()):
            raise ValueError("Execution source changed")
        receipt = {"state": "completed", "stage": stage, "exit_code": 0, "wall_seconds": time.monotonic()-started,
                   "sampled_peak_process_tree_rss_bytes": peak, "summary_sha256": claim["summary_sha256"], **expected}
        write_json(out / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(out / "supervisor.json"))
        write_json(cp, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = guards.terminate_tree(process) if process is not None else None
        claim = json.loads(cp.read_bytes())
        claim.update(state="failed", error=str(error), error_type=type(error).__name__, termination=termination,
                     wall_seconds=time.monotonic()-started, sampled_peak_process_tree_rss_bytes=peak)
        write_json(cp, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development", "final"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    c = configuration()
    if args.execute:
        execute(c, args.stage)
        return
    cp, out = paths(args.stage)
    if cp.exists() or out.exists():
        raise SystemExit("blocked: claim/output exists; retry and resume forbidden")
    value = contract(c, args.stage)
    print(json.dumps(supervise(c, args.stage, value) if args.run else {"state": "ready", "stage": args.stage, "contract_sha256": dev.object_hash(value)}))


if __name__ == "__main__":
    main()
