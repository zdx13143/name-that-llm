#!/usr/bin/env python3
"""One shared, durable holdout claim across all CPU search rounds.

Guardrail-only revisions to historical search drivers may be matched against
their exact archived source. Executed model implementations must still match
the selected experiment's current source hashes. A failed claimed evaluation
is never retried automatically.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

from model_cpu import checked_submission, config_id, file_hash, read_csv
from artifact_guards import assert_holdout_available, validate_prior

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify_frozen_code(summary, run_dir):
    if "model_cpu.py" not in summary["code_sha256"]:
        raise ValueError("Frozen base model implementation is not bound")
    config = summary["selected"]["config"]
    needs_extension = (config.get("style_weight", 0) != 0 or config.get("char_analyzer", "char") != "char"
                       or config.get("classifier") == "score_fusion")
    if needs_extension and "model_cpu_round2.py" not in summary["code_sha256"]:
        raise ValueError("Selected model requires a bound extension implementation")
    verified = {}
    for name, digest in summary["code_sha256"].items():
        current = ROOT / "scripts" / name
        if file_hash(current) == digest:
            verified[name] = {"sha256": digest, "source": "current"}
            continue
        # Prediction semantics cannot be silently replaced by a newer model.
        if name.startswith("model_"):
            raise ValueError(f"Frozen model implementation changed: {name}")
        snapshot = run_dir / "code_snapshot" / name
        if not snapshot.is_file() or file_hash(snapshot) != digest:
            raise ValueError(f"Exact historical search source unavailable: {name}")
        verified[name] = {"sha256": digest, "source": "exact_code_snapshot"}
    return verified


def verify_runtime(summary):
    actual = {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
              "numpy": np.__version__, **{name: importlib.metadata.version(name)
                  for name in ("scipy", "pyarrow", "joblib")}}
    expected = summary["environment"]
    for name in ("python", "scikit_learn", "numpy"):
        if expected.get(name) != actual[name]:
            raise ValueError(f"Runtime version differs from selected experiment: {name}")
    for name in ("scipy", "pyarrow", "joblib"):
        if name in expected and expected[name] != actual[name]:
            raise ValueError(f"Runtime version differs from selected experiment: {name}")
    return actual


def claim_holdout(run_dir, train_hash, split_hash):
    run_dir = Path(run_dir).resolve()
    assert_holdout_available(train_hash, ROOT)
    related = []
    for summary_path in sorted((ROOT / "outputs").glob("*/search_summary.json")):
        if json.loads(summary_path.read_text())["train_sha256"] == train_hash:
            related.append(summary_path.parent)
    if run_dir not in related:
        related.append(run_dir)
    if any((folder / "holdout_opened.json").exists() for folder in related):
        raise ValueError("Shared final holdout was already claimed or opened")
    directory = ROOT / "outputs" / "holdout_claims"
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / f"{train_hash}.json"
    metadata = {"status": "claimed_before_evaluation", "claimed_at_unix": time.time(),
                "selected_run": run_dir.name, "train_sha256": train_hash,
                "split_sha256": split_hash, "automatic_retry_allowed": False}
    try:
        with claim.open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("Shared final holdout already has a durable claim") from exc
    # All related rounds are marked before any holdout fit or prediction, and
    # the exclusive claim remains even if writing a per-round marker fails.
    for folder in related:
        save(folder / "holdout_opened.json", metadata)
    return claim, metadata


def holdout_interval(y, predictions, groups):
    unique = np.unique(groups)
    classes = np.unique(y)
    totals = np.zeros((len(unique), len(classes)))
    correct = np.zeros_like(totals)
    for index, group in enumerate(unique):
        for column, label in enumerate(classes):
            mask = (groups == group) & (y == label)
            totals[index, column] = mask.sum()
            correct[index, column] = (predictions[mask] == y[mask]).sum()
    weights = np.random.default_rng(20261218).multinomial(len(unique),
                np.full(len(unique), 1 / len(unique)), size=2000)
    sampled = ((weights @ correct) / (weights @ totals)).mean(axis=1)
    return {"ci95_low": float(np.quantile(sampled, .025)),
            "ci95_high": float(np.quantile(sampled, .975)), "group_bootstrap_draws": 2000}


def finalize_once(args):
    run_dir = Path(args.output).resolve()
    if (run_dir / "holdout_opened.json").exists():
        raise ValueError("Shared final holdout was already opened or claimed")
    summary_path = run_dir / "search_summary.json"
    summary = json.loads(summary_path.read_text())
    if "selected" not in summary:
        raise ValueError("Complete and freeze model selection before opening the holdout")
    if file_hash(run_dir / "split_manifest.json") != summary["split_sha256"]:
        raise ValueError("Frozen split changed since model selection")
    verified_code = verify_frozen_code(summary, run_dir)
    runtime = verify_runtime(summary)
    validated = validate_prior(run_dir, args.data_dir, ROOT / "config" / "labels.json")
    if file_hash(ROOT / "config" / "labels.json") != summary["label_reference_sha256"]:
        raise ValueError("Label reference changed since model selection")
    if file_hash(args.data_dir / "train.csv") != summary["train_sha256"]:
        raise ValueError("Training input changed since model selection")
    rows = read_csv(args.data_dir / "train.csv")
    labels = json.loads((ROOT / "config" / "labels.json").read_text())["labels"]
    manifest = json.loads((run_dir / "split_manifest.json").read_text())
    dev, heldout = np.array(manifest["development_indices"]), np.array(manifest["holdout_indices"])
    groups = np.array([row["prompt_id"] for row in rows])
    y = np.array([labels.index(row["model"]) for row in rows])
    if (set(dev) & set(heldout) or set(dev) | set(heldout) != set(range(len(rows)))
            or set(groups[dev]) & set(groups[heldout])):
        raise ValueError("Frozen split is not a complete disjoint prompt partition")
    config = summary["selected"]["config"]
    if config_id(config) != summary["selected"]["id"]:
        raise ValueError("Selected configuration identifier changed")
    claim, metadata = claim_holdout(run_dir, summary["train_sha256"], summary["split_sha256"])
    selected = {"selected_config": config, "selected_id": config_id(config),
                "search_summary_sha256": file_hash(summary_path), "verified_code": verified_code,
                "finalization_code_sha256": file_hash(Path(__file__)),
                "holdout_claim": claim.name, "no_further_tuning": True,
                "artifact_contract_sha256": validated["contract_sha256"], "verified_runtime": runtime}
    save(run_dir / "frozen_selection.json", selected)
    save(run_dir / "selected_config.json", config)
    try:
        if "model_cpu_round2.py" in summary["code_sha256"]:
            from model_cpu_round2 import make_pipeline
        else:
            from model_cpu import make_pipeline
        start = time.monotonic()
        texts = np.array([row["text"] for row in rows])
        model = make_pipeline(config)
        model.fit(texts[dev], y[dev])
        predictions = model.predict(texts[heldout])
        np.savez_compressed(run_dir / "holdout_predictions.npz", row_index=heldout,
                            truth=y[heldout], prediction=predictions)
        final = {"selected_id": config_id(config), "heldout_rows": len(heldout),
            "heldout_groups": len(set(groups[heldout])), "holdout_evaluations": 1,
            "balanced_accuracy": float(balanced_accuracy_score(y[heldout], predictions)),
            "group_bootstrap": holdout_interval(y[heldout], predictions, groups[heldout]),
            "per_class_recall": dict(zip(labels, recall_score(y[heldout], predictions,
                                     labels=np.arange(len(labels)), average=None).tolist())),
            "confusion_matrix": confusion_matrix(y[heldout], predictions, labels=np.arange(len(labels))).tolist(),
            "label_order": labels, "holdout_fit_seconds": time.monotonic() - start,
            "evidence_scope": "Final local prompt-group holdout opened once after selection; not a leaderboard score."}
        save(run_dir / "holdout_metrics.json", final)
        begin = time.monotonic()
        model = make_pipeline(config)
        model.fit(texts, y)
        joblib.dump({"pipeline": model, "labels": labels, "config": config,
                     "train_sha256": summary["train_sha256"], "sklearn_version": sklearn.__version__}, run_dir / "model.joblib")
        test = read_csv(args.data_dir / "test.csv")
        test_predictions = [labels[index] for index in model.predict([row["text"] for row in test])]
        check = checked_submission(test, test_predictions, labels, run_dir / "submission.parquet")
        final.update({"all_train_fit_seconds": time.monotonic() - begin,
                      "model_bytes": (run_dir / "model.joblib").stat().st_size,
                      "public_test_format_check": check,
                      "additional_final_fits": 4 if config.get("classifier") == "score_fusion" else 2,
                      "formal_competition_submission": False})
        save(run_dir / "holdout_metrics.json", final)
        save(claim, {**metadata, "status": "completed", "completed_at_unix": time.time()})
        print(json.dumps({"holdout_balanced_accuracy": final["balanced_accuracy"],
                          "group_bootstrap": final["group_bootstrap"], "format_check": check}))
        return final
    except BaseException:
        save(claim, {**metadata, "status": "failed_after_claim", "automatic_retry_allowed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=2):
        finalize_once(parser.parse_args())
