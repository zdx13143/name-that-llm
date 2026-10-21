"""A finite development-only comparison. No final-holdout or inference route.

The old holdout has been used. This separate, explicitly authorized protocol
uses only the original development rows; it does not resume the frozen search.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import platform
import resource
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

MODULE = Path(__file__).resolve().parents[1]
REPO = MODULE.parent
sys.path.insert(0, str(REPO / "src"))
from features import extract_features
from run_sweep import BLOCK_DEFS, VIEW_DEFS
from model_cpu import file_hash
from model_cpu_round2 import make_pipeline
from artifact_guards import object_hash

CONFIG = MODULE / "config/team_dev_20261021.json"
OUTPUT = MODULE / "outputs/team_dev_20261021"
CLAIM = MODULE / "outputs/continuation_claims/team-dev-groupkfold5-20261021.json"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_development(train_path, split_path, labels_path, config):
    """Verify immutable identity and only retain the permitted development text."""
    for path, key in ((train_path, "train_sha256"), (split_path, "split_sha256"),
                      (labels_path, "labels_sha256")):
        if file_hash(path) != config[key]:
            raise ValueError(f"Frozen {key} mismatch")
    manifest = json.loads(Path(split_path).read_text())
    if manifest["train_sha256"] != config["train_sha256"]:
        raise ValueError("Manifest refers to different training data")
    dev = manifest["development_indices"]
    hold = manifest["holdout_indices"]
    entries = manifest["rows"]
    if any(type(i) is not int for i in dev + hold):
        raise ValueError("Split indices must be integers")
    if (len(set(dev)) != len(dev) or len(set(hold)) != len(hold)
            or set(dev) & set(hold) or set(dev) | set(hold) != set(range(len(entries)))):
        raise ValueError("Split must be a complete disjoint partition")
    if any(row["row_index"] != i for i, row in enumerate(entries)):
        raise ValueError("Manifest row identity mismatch")
    if any(entries[i]["role"] != "development" for i in dev):
        raise ValueError("Development includes a protected row")
    if any(entries[i]["role"] != "holdout" for i in hold):
        raise ValueError("Holdout role mismatch")
    dev_groups = {entries[i]["prompt_id"] for i in dev}
    if dev_groups & {entries[i]["prompt_id"] for i in hold}:
        raise ValueError("Development shares protected prompt groups")
    if len(dev) != config["development_rows"] or len(dev_groups) != config["development_groups"]:
        raise ValueError("Unexpected development size")
    labels = json.loads(Path(labels_path).read_text())["labels"]
    if len(labels) != len(set(labels)) or len(labels) != 18:
        raise ValueError("Expected 18 distinct labels")
    selected = {}
    wanted = set(dev)
    count = 0
    with Path(train_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or sorted(reader.fieldnames) != sorted(["id", "prompt_id", "text", "model"]):
            raise ValueError("Training schema changed")
        for i, row in enumerate(reader):
            count += 1
            if i not in wanted:
                continue  # No protected text, labels or predictions enter returned data.
            if None in row or any(not row.get(k, "").strip() for k in reader.fieldnames):
                raise ValueError("Invalid development record")
            if row["id"] != entries[i]["id"] or row["prompt_id"] != entries[i]["prompt_id"]:
                raise ValueError("CSV and manifest row identity differ")
            if row["model"] not in labels:
                raise ValueError("Unknown development label")
            selected[i] = row
    if count != len(entries) or len(selected) != len(dev):
        raise ValueError("Row count mismatch")
    rows = [selected[i] for i in dev]
    ids = np.array([row["id"] for row in rows])
    if len(set(ids)) != len(ids):
        raise ValueError("Development IDs must be unique")
    return {
        "row_index": np.asarray(dev, dtype=np.int64), "ids": ids,
        "groups": np.array([row["prompt_id"] for row in rows]),
        "texts": np.array([row["text"] for row in rows]), "labels": labels,
        "truth": np.array([labels.index(row["model"]) for row in rows], dtype=np.int64),
    }


def grouped_folds(data, n_splits):
    folds = list(GroupKFold(n_splits=n_splits).split(data["texts"], data["truth"], data["groups"]))
    ids = np.full(len(data["truth"]), -1, dtype=np.int64)
    for number, (tr, va) in enumerate(folds):
        if set(data["groups"][tr]) & set(data["groups"][va]):
            raise ValueError("Training and validation share a prompt")
        if np.any(ids[va] != -1):
            raise ValueError("Fold coverage is duplicated")
        if set(data["truth"][tr]) != set(range(len(data["labels"]))):
            raise ValueError("Training fold lacks a reference class")
        ids[va] = number
    if np.any(ids < 0):
        raise ValueError("Fold coverage is incomplete")
    return folds, ids


def fingerprint_matrix(texts):
    return extract_features(texts)[0]


def build_model(spec, config):
    if spec["name"] == "original_style":
        return make_pipeline(config["original_config"], seed=config["seed"])
    view = spec["view"]
    if view == "fp":
        features = Pipeline([("counts", FunctionTransformer(fingerprint_matrix)),
                             ("scale", StandardScaler())])
    else:
        features = FeatureUnion([(block, TfidfVectorizer(lowercase=True, **BLOCK_DEFS[block]))
                                 for block in VIEW_DEFS[view]], n_jobs=1)
    return Pipeline([("features", features), ("classifier", LogisticRegression(
        C=spec["C"], max_iter=3000, class_weight="balanced", random_state=0))])


def validate_probabilities(probabilities, n_rows, n_labels):
    if probabilities.shape != (n_rows, n_labels) or probabilities.dtype.kind != "f":
        raise ValueError("Invalid probability schema")
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Invalid or uncovered probability rows")


def save_oof(path, probabilities, data, fold_id, contract_hash):
    validate_probabilities(probabilities, len(data["truth"]), len(data["labels"]))
    np.savez_compressed(path, probabilities=probabilities, row_index=data["row_index"],
                        ids=data["ids"], groups=data["groups"], truth=data["truth"],
                        labels=np.asarray(data["labels"]), fold_id=fold_id,
                        contract_sha256=np.asarray(contract_hash))


def load_oof(path, data, fold_id, contract_hash):
    with np.load(path, allow_pickle=False) as artifact:
        expected = {"row_index": data["row_index"], "ids": data["ids"],
                    "groups": data["groups"], "truth": data["truth"],
                    "labels": np.asarray(data["labels"]), "fold_id": fold_id,
                    "contract_sha256": np.asarray(contract_hash)}
        if set(artifact.files) != set(expected) | {"probabilities"}:
            raise ValueError("Unexpected OOF fields")
        for name, value in expected.items():
            if not np.array_equal(artifact[name], value):
                raise ValueError(f"OOF identity mismatch: {name}")
        probabilities = artifact["probabilities"].copy()
    validate_probabilities(probabilities, len(data["truth"]), len(data["labels"]))
    return probabilities


def metrics(probabilities, data, fold_id):
    prediction = probabilities.argmax(axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(data["truth"], prediction)),
        "fold_balanced_accuracy": [float(balanced_accuracy_score(data["truth"][fold_id == i],
                                    prediction[fold_id == i])) for i in np.unique(fold_id)],
        "per_class_recall": dict(zip(data["labels"], recall_score(data["truth"], prediction,
                             labels=np.arange(len(data["labels"])), average=None, zero_division=0).tolist())),
    }


def paired_interval(base, candidate, data, config):
    groups, gi = np.unique(data["groups"], return_inverse=True)
    shape = (len(groups), len(data["labels"]))
    counts, hits_base, hits_candidate = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    truth = data["truth"]
    np.add.at(counts, (gi, truth), 1)
    np.add.at(hits_base, (gi, truth), base.argmax(axis=1) == truth)
    np.add.at(hits_candidate, (gi, truth), candidate.argmax(axis=1) == truth)
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)),
                            size=config["bootstrap_repeats"])
    denominators = draws @ counts
    if np.any(denominators == 0):
        raise ValueError("Bootstrap omitted a class")
    differences = ((draws @ (hits_candidate - hits_base)) / denominators).mean(axis=1)
    return np.quantile(differences, [.025, .975]).tolist()


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def claim_run(path, contract_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = {"state": "running", "contract_sha256": contract_hash, "attempted_fits": 0,
             "completed_fits": 0, "final_holdout_evaluations": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with path.open("x") as handle:
        json.dump(claim, handle, indent=2)
    return claim


def sources():
    paths = [Path(__file__), CONFIG, MODULE / "docs/TEAM_CONTINUATION_PLAN.md",
             MODULE / "scripts/model_cpu.py", MODULE / "scripts/model_cpu_round2.py",
             MODULE / "scripts/artifact_guards.py", REPO / "src/features.py",
             REPO / "src/run_sweep.py", REPO / "src/cv.py", REPO / "src/common.py"]
    return {str(p.relative_to(REPO)): file_hash(p) for p in paths}


def run(data, folds, fold_id, config, contract):
    contract_hash = object_hash(contract)
    claim = claim_run(CLAIM, contract_hash)
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError("Predeclared experiment wall-clock budget exhausted")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, config["max_seconds"])
    try:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        write_json(OUTPUT / "contract.json", contract)
        write_json(OUTPUT / "config.json", config)
        (OUTPUT / "source_snapshot").mkdir()
        for relative in contract["source_sha256"]:
            destination = OUTPUT / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / relative).read_bytes())
        write_json(OUTPUT / "fold_manifest.json", {"row_index": data["row_index"].tolist(),
            "ids": data["ids"].tolist(), "groups": data["groups"].tolist(), "fold_id": fold_id.tolist()})
        probabilities, results = {}, {}
        with threadpool_limits(limits=config["threads"]):
            for spec in config["models"]:
                name = spec["name"]
                oof = np.full((len(data["truth"]), len(data["labels"])), np.nan, dtype=np.float64)
                fit_records = []
                for number, (tr, va) in enumerate(folds):
                    if claim["attempted_fits"] >= config["max_fits"]:
                        raise RuntimeError("Fit budget exhausted")
                    claim["attempted_fits"] += 1
                    claim["current"] = {"model": name, "fold": number}
                    write_json(CLAIM, claim)
                    start_fit = time.monotonic()
                    model = build_model(spec, config)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(data["texts"][tr], data["truth"][tr])
                        if not np.array_equal(model.classes_, np.arange(len(data["labels"]))):
                            raise ValueError("Classifier class order mismatch")
                        predicted = model.predict_proba(data["texts"][va])
                    validate_probabilities(predicted, len(va), len(data["labels"]))
                    oof[va] = predicted
                    convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                    record = {"fold": number, "seconds": time.monotonic() - start_fit,
                              "convergence_warnings": convergence, "peak_rss_bytes": peak_rss()}
                    fit_records.append(record)
                    claim["completed_fits"] += 1
                    write_json(CLAIM, claim)
                    write_json(OUTPUT / f"{name}_progress.json", fit_records)
                    print(json.dumps({"model": name, **record}), flush=True)
                    del model
                    gc.collect()
                    if convergence:
                        raise RuntimeError("Convergence warning: stop rather than silently retune")
                    if record["peak_rss_bytes"] > config["max_rss_bytes"]:
                        raise RuntimeError("Observed RSS budget exceeded")
                artifact = OUTPUT / f"{name}_oof.npz"
                save_oof(artifact, oof, data, fold_id, contract_hash)
                probabilities[name] = load_oof(artifact, data, fold_id, contract_hash)
                results[name] = {**metrics(oof, data, fold_id), "fits": fit_records,
                                 "oof_sha256": file_hash(artifact)}
                print(json.dumps({"model": name, "balanced_accuracy": results[name]["balanced_accuracy"]}), flush=True)
            for spec in config["models"][1:]:
                for weight in config["blend_team_weights"]:
                    name = f"blend_{spec['name']}_{weight:g}"
                    probabilities[name] = (1 - weight) * probabilities["original_style"] + weight * probabilities[spec["name"]]
                    results[name] = {**metrics(probabilities[name], data, fold_id),
                                     "fixed_team_weight": weight, "team_component": spec["name"], "fits": []}
            base = results["original_style"]
            for name, result in results.items():
                result["delta_vs_original"] = result["balanced_accuracy"] - base["balanced_accuracy"]
                result["paired_prompt_ci95"] = paired_interval(probabilities["original_style"], probabilities[name], data, config)
                result["positive_folds"] = int(np.sum(np.array(result["fold_balanced_accuracy"]) > np.array(base["fold_balanced_accuracy"])))
                result["passes_development_gate"] = bool(result["delta_vs_original"] >= config["adoption_min_delta"]
                    and result["paired_prompt_ci95"][0] > 1e-12
                    and result["positive_folds"] >= config["adoption_min_positive_folds"])
        if sources() != contract["source_sha256"]:
            raise RuntimeError("Source changed during experiment")
        summary = {"protocol": config["protocol"], "scope": "adaptive development only; no new independent generalization estimate",
                   "contract_sha256": contract_hash, "attempted_fits": claim["attempted_fits"],
                   "completed_fits": claim["completed_fits"], "wall_seconds": time.monotonic() - started,
                   "peak_rss_bytes": peak_rss(), "final_holdout_evaluations": 0, "results": results,
                   "engineering_stop": "predeclared finite comparison completed; old frozen delivery retained"}
        write_json(OUTPUT / "summary.json", summary)
        claim.update(state="completed", summary_sha256=file_hash(OUTPUT / "summary.json"), wall_seconds=summary["wall_seconds"])
        write_json(CLAIM, claim)
        print(json.dumps({"completed_fits": claim["completed_fits"], "wall_seconds": summary["wall_seconds"], "results":
                          {name: result["balanced_accuracy"] for name, result in results.items()}}), flush=True)
    except BaseException as error:
        claim.update(state="failed", error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
        write_json(CLAIM, claim)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Perform the one finite run; default only audits inputs")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    data = load_development(args.train, args.split, MODULE / "config/labels.json", config)
    folds, fold_id = grouped_folds(data, config["folds"])
    contract = {"protocol": config["protocol"], "config_sha256": file_hash(CONFIG),
        "train_sha256": config["train_sha256"], "split_sha256": config["split_sha256"],
        "labels_sha256": config["labels_sha256"], "ordered_ids_sha256": object_hash(data["ids"].tolist()),
        "fold_id_sha256": object_hash(fold_id.tolist()), "labels": data["labels"],
        "development_rows": len(data["truth"]), "development_groups": len(set(data["groups"])),
        "folds": "sklearn GroupKFold(5), shuffle=False, on original development only",
        "source_sha256": sources(), "python": platform.python_version(), "platform": platform.platform(),
        "packages": {package: importlib.metadata.version(package) for package in
                     ("numpy", "scipy", "scikit-learn", "pandas", "joblib", "threadpoolctl")},
        "final_holdout_used_previously": True, "final_holdout_evaluations_this_run": 0}
    print(json.dumps({"audit": "passed", "development_rows": len(data["truth"]),
        "development_groups": len(set(data["groups"])), "contract_sha256": object_hash(contract),
        "claim_exists": CLAIM.exists()}), flush=True)
    if args.run:
        run(data, folds, fold_id, config, contract)


if __name__ == "__main__":
    main()
