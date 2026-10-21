"""Finite development-only centering experiment; no holdout/test inference route.

The CPU runner is POSIX-only (resource and alarm timer). The separately provided
GPU environment probe is cross-platform. Completed experiment claims cannot resume.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import signal
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from compare_team_cpu import (MODULE, REPO, canonical_probabilities, claim_run,
    grouped_folds, load_development, load_oof, metrics, paired_interval, peak_rss,
    positive_fold_count, save_oof, sources as previous_sources, write_json)
from artifact_guards import object_hash
from model_cpu import file_hash

CONFIG = MODULE / "config/prompt_centering_20261021.json"
OUTPUT = MODULE / "outputs/prompt_centering_20261021"
CLAIM = MODULE / "outputs/continuation_claims/prompt-centering-dev-20261021.json"
BASELINE = MODULE / "outputs/team_dev_20261021/recovery_01/original_style_oof.npz"


def soft_center(matrix, groups, strength, max_bytes):
    """Subtract each row's group mean; bound the union-support allocation first."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != matrix.shape[0]:
        raise ValueError("One nonmissing group is required per row")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Centering strength must be in [0, 1]")
    if any(value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
           for value in groups):
        raise ValueError("Missing groups are not permitted")
    x = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if not len(groups):
        return x
    x = normalize(x, copy=False).tocsr()
    if strength == 0:
        return x
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    # Singleton means are zero so single-item inference retains its input.
    weights = np.where(counts[inverse] > 1, 1.0 / counts[inverse], 0.0)
    membership = sparse.csr_matrix((weights, (inverse, np.arange(len(groups)))),
                                  shape=(len(counts), len(groups)))
    membership.eliminate_zeros()
    means = (membership @ x).tocsr()
    # Upper bound for union of input and repeated means, using 64-bit indices.
    estimated_nnz = x.nnz + int(np.dot(np.diff(means.indptr), counts))
    if estimated_nnz * 16 + (len(groups) + 1) * 8 > max_bytes:
        raise MemoryError("Centered CSR admission budget exceeded")
    centered = x - strength * means[inverse]
    centered.eliminate_zeros()
    return normalize(centered, copy=False).tocsr()


def cluster_batch(texts, ids, threshold, max_features):
    """Transductive text-only clustering, without prompt IDs or class labels."""
    ids, texts = np.asarray(ids), np.asarray(texts)
    if len(ids) != len(texts) or len(set(ids)) != len(ids):
        raise ValueError("Unique row IDs must align with texts")
    if not 0 < threshold < 1:
        raise ValueError("Cosine threshold must be in (0, 1)")
    order = np.argsort(ids, kind="stable")
    labels = np.arange(len(texts), dtype=np.int64)
    if len(texts) < 2:
        return labels
    try:
        x = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english",
             sublinear_tf=True, max_features=max_features, dtype=np.float64).fit_transform(texts[order])
    except ValueError as error:
        if "empty vocabulary" in str(error):
            return labels
        raise
    nonzero = np.flatnonzero(np.asarray(x.getnnz(axis=1)).ravel() > 0)
    if len(nonzero) < 2:
        return labels
    similarities = (x[nonzero] @ x[nonzero].T).toarray()
    distance = np.clip(1.0 - similarities, 0.0, 1.0)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)
    clusters = AgglomerativeClustering(n_clusters=None, metric="precomputed",
               linkage="average", distance_threshold=threshold).fit_predict(distance)
    # Offset assigned labels to avoid collision with zero-vector singleton IDs.
    labels[order[nonzero]] = clusters + len(texts)
    return labels


def half_batch_indices(ids):
    """Deterministic ID-only deletion, independent of labels and true prompts."""
    keys = [hashlib.sha256(str(value).encode()).hexdigest() for value in ids]
    return np.sort(np.argsort(keys, kind="stable")[:(len(ids) + 1) // 2])


def feature_pair(train_texts, valid_texts, config):
    train_blocks, valid_blocks = [], []
    for analyzer, ngram, weight, limit in (
            ("char_wb", (2, 5), 1.0, config["char_max_features"]),
            ("word", (1, 2), config["word_weight"], config["word_max_features"])):
        opts = dict(analyzer=analyzer, ngram_range=ngram, lowercase=True, min_df=2,
                    sublinear_tf=True, max_features=limit, dtype=np.float64)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**opts)
        train_blocks.append(vectorizer.fit_transform(train_texts) * weight)
        valid_blocks.append(vectorizer.transform(valid_texts) * weight)
    return tuple(normalize(sparse.hstack(blocks, format="csr"))
                 for blocks in (train_blocks, valid_blocks))


def subset_data(data, indices):
    return {key: value if key == "labels" else value[indices] for key, value in data.items()}


def sources():
    # Include all imported local numerical/identity helpers transitively.
    paths = {**previous_sources(), str(Path(__file__).relative_to(REPO)): file_hash(__file__),
             str(CONFIG.relative_to(REPO)): file_hash(CONFIG)}
    plan = MODULE / "docs/PROMPT_CENTERING_PLAN.md"
    paths[str(plan.relative_to(REPO))] = file_hash(plan)
    return paths


def gate(result, config):
    return bool(set(result["stress"]) == {"half", "singleton"}
        and result["delta_vs_original"] >= config["adoption_min_delta"]
        and result["paired_prompt_ci95"][0] > 1e-12
        and result["positive_folds"] >= config["adoption_min_positive_folds"]
        and all(s["delta_vs_original"] >= config["stress_min_delta"]
                for s in result["stress"].values()))


def run(data, folds, fold_id, baseline, config, contract):
    contract_hash = object_hash(contract)
    claim = claim_run(CLAIM, contract_hash)
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError("Predeclared wall-clock budget exhausted")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, config["max_seconds"])
    try:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        write_json(OUTPUT / "contract.json", contract)
        write_json(OUTPUT / "config.json", config)
        for relative in contract["source_sha256"]:
            destination = OUTPUT / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / relative).read_bytes())
        write_json(OUTPUT / "fold_manifest.json", {"row_index": data["row_index"].tolist(),
            "ids": data["ids"].tolist(), "groups": data["groups"].tolist(), "fold_id": fold_id.tolist()})
        shape = (len(data["truth"]), len(data["labels"]))
        probs = {spec["name"]: {mode: np.full(shape, np.nan)
                    for mode in ("batch", "half", "singleton")} for spec in config["models"]}
        half_mask = np.zeros(shape[0], dtype=bool)
        records, cluster_records = [], []
        with threadpool_limits(limits=config["threads"]):
            for fold, (tr, va) in enumerate(folds):
                a, b = feature_pair(data["texts"][tr], data["texts"][va], config)
                half = half_batch_indices(data["ids"][va])
                half_mask[va[half]] = True
                cluster_args = (config["cluster_distance_threshold"], config["cluster_max_features"])
                batch_groups = cluster_batch(data["texts"][va], data["ids"][va], *cluster_args)
                half_groups = cluster_batch(data["texts"][va[half]], data["ids"][va[half]], *cluster_args)
                for spec in config["models"]:
                    strength, name = spec["strength"], spec["name"]
                    if peak_rss() > config["max_rss_bytes"]:
                        raise MemoryError("Observed RSS budget exceeded")
                    train_x = soft_center(a, data["groups"][tr], strength, config["max_centered_csr_bytes"])
                    if claim["attempted_fits"] >= config["max_fits"]:
                        raise RuntimeError("Fit budget exhausted")
                    claim["attempted_fits"] += 1
                    claim["current"] = {"model": name, "fold": fold}
                    write_json(CLAIM, claim)
                    fit_start = time.monotonic()
                    classifier = LogisticRegression(C=config["C"], class_weight="balanced",
                                     max_iter=config["max_iter"], random_state=config["seed"])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        classifier.fit(train_x, data["truth"][tr])
                    claim["completed_fits"] += 1
                    write_json(CLAIM, claim)
                    if not np.array_equal(classifier.classes_, np.arange(shape[1])):
                        raise ValueError("Class order changed")
                    convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                    if convergence:
                        raise RuntimeError("Convergence warning; finite plan stops without retuning")
                    del train_x
                    for mode, matrix, groups, target in (
                            ("batch", b, batch_groups, va),
                            ("half", b[half], half_groups, va[half]),
                            ("singleton", b, np.arange(len(va)), va)):
                        transformed = soft_center(matrix, groups, strength, config["max_centered_csr_bytes"])
                        probs[name][mode][target] = canonical_probabilities(
                            classifier.predict_proba(transformed), len(target), shape[1])
                        del transformed
                    record = {"model": name, "fold": fold, "seconds": time.monotonic() - fit_start,
                              "convergence_warnings": convergence, "peak_rss_bytes": peak_rss()}
                    records.append(record)
                    write_json(OUTPUT / "fit_progress.json", records)
                    print(json.dumps(record), flush=True)
                    del classifier
                    gc.collect()
                    if peak_rss() > config["max_rss_bytes"]:
                        raise MemoryError("Observed RSS budget exceeded")
                # Diagnostic only, evaluated after all predictions for this fold.
                cluster_records.append({"fold": fold, "batch_clusters": len(np.unique(batch_groups)),
                    "batch_sizes": sorted(np.unique(batch_groups, return_counts=True)[1].tolist()),
                    "batch_ari": float(adjusted_rand_score(data["groups"][va], batch_groups)),
                    "half_clusters": len(np.unique(half_groups)),
                    "half_sizes": sorted(np.unique(half_groups, return_counts=True)[1].tolist()),
                    "half_ari": float(adjusted_rand_score(data["groups"][va[half]], half_groups))})
                del a, b
                gc.collect()
            half_indices = np.flatnonzero(half_mask)
            half_data = subset_data(data, half_indices)
            artifacts = {}
            for name in probs:
                artifacts[name] = {}
                for mode in probs[name]:
                    indices = half_indices if mode == "half" else np.arange(shape[0])
                    selected = subset_data(data, indices)
                    path = OUTPUT / f"{name}_{mode}_oof.npz"
                    save_oof(path, probs[name][mode][indices], selected, fold_id[indices], contract_hash)
                    loaded = load_oof(path, selected, fold_id[indices], contract_hash)
                    probs[name][mode][indices] = loaded
                    artifacts[name][mode] = file_hash(path)
            candidates = dict(probs)
            for spec in config["models"]:
                for weight in config["blend_weights"]:
                    name = f"blend_{spec['name']}_{weight:g}"
                    candidates[name] = {mode: (1 - weight) * baseline + weight * values
                                        for mode, values in probs[spec["name"]].items()}
            base_metrics = metrics(baseline, data, fold_id)
            half_base = metrics(baseline[half_indices], half_data, fold_id[half_indices])
            results = {}
            for name, modes in candidates.items():
                result = metrics(modes["batch"], data, fold_id)
                result["delta_vs_original"] = result["balanced_accuracy"] - base_metrics["balanced_accuracy"]
                result["paired_prompt_ci95"] = paired_interval(baseline, modes["batch"], data, config)
                result["positive_folds"] = positive_fold_count(base_metrics["fold_balanced_accuracy"], result["fold_balanced_accuracy"])
                plain_name = "plain" if not name.startswith("blend_") else f"blend_plain_{name.rsplit('_', 1)[1]}"
                control = candidates[plain_name]["batch"]
                control_metrics = metrics(control, data, fold_id)
                result["matched_plain_delta"] = result["balanced_accuracy"] - control_metrics["balanced_accuracy"]
                result["matched_plain_ci95"] = paired_interval(control, modes["batch"], data, config)
                result["stress"] = {}
                for mode, indices, selected, base_result in (
                        ("half", half_indices, half_data, half_base),
                        ("singleton", np.arange(shape[0]), data, base_metrics)):
                    stress = metrics(modes[mode][indices], selected, fold_id[indices])
                    stress["rows"] = len(indices)
                    stress["delta_vs_original"] = stress["balanced_accuracy"] - base_result["balanced_accuracy"]
                    result["stress"][mode] = stress
                result["passes_development_gate"] = gate(result, config)
                results[name] = result
        if sources() != contract["source_sha256"] or file_hash(BASELINE) != config["baseline_oof_sha256"]:
            raise RuntimeError("Source or baseline changed during experiment")
        summary = {"protocol": config["protocol"], "scope": "adaptive transductive development only",
            "contract_sha256": contract_hash, "attempted_fits": claim["attempted_fits"],
            "completed_fits": claim["completed_fits"], "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss(), "final_holdout_evaluations": 0,
            "baseline": base_metrics, "baseline_half": half_base,
            "results": results, "oof_sha256": artifacts, "clustering_diagnostics": cluster_records,
            "engineering_stop": "predeclared finite experiment completed; frozen delivery retained"}
        write_json(OUTPUT / "summary.json", summary)
        claim.update(state="completed", summary_sha256=file_hash(OUTPUT / "summary.json"), wall_seconds=summary["wall_seconds"])
        write_json(CLAIM, claim)
        print(json.dumps({"completed_fits": claim["completed_fits"], "wall_seconds": summary["wall_seconds"],
            "results": {name: {"BA": v["balanced_accuracy"], "gate": v["passes_development_gate"]} for name, v in results.items()}}), flush=True)
    except BaseException as error:
        claim.update(state="failed", error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
        write_json(CLAIM, claim)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    data = load_development(args.train, args.split, MODULE / "config/labels.json", config)
    folds, fold_id = grouped_folds(data, config["folds"])
    if file_hash(BASELINE) != config["baseline_oof_sha256"]:
        raise ValueError("Prior baseline OOF hash mismatch")
    baseline = load_oof(BASELINE, data, fold_id, config["baseline_contract_sha256"])
    contract = {"protocol": config["protocol"], "config_sha256": file_hash(CONFIG),
        "train_sha256": config["train_sha256"], "split_sha256": config["split_sha256"],
        "labels_sha256": config["labels_sha256"], "baseline_oof_sha256": file_hash(BASELINE),
        "baseline_contract_sha256": config["baseline_contract_sha256"],
        "ordered_ids_sha256": object_hash(data["ids"].tolist()), "fold_id_sha256": object_hash(fold_id.tolist()),
        "labels": data["labels"], "source_sha256": sources(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "pandas", "joblib", "threadpoolctl")},
        "final_holdout_used_previously": True, "final_holdout_evaluations_this_run": 0}
    print(json.dumps({"audit": "passed", "contract_sha256": object_hash(contract), "claim_exists": CLAIM.exists(),
                      "development_rows": len(data["truth"])}), flush=True)
    if args.run:
        run(data, folds, fold_id, baseline, config, contract)


if __name__ == "__main__":
    main()
