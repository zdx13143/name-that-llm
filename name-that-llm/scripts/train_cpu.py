#!/usr/bin/env python3
"""Bounded prompt-grouped CPU search; final holdout is opened only by finalize.

All row-level artifacts, features, and weights are under ignored outputs/.
Search fits preprocessing inside each training fold. Cached features are tied to
the input hash, split seed, fold, and feature configuration. No test labels exist.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

from model_cpu import (checked_submission, config_id, feature_config, file_hash,
                       make_classifier, make_features, make_pipeline, read_csv)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261018
STABILITY_SEEDS = [20261019, 20261020]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def grouped_folds(groups, dev, seed):
    unique = np.unique(groups[dev])
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    folds = []
    for selected in np.array_split(unique, 5):
        valid = dev[np.isin(groups[dev], selected)]
        train = dev[~np.isin(groups[dev], selected)]
        assert not set(groups[train]) & set(groups[valid])
        folds.append((train, valid))
    assert sorted(np.concatenate([valid for _, valid in folds]).tolist()) == sorted(dev.tolist())
    return folds


def freeze_split(rows, labels, destination, source_hash):
    groups = np.array([row["prompt_id"] for row in rows])
    y = np.array([labels.index(row["model"]) for row in rows])
    unique = np.unique(groups)
    if len(unique) < 10:
        raise ValueError("At least ten prompt groups are required")
    # The audit found no duplicates. Fail closed if later data introduces them.
    text_groups = {}
    for row in rows:
        normalized = " ".join(row["text"].split())
        if normalized in text_groups and text_groups[normalized] != row["prompt_id"]:
            raise ValueError("Cross-prompt duplicate text requires a newly reviewed grouping policy")
        text_groups[normalized] = row["prompt_id"]
    destination = Path(destination)
    if destination.exists():
        manifest = json.loads(destination.read_text())
        if manifest["train_sha256"] != source_hash or manifest["seed"] != SEED:
            raise ValueError("Frozen split does not match input data or seed")
        dev = np.array(manifest["development_indices"], dtype=int)
        heldout = np.array(manifest["holdout_indices"], dtype=int)
    else:
        shuffled = np.random.default_rng(SEED).permutation(unique)
        holdout_groups = shuffled[:max(1, int(np.ceil(len(unique) * 0.2))) ]
        heldout = np.flatnonzero(np.isin(groups, holdout_groups))
        dev = np.flatnonzero(~np.isin(groups, holdout_groups))
        folds = grouped_folds(groups, dev, SEED)
        fold_of = {int(index): fold for fold, (_, valid) in enumerate(folds) for index in valid}
        manifest = {"seed": SEED, "stability_seeds": STABILITY_SEEDS,
                    "train_sha256": source_hash, "development_indices": dev.tolist(),
                    "holdout_indices": heldout.tolist(),
                    "rows": [{"row_index": index, "id": row["id"], "prompt_id": row["prompt_id"],
                              "role": "development" if index in fold_of else "holdout",
                              "development_fold": fold_of.get(index)} for index, row in enumerate(rows)]}
        save_json(destination, manifest)
    assert not set(groups[dev]) & set(groups[heldout])
    assert set(dev) | set(heldout) == set(range(len(rows)))
    assert not set(dev) & set(heldout)
    for train, valid in grouped_folds(groups, dev, SEED):
        if len(set(y[train])) != len(labels) or len(set(y[valid])) != len(labels):
            raise ValueError("A development fold lacks a class")
    if len(set(y[heldout])) != len(labels):
        raise ValueError("Final holdout lacks a class")
    return dev, heldout, groups, y


def paired_group_bootstrap(y, old, new, groups, draws=2000):
    unique = np.unique(groups)
    classes = np.unique(y)
    totals = np.zeros((len(unique), len(classes)))
    delta = np.zeros_like(totals)
    for index, group in enumerate(unique):
        selection = groups == group
        for column, label in enumerate(classes):
            mask = selection & (y == label)
            totals[index, column] = mask.sum()
            delta[index, column] = ((new[mask] == y[mask]).sum()
                                    - (old[mask] == y[mask]).sum())
    draws_weights = np.random.default_rng(SEED + 100).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=draws)
    draws_delta = ((draws_weights @ delta) / (draws_weights @ totals)).mean(axis=1)
    return {"delta_balanced_accuracy": float(balanced_accuracy_score(y, new) - balanced_accuracy_score(y, old)),
            "ci95_low": float(np.quantile(draws_delta, .025)),
            "ci95_high": float(np.quantile(draws_delta, .975)), "group_bootstrap_draws": draws,
            "groups": len(unique), "adaptive_comparison_warning": True}


def confirmed_improvement(comparison, fold_delta):
    """Treat floating point residues at zero as a CI touching zero."""
    fold_delta = np.asarray(fold_delta)
    return bool(comparison["delta_balanced_accuracy"] >= .005
                and comparison["ci95_low"] > 1e-12
                and (fold_delta >= 0).sum() >= 3 and fold_delta.min() > -.05)


class Search:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.labels = json.loads((ROOT / "config" / "labels.json").read_text())["labels"]
        self.rows = read_csv(args.data_dir / "train.csv")
        self.source_hash = file_hash(args.data_dir / "train.csv")
        from artifact_guards import assert_holdout_available
        assert_holdout_available(self.source_hash)
        self.dev, self.heldout, self.groups, self.y = freeze_split(
            self.rows, self.labels, self.output / "split_manifest.json", self.source_hash)
        self.split_hash = file_hash(self.output / "split_manifest.json")
        self.label_hash = file_hash(ROOT / "config" / "labels.json")
        self.code_hashes = {name: file_hash(ROOT / "scripts" / name)
                            for name in ("model_cpu.py", "train_cpu.py")}
        self.texts = np.array([row["text"] for row in self.rows])
        self.started = time.monotonic()
        self.fit_count = 0
        self.stability_fit_count = 0
        self.results = []
        self.batches = []
        self.predictions = {}

    def evaluate(self, config, batch, seed=SEED, stability=False):
        if file_hash(self.output / "split_manifest.json") != self.split_hash:
            raise ValueError("Frozen split changed during the search")
        if any(file_hash(ROOT / "scripts" / name) != digest for name, digest in self.code_hashes.items()):
            raise ValueError("Experiment code changed during the search")
        if time.monotonic() - self.started >= self.args.hours * 3600:
            raise TimeoutError("CPU wall-clock search budget reached")
        if (self.fit_count >= 100 and not stability) or (self.stability_fit_count >= 20 and stability):
            raise TimeoutError("CV fit budget reached")
        identity = config_id(config)
        result_path = self.output / "trials" / f"{identity}-{seed}.json"
        if result_path.exists():
            raise ValueError("A trial already exists; do not silently reset search budgets")
        start = time.monotonic()
        prediction = np.full(len(self.rows), -1, dtype=int)
        fold_scores, fold_times, convergence = [], [], []
        for fold, (train, valid) in enumerate(grouped_folds(self.groups, self.dev, seed)):
            if time.monotonic() - self.started >= self.args.hours * 3600:
                raise TimeoutError("CPU wall-clock search budget reached")
            begin = time.monotonic()
            cache_key = (f"{self.source_hash[:12]}-{self.split_hash[:12]}-"
                         f"{self.code_hashes['model_cpu.py'][:12]}-"
                         f"{config_id(feature_config(config))}-{seed}-{fold}")
            cache_file = self.output / "feature_cache" / f"{cache_key}.joblib"
            if cache_file.exists():
                x_train, x_valid = joblib.load(cache_file)
            else:
                features = make_features(config)
                x_train = features.fit_transform(self.texts[train])
                x_valid = features.transform(self.texts[valid])
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump((x_train, x_valid), cache_file)
                del features
            model = make_classifier(config)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train, self.y[train])
            if stability:
                self.stability_fit_count += 1
            else:
                self.fit_count += 1
            prediction[valid] = model.predict(x_valid)
            convergence.append(sum(isinstance(item.message, ConvergenceWarning) for item in recorded))
            fold_scores.append(float(balanced_accuracy_score(self.y[valid], prediction[valid])))
            fold_times.append(time.monotonic() - begin)
            del x_train, x_valid, model
        assert (prediction[self.dev] >= 0).all()
        result = {"id": identity, "config": config, "batch": batch, "seed": seed,
                  "balanced_accuracy": float(balanced_accuracy_score(self.y[self.dev], prediction[self.dev])),
                  "fold_balanced_accuracy": fold_scores, "fold_mean": float(np.mean(fold_scores)),
                  "fold_std": float(np.std(fold_scores)), "fold_seconds": fold_times,
                  "seconds": time.monotonic() - start, "convergence_warnings": convergence,
                  "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "fit_count": 5, "feature_preprocessing": "fit on training fold only"}
        np.savez_compressed(self.output / "trials" / f"{identity}-{seed}-oof.npz",
                            row_index=self.dev, truth=self.y[self.dev], prediction=prediction[self.dev])
        save_json(result_path, result)
        self.predictions[(identity, seed)] = prediction[self.dev]
        if not stability:
            self.results.append(result)
        print(json.dumps({"trial": len(self.results), "id": identity, "batch": batch,
                          "seed": seed, "balanced_accuracy": result["balanced_accuracy"],
                          "seconds": round(result["seconds"], 2), "config": config}), flush=True)
        self.checkpoint("running")
        return result

    def checkpoint(self, reason, extra=None):
        payload = {"status": reason, "train_sha256": self.source_hash,
                   "split_sha256": self.split_hash,
                   "label_reference_sha256": self.label_hash,
                   "seed": SEED, "stability_seeds": STABILITY_SEEDS,
                   "development_rows": len(self.dev), "development_groups": len(set(self.groups[self.dev])),
                   "holdout_rows": len(self.heldout), "holdout_groups": len(set(self.groups[self.heldout])),
                   "holdout_opened": False, "configurations": len(self.results),
                   "main_cv_fits": self.fit_count, "stability_cv_fits": self.stability_fit_count,
                   "wall_seconds": time.monotonic() - self.started, "results": self.results,
                   "batches": self.batches, "environment": {"python": platform.python_version(),
                   "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "threads": 2,
                   **{name: importlib.metadata.version(name) for name in ("scipy", "pyarrow", "joblib")}},
                   "code_sha256": self.code_hashes,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.strip(),
                   "code_was_uncommitted_at_run": True,
                   "evidence_scope": "Real local grouped validation; no Kaggle execution, submission or leaderboard score."}
        if extra:
            payload.update(extra)
        save_json(self.output / "search_summary.json", payload)
        return payload

    def run_batch(self, name, configs):
        incumbent = max(self.results, key=lambda row: row["balanced_accuracy"]) if self.results else None
        current = [self.evaluate(config, name) for config in configs]
        best = max(current, key=lambda row: row["balanced_accuracy"])
        batch = {"name": name, "configurations": len(current), "best_id": best["id"]}
        if incumbent:
            comparison = paired_group_bootstrap(self.y[self.dev],
                self.predictions[(incumbent["id"], SEED)], self.predictions[(best["id"], SEED)], self.groups[self.dev])
            fold_delta = np.array(best["fold_balanced_accuracy"]) - incumbent["fold_balanced_accuracy"]
            comparison["fold_deltas"] = fold_delta.tolist()
            comparison["confirmed_improvement"] = confirmed_improvement(comparison, fold_delta)
            comparison["incumbent_id"] = incumbent["id"]
            batch["comparison"] = comparison
        self.batches.append(batch)
        self.checkpoint("running")
        return current

    def run(self):
        if (self.output / "search_summary.json").exists():
            raise ValueError("Search output already exists; preserve budgets and evidence instead of restarting")
        (self.output / "trials").mkdir(exist_ok=True)
        base = {"classifier": "svm", "C": 1.0, "class_weight": None,
                "lowercase": False, "sublinear_tf": True, "min_df": 2,
                "char_ngram": [3, 5], "word_ngram": [1, 2]}
        baseline = self.run_batch("1_representations", [{**base, "representation": name}
                                  for name in ("char", "word", "combined")])
        top_representations = sorted(baseline, key=lambda row: row["balanced_accuracy"], reverse=True)[:2]
        configurations = []
        for item in top_representations:
            for c, weight in ((.25, None), (4., None), (.25, "balanced"), (1., "balanced"), (4., "balanced")):
                configurations.append({**item["config"], "C": c, "class_weight": weight})
        self.run_batch("2_regularization_and_class_weights", configurations)
        best = max(self.results, key=lambda row: row["balanced_accuracy"])["config"]
        expanded = {"char_ngram": [2, 5]} if best["representation"] != "word" else {"word_ngram": [1, 3]}
        self.run_batch("3_feature_ablations", [{**best, **change} for change in
            ({"lowercase": True}, {"min_df": 1}, {"sublinear_tf": False}, expanded)])
        best = max(self.results, key=lambda row: row["balanced_accuracy"])["config"]
        self.run_batch("4_logistic_regression", [{**best, "classifier": "logistic", "C": c}
                       for c in (.25, 1., 4.)])
        finalists = sorted(self.results, key=lambda row: row["balanced_accuracy"], reverse=True)[:2]
        stability = []
        for seed in STABILITY_SEEDS:
            seed_results = [self.evaluate(row["config"], "stability", seed, True) for row in finalists]
            comparison = paired_group_bootstrap(self.y[self.dev],
                self.predictions[(finalists[1]["id"], seed)],
                self.predictions[(finalists[0]["id"], seed)], self.groups[self.dev])
            stability.append({"seed": seed, "results": seed_results, "comparison_first_vs_second": comparison})
        recent = self.batches[-3:]
        no_confirmed = all(not batch["comparison"]["confirmed_improvement"] for batch in recent)
        small_supported = all(batch["comparison"]["ci95_high"] < .005 for batch in recent)
        stability_consistent = all(item["comparison_first_vs_second"]["delta_balanced_accuracy"] >= 0
                                   for item in stability)
        all_converged = all(not any(row["convergence_warnings"]) for row in self.results)
        baseline_best = max(baseline, key=lambda row: row["balanced_accuracy"])
        total_gain = paired_group_bootstrap(self.y[self.dev], self.predictions[(baseline_best["id"], SEED)],
                    self.predictions[(finalists[0]["id"], SEED)], self.groups[self.dev])
        reason = "three_batches_without_confirmed_improvement" if no_confirmed else "configuration_budget_boundary"
        if not all_converged:
            reason = "incomplete_direction_due_to_convergence"
        return self.checkpoint(reason, {"selected": finalists[0], "stability": stability,
                "baseline_best": baseline_best, "selected_vs_baseline": total_gain,
                "small_gain_supported_in_tried_batches": small_supported and all_converged,
                "selected_rank_consistent_across_stability_seeds": stability_consistent,
                "all_main_trials_converged": all_converged,
                "stability_interpretation": "Rank reversals or convergence warnings prevent a claim of robust improvement or a demonstrated plateau.",
                "stop_interpretation": "Engineering stopping rule; not proof of global optimum. Wide intervals cannot establish a plateau."})


def finalize(args):
    """All legacy callers share the durable cross-round holdout claim."""
    from finalize_cpu import finalize_once
    return finalize_once(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["search", "finalize"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cpu_round1")
    parser.add_argument("--hours", type=float, default=2.)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.action == "search":
            runner = Search(args)
            try:
                runner.run()
            except TimeoutError as exc:
                runner.checkpoint("budget_boundary", {"reason": str(exc)})
                raise
        else:
            finalize(args)


if __name__ == "__main__":
    main()
