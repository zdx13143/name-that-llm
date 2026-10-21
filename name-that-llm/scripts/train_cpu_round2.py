#!/usr/bin/env python3
"""Shared-holdout second CPU round; cumulative fit counts include round one."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

import train_cpu as previous
import model_cpu_round2 as extended
from model_cpu import config_id, file_hash, make_classifier
from artifact_guards import validate_prior

ROOT = Path(__file__).resolve().parents[1]
SEED = previous.SEED


class RoundTwo(previous.Search):
    def __init__(self, args):
        self.prior_dir = args.prior
        checked_prior = validate_prior(args.prior, args.data_dir, ROOT / "config" / "labels.json")
        self.prior = checked_prior["summary"]
        if (args.prior / "holdout_opened.json").exists():
            raise ValueError("Do not tune a second round after opening the final holdout")
        source_manifest = args.prior / "split_manifest.json"
        if file_hash(source_manifest) != self.prior["split_sha256"]:
            raise ValueError("Original split manifest changed")
        args.output.mkdir(parents=True, exist_ok=True)
        target_manifest = args.output / "split_manifest.json"
        if not target_manifest.exists():
            shutil.copy2(source_manifest, target_manifest)
        if file_hash(target_manifest) != file_hash(source_manifest):
            raise ValueError("Round two must share the exact original split manifest")
        super().__init__(args)
        self.code_hashes.update({name: file_hash(ROOT / "scripts" / name)
                                for name in ("model_cpu_round2.py", "train_cpu_round2.py")})
        self.incumbent = {**self.prior["selected"], "from_prior_round": True}
        self.results = [self.incumbent]
        oof = checked_prior["oof"]
        if not np.array_equal(oof["row_index"], self.dev):
            raise ValueError("Prior OOF row order does not match shared development rows")
        self.predictions[(self.incumbent["id"], SEED)] = oof["prediction"]
        self.word_config = next(row["config"] for row in self.prior["results"]
                                if row["batch"] == "1_representations" and row["config"]["representation"] == "word")

    def checkpoint(self, reason, extra=None):
        report = super().checkpoint(reason, extra)
        report["configurations"] = len(self.results) - 1
        report["cumulative_configurations"] = self.prior["configurations"] + len(self.results) - 1
        report["cumulative_main_cv_fits"] = self.prior["main_cv_fits"] + self.fit_count
        report["cumulative_stability_cv_fits"] = self.prior["stability_cv_fits"] + self.stability_fit_count
        report["cumulative_search_wall_seconds"] = self.prior["wall_seconds"] + report["wall_seconds"]
        report["prior_search_summary_sha256"] = file_hash(self.prior_dir / "search_summary.json")
        report["prior_numerical_review"] = "Round-one feature-batch CI lower bound was 3.8e-19, numerically zero; its raw confirmed flag is not accepted as robust positive evidence. Exact original code is retained in round-one code_snapshot."
        previous.save_json(self.output / "search_summary.json", report)
        return report

    def compare_batch(self, name, incumbent, results):
        best = max(results, key=lambda row: row["balanced_accuracy"])
        comparison = previous.paired_group_bootstrap(self.y[self.dev],
            self.predictions[(incumbent["id"], SEED)], self.predictions[(best["id"], SEED)], self.groups[self.dev])
        delta = np.array(best["fold_balanced_accuracy"]) - incumbent["fold_balanced_accuracy"]
        comparison.update({"fold_deltas": delta.tolist(), "incumbent_id": incumbent["id"],
                           "confirmed_improvement": previous.confirmed_improvement(comparison, delta)})
        self.batches.append({"name": name, "configurations": len(results), "best_id": best["id"], "comparison": comparison})
        self.checkpoint("running")

    def fusion(self, weights, seed=SEED, stability=False):
        if any(file_hash(ROOT / "scripts" / name) != digest for name, digest in self.code_hashes.items()):
            raise ValueError("Experiment code changed during fusion")
        if file_hash(self.output / "split_manifest.json") != self.split_hash:
            raise ValueError("Frozen split changed during fusion")
        if time.monotonic() - self.started > self.args.hours * 3600:
            raise TimeoutError("Round-two CPU budget reached")
        char_config = self.incumbent["config"]
        configs = [{"classifier": "score_fusion", "char_config": char_config,
                    "word_config": self.word_config, "char_weight": weight} for weight in weights]
        predictions = [np.full(len(self.rows), -1, dtype=int) for _ in weights]
        char_prediction = np.full(len(self.rows), -1, dtype=int)
        fold_scores = [[] for _ in weights]
        component_fold_scores, fold_times, warning_counts = [], [], []
        start = time.monotonic()
        score_char, score_word = np.zeros((len(self.rows), len(self.labels))), np.zeros((len(self.rows), len(self.labels)))
        for train, valid in previous.grouped_folds(self.groups, self.dev, seed):
            if time.monotonic() - self.started > self.args.hours * 3600:
                raise TimeoutError("Round-two CPU budget reached")
            begin = time.monotonic()
            scores = []
            fold_warnings = 0
            for config in (char_config, self.word_config):
                pipeline = extended.make_pipeline(config)
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always", ConvergenceWarning)
                    pipeline.fit(self.texts[train], self.y[train])
                if stability:
                    self.stability_fit_count += 1
                else:
                    self.fit_count += 1
                fold_warnings += sum(isinstance(item.message, ConvergenceWarning) for item in recorded)
                scores.append(extended.normalize_scores(pipeline.decision_function(self.texts[valid])))
                del pipeline
            score_char[valid], score_word[valid] = scores
            char_prediction[valid] = scores[0].argmax(axis=1)
            component_fold_scores.append(float(balanced_accuracy_score(self.y[valid], char_prediction[valid])))
            for index, weight in enumerate(weights):
                predictions[index][valid] = (weight * scores[0] + (1 - weight) * scores[1]).argmax(axis=1)
                fold_scores[index].append(float(balanced_accuracy_score(self.y[valid], predictions[index][valid])))
            fold_times.append(time.monotonic() - begin)
            warning_counts.append(fold_warnings)
        total = time.monotonic() - start
        np.savez_compressed(self.output / "trials" / f"fusion-component-scores-{seed}.npz",
                            row_index=self.dev, char=score_char[self.dev], word=score_word[self.dev])
        results = []
        for index, config in enumerate(configs):
            identity = config_id(config)
            result = {"id": identity, "config": config, "batch": "3_independent_score_fusion",
                      "seed": seed, "balanced_accuracy": float(balanced_accuracy_score(self.y[self.dev], predictions[index][self.dev])),
                      "fold_balanced_accuracy": fold_scores[index], "fold_mean": float(np.mean(fold_scores[index])),
                      "fold_std": float(np.std(fold_scores[index])), "fold_seconds": fold_times,
                      "seconds": total, "fit_count": 10 if index == 0 else 0,
                      "shared_fit_group": f"fusion-{seed}", "convergence_warnings": warning_counts}
            previous.save_json(self.output / "trials" / f"{identity}-{seed}.json", result)
            np.savez_compressed(self.output / "trials" / f"{identity}-{seed}-oof.npz",
                                row_index=self.dev, truth=self.y[self.dev], prediction=predictions[index][self.dev])
            self.predictions[(identity, seed)] = predictions[index][self.dev]
            if not stability:
                self.results.append(result)
            results.append(result)
            print(json.dumps({"round": 2, "config": config, "seed": seed,
                              "balanced_accuracy": result["balanced_accuracy"], "shared_fusion_seconds": total}), flush=True)
        self.predictions[(self.incumbent["id"], seed)] = char_prediction[self.dev]
        component = {"id": self.incumbent["id"], "config": char_config, "seed": seed,
                     "balanced_accuracy": float(balanced_accuracy_score(self.y[self.dev], char_prediction[self.dev])),
                     "fold_balanced_accuracy": component_fold_scores,
                     "fit_count": 0, "fit_reuse": "Already counted inside fusion components"}
        self.checkpoint("running")
        return results, component

    def run(self):
        if (self.output / "search_summary.json").exists():
            raise ValueError("Round-two search already exists; do not reset budgets")
        (self.output / "trials").mkdir(exist_ok=True)
        base = self.incumbent["config"]
        self.run_batch("1_stylometry", [{**base, "style_weight": weight} for weight in (.25, 1., 4.)])
        ranges = [("char", [1, 3]), ("char", [2, 3]), ("char", [2, 4]),
                  ("char_wb", [2, 4]), ("char_wb", [3, 5]), ("char_wb", [4, 6])]
        self.run_batch("2_short_and_boundary_ngrams", [{**base, "char_analyzer": analyzer, "char_ngram": interval}
                                                      for analyzer, interval in ranges])
        incumbent = max(self.results, key=lambda row: row["balanced_accuracy"])
        fusion_results, _ = self.fusion([.25, .5, .75])
        self.compare_batch("3_independent_score_fusion", incumbent, fusion_results)
        best_new = max(self.results[1:], key=lambda row: row["balanced_accuracy"])
        selected = max(self.results, key=lambda row: row["balanced_accuracy"])
        stability = []
        for seed in previous.STABILITY_SEEDS:
            if best_new["config"].get("classifier") == "score_fusion":
                new_results, old_result = self.fusion([best_new["config"]["char_weight"]], seed, True)
                new_result = new_results[0]
            else:
                new_result = self.evaluate(best_new["config"], "stability", seed, True)
                old_result = self.evaluate(self.incumbent["config"], "stability_incumbent", seed, True)
            comparison = previous.paired_group_bootstrap(self.y[self.dev],
                self.predictions[(self.incumbent["id"], seed)], self.predictions[(best_new["id"], seed)], self.groups[self.dev])
            stability.append({"seed": seed, "candidate": new_result, "incumbent": old_result, "comparison": comparison})
        converged = all(not any(row["convergence_warnings"]) for row in self.results)
        no_confirmed = all(not row["comparison"]["confirmed_improvement"] for row in self.batches)
        total_gain = previous.paired_group_bootstrap(self.y[self.dev],
            self.predictions[(self.incumbent["id"], SEED)], self.predictions[(selected["id"], SEED)], self.groups[self.dev])
        reason = "three_batches_without_confirmed_improvement" if no_confirmed else "configuration_budget_boundary"
        if not converged:
            reason = "incomplete_direction_due_to_convergence"
        return self.checkpoint(reason, {"selected": selected, "best_new": best_new, "stability": stability,
                "selected_vs_prior_incumbent": total_gain,
                "all_main_trials_converged": converged,
                "candidate_rank_consistent_across_stability_seeds": all(item["comparison"]["delta_balanced_accuracy"] >= 0 for item in stability),
                "small_gain_supported_in_tried_batches": converged and all(row["comparison"]["ci95_high"] < .005 for row in self.batches),
                "stop_interpretation": "Engineering stopping evidence only; wide intervals cannot establish a plateau."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["search", "finalize"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--prior", type=Path, default=ROOT / "outputs" / "cpu_round1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cpu_round2")
    parser.add_argument("--hours", type=float, default=.5)
    args = parser.parse_args()
    previous.make_features = extended.make_features
    previous.make_pipeline = extended.make_pipeline
    original_feature_config = previous.feature_config
    previous.feature_config = lambda config: {**original_feature_config(config),
        "round2_implementation_sha256": file_hash(ROOT / "scripts" / "model_cpu_round2.py")}
    with threadpool_limits(limits=2):
        if args.action == "search":
            runner = RoundTwo(args)
            try:
                runner.run()
            except TimeoutError as exc:
                runner.checkpoint("budget_boundary", {"reason": str(exc)})
                raise
        else:
            previous.finalize(args)


if __name__ == "__main__":
    main()
