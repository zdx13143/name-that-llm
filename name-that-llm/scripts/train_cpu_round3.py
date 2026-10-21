#!/usr/bin/env python3
"""Third predeclared CPU search: style weights, regularization, short n-grams.

Reuses the original prompt holdout and existing OOF artifacts without fitting
them again. All counts are cumulative; finalization is a separate action.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from threadpoolctl import threadpool_limits

import train_cpu as base
import model_cpu_round2 as extended
from model_cpu import config_id, file_hash
from artifact_guards import object_hash, validate_oof, validate_prior
from finalize_cpu import verify_frozen_code, verify_runtime

ROOT = Path(__file__).resolve().parents[1]
SEED = base.SEED


class RoundThree(base.Search):
    def __init__(self, args):
        if (args.output / "search_summary.json").exists():
            raise ValueError("Round-three output already exists; preserve the existing budget and evidence")
        self.prior_dir = args.prior
        checked = validate_prior(args.prior, args.data_dir, ROOT / "config" / "labels.json")
        self.prior = checked["summary"]
        verify_frozen_code(self.prior, args.prior)
        verify_runtime(self.prior)
        self.prior_contract_hash = checked["contract_sha256"]
        args.output.mkdir(parents=True, exist_ok=True)
        source = args.prior / "split_manifest.json"
        target = args.output / "split_manifest.json"
        if target.exists() and file_hash(target) != file_hash(source):
            raise ValueError("Round three must use the unchanged original split")
        if not target.exists():
            shutil.copy2(source, target)
        super().__init__(args)
        self.code_hashes.update({name: file_hash(ROOT / "scripts" / name) for name in
            ("model_cpu_round2.py", "train_cpu_round3.py", "artifact_guards.py", "finalize_cpu.py")})
        self.incumbent = {**self.prior["selected"], "from_prior_round": True}
        self.results = [self.incumbent]
        self.predictions[(self.incumbent["id"], SEED)] = checked["oof"]["prediction"]
        (self.output / "trials").mkdir(exist_ok=True)
        # The selected result may remain the prior incumbent. Keep its exact
        # OOF available here so finalization verifies it in this run directory.
        source_oof = args.prior / "trials" / f"{self.incumbent['id']}-{SEED}-oof.npz"
        shutil.copy2(source_oof, self.output / "trials" / source_oof.name)
        self.registry = {}
        receipt = []
        for historical in sorted(args.prior.parent.glob("cpu_round*/search_summary.json")):
            if historical.parent == self.output:
                continue
            summary = json.loads(historical.read_text())
            if (summary["train_sha256"] != self.source_hash or summary["split_sha256"] != self.split_hash
                    or summary["label_reference_sha256"] != self.label_hash):
                raise ValueError("Historical round does not share frozen data, labels and split")
            for record_path in sorted((historical.parent / "trials").glob("*.json")):
                result = json.loads(record_path.read_text())
                if result["id"] != config_id(result["config"]):
                    raise ValueError("Historical trial configuration hash is inconsistent")
                source_oof = record_path.with_name(f"{result['id']}-{result['seed']}-oof.npz")
                frozen = {"record_sha256": file_hash(record_path), "oof_sha256": file_hash(source_oof)}
                self.registry[(result["id"], result["seed"])] = (record_path, result, frozen)
                receipt.append({"id": result["id"], "seed": result["seed"],
                                "record": str(record_path.relative_to(ROOT)), **frozen})
        base.save_json(self.output / "historical_registry.json", receipt)
        self.registry_receipt_hash = file_hash(self.output / "historical_registry.json")
        self.new_config_ids = set()
        self.reused = []
        snapshot = self.output / "code_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name, digest in self.code_hashes.items():
            shutil.copy2(ROOT / "scripts" / name, snapshot / name)
            assert file_hash(snapshot / name) == digest
        base.save_json(snapshot / "snapshot_hashes.json", self.code_hashes)
        self.checkpoint("initialized_before_first_fit")

    def evaluate(self, config, batch, seed=SEED, stability=False):
        key = (config_id(config), seed)
        if key in self.registry:
            record_path, historical, frozen = self.registry[key]
            source_oof = record_path.with_name(f"{key[0]}-{seed}-oof.npz")
            if historical["config"] != config or historical["id"] != config_id(config):
                raise ValueError("Requested configuration differs from frozen historical trial")
            if (file_hash(record_path) != frozen["record_sha256"] or
                    file_hash(source_oof) != frozen["oof_sha256"]):
                raise ValueError("Historical trial or OOF changed after registry freeze")
            arrays, schema, _ = validate_oof(source_oof, self.dev, self.y, len(self.labels),
                                              historical["balanced_accuracy"])
            self.predictions[key] = arrays["prediction"]
            result = {**copy.deepcopy(historical), "batch": batch, "reused_without_fit": True,
                      "fit_count": 0, "historical_fit_count": historical["fit_count"],
                      "source_record_sha256": file_hash(record_path),
                      "source_oof_sha256": file_hash(source_oof), "oof_schema_sha256": object_hash(schema)}
            target_oof = self.output / "trials" / source_oof.name
            if not target_oof.exists():
                shutil.copy2(source_oof, target_oof)
            base.save_json(self.output / "trials" / record_path.name, result)
            if not stability and not any(row["id"] == key[0] for row in self.results):
                self.results.append(result)
            self.reused.append({"id": key[0], "seed": seed, "batch": batch,
                                "source": str(record_path.relative_to(ROOT)), "new_fits": 0})
            self.checkpoint("running")
            return result
        if not stability and (len(self.new_config_ids) >= 9 or self.fit_count + 5 > 45):
            raise TimeoutError("Round-three 9-configuration /45-fit budget reached")
        if stability and self.stability_fit_count + 5 > 20:
            raise TimeoutError("Round-three 20-stability-fit budget reached")
        result = super().evaluate(config, batch, seed, stability)
        if not stability:
            self.new_config_ids.add(result["id"])
        self.checkpoint("running")
        return result

    def checkpoint(self, reason, extra=None):
        report = super().checkpoint(reason, extra)
        report["configurations"] = len(getattr(self, "new_config_ids", []))
        report["evaluated_candidates"] = max(len(self.results) - 1, 0)
        report["reused_trials"] = getattr(self, "reused", [])
        report["prior_artifact_contract_sha256"] = self.prior_contract_hash
        report["historical_registry_sha256"] = self.registry_receipt_hash
        report["prior_search_summary_sha256"] = file_hash(self.prior_dir / "search_summary.json")
        report["cumulative_configurations"] = self.prior["cumulative_configurations"] + report["configurations"]
        report["cumulative_main_cv_fits"] = self.prior["cumulative_main_cv_fits"] + self.fit_count
        report["cumulative_stability_cv_fits"] = self.prior["cumulative_stability_cv_fits"] + self.stability_fit_count
        report["cumulative_search_wall_seconds"] = self.prior["cumulative_search_wall_seconds"] + report["wall_seconds"]
        report["budget"] = {"new_configurations": 9, "main_fits": 45, "stability_fits": 20,
                            "wall_seconds": self.args.hours * 3600}
        base.save_json(self.output / "search_summary.json", report)
        return report

    def run(self):
        initial = self.incumbent["config"]
        self.run_batch("1_style_weight_refinement", [{**initial, "style_weight": weight}
                       for weight in (.05, .1, .5)])
        best = max(self.results, key=lambda row: row["balanced_accuracy"])["config"]
        self.run_batch("2_style_regularization", [{**best, "classifier": "logistic", "C": c}
                       for c in (.25, 1., 16.)])
        best = max(self.results, key=lambda row: row["balanced_accuracy"])["config"]
        self.run_batch("3_style_features_and_classifier", [
            {**best, "char_ngram": [1, 3]}, {**best, "char_ngram": [2, 4]},
            {**best, "classifier": "svm", "C": 1.}])
        candidate = max(self.results[1:], key=lambda row: row["balanced_accuracy"])
        selected = max(self.results, key=lambda row: row["balanced_accuracy"])
        stability = []
        for seed in base.STABILITY_SEEDS:
            new = self.evaluate(candidate["config"], "stability_candidate", seed, True)
            old = self.evaluate(self.incumbent["config"], "stability_incumbent", seed, True)
            comparison = base.paired_group_bootstrap(self.y[self.dev],
                self.predictions[(self.incumbent["id"], seed)], self.predictions[(candidate["id"], seed)], self.groups[self.dev])
            stability.append({"seed": seed, "candidate": new, "incumbent": old, "comparison": comparison})
        total_gain = base.paired_group_bootstrap(self.y[self.dev], self.predictions[(self.incumbent["id"], SEED)],
                    self.predictions[(selected["id"], SEED)], self.groups[self.dev])
        total_folds = np.asarray(selected["fold_balanced_accuracy"]) - self.incumbent["fold_balanced_accuracy"]
        total_gain["fold_deltas"] = total_folds.tolist()
        total_gain["confirmed_improvement"] = base.confirmed_improvement(total_gain, total_folds)
        all_converged = all(not any(row["convergence_warnings"]) for row in self.results)
        all_converged &= all(not any(item[side]["convergence_warnings"])
                             for item in stability for side in ("candidate", "incumbent"))
        stable = all(item["comparison"]["delta_balanced_accuracy"] >= 0 for item in stability)
        no_confirmed = all(not item["comparison"]["confirmed_improvement"] for item in self.batches)
        engineering_stop = no_confirmed and not total_gain["confirmed_improvement"] and all_converged
        bounded_gain = engineering_stop and stable and total_gain["ci95_high"] < .005 and all(
            item["comparison"]["ci95_high"] < .005 for item in self.batches)
        reason = "three_batches_without_confirmed_improvement" if engineering_stop else "configuration_budget_boundary"
        if not all_converged:
            reason = "incomplete_direction_due_to_convergence"
        result = self.checkpoint(reason, {"selected": selected, "best_new": candidate,
            "selected_vs_prior_incumbent": total_gain, "stability": stability,
            "candidate_rank_consistent_across_stability_seeds": stable,
            "all_main_and_stability_trials_converged": bool(all_converged),
            "small_gain_supported_in_tried_batches": bool(bounded_gain),
            "stop_interpretation": "Engineering stopping evidence only; wide intervals or rank reversal cannot establish a plateau."})
        validate_prior(self.output, self.args.data_dir, ROOT / "config" / "labels.json")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["search", "finalize"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--prior", type=Path, default=ROOT / "outputs" / "cpu_round2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cpu_round3")
    parser.add_argument("--hours", type=float, default=.5)
    args = parser.parse_args()
    base.make_features = extended.make_features
    original_feature_config = base.feature_config
    base.feature_config = lambda config: {**original_feature_config(config),
        "round2_implementation_sha256": file_hash(ROOT / "scripts" / "model_cpu_round2.py")}
    with threadpool_limits(limits=2):
        if args.action == "search":
            runner = RoundThree(args)
            try:
                runner.run()
            except (TimeoutError, KeyboardInterrupt) as exc:
                runner.checkpoint("incomplete_budget_or_interruption", {"reason": type(exc).__name__})
                raise
        else:
            base.finalize(args)


if __name__ == "__main__":
    main()
