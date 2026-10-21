"""Risk-focused tests for the development-only runner using synthetic files."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / "scripts"))
from compare_team_cpu import (canonical_probabilities, claim_run, grouped_folds,
                              load_development, load_oof, paired_interval,
                              resume_float32_failure, save_oof,
                              validate_probabilities, write_json)
from artifact_guards import object_hash
from model_cpu import file_hash


class TeamComparisonSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ntl-dev-guard-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.train = self.folder / "train.csv"
        self.split = self.folder / "split.json"
        self.labels = self.folder / "labels.json"
        label_names = [f"synthetic-model-{i}" for i in range(18)]
        self.labels.write_text(json.dumps({"labels": label_names}))
        rows, entries, development, protected = [], [], [], []
        # Interleave groups so row slicing cannot accidentally imitate grouping.
        for label_index, label in enumerate(label_names):
            for prompt in range(7):
                index = len(rows)
                role = "holdout" if prompt == 6 else "development"
                text = "PROTECTED SENTINEL" if role == "holdout" else f"Synthetic response {label_index} {prompt}"
                rows.append([f"row-{index}", f"prompt-{prompt}", text, label])
                entries.append({"row_index": index, "id": f"row-{index}",
                                "prompt_id": f"prompt-{prompt}", "role": role})
                (protected if role == "holdout" else development).append(index)
        with self.train.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "prompt_id", "text", "model"])
            writer.writerows(rows)
        self.manifest = {"train_sha256": file_hash(self.train), "rows": entries,
                         "development_indices": development, "holdout_indices": protected}
        self.split.write_text(json.dumps(self.manifest))
        self.config = {"train_sha256": file_hash(self.train), "split_sha256": file_hash(self.split),
                       "labels_sha256": file_hash(self.labels), "development_rows": len(development),
                       "development_groups": 6, "bootstrap_seed": 42, "bootstrap_repeats": 100}

    def load(self, config=None):
        return load_development(self.train, self.split, self.labels, config or self.config)

    def test_frozen_split_rejects_changed_order_even_when_partition_remains_valid(self):
        original = self.load()
        self.assertEqual(len(original["texts"]), 108)
        self.assertFalse(any("PROTECTED" in text for text in original["texts"]))
        changed = copy.deepcopy(self.manifest)
        changed["development_indices"].reverse()
        self.split.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "Frozen split_sha256 mismatch"):
            self.load()

    def test_role_and_group_protection_survive_a_matching_file_hash(self):
        # Rebind hashes only in the synthetic fixture to exercise semantic checks.
        cases = ["development_role", "holdout_role", "shared_prompt", "overlapping_rows"]
        for case in cases:
            with self.subTest(case=case):
                manifest = copy.deepcopy(self.manifest)
                dev = manifest["development_indices"][0]
                hold = manifest["holdout_indices"][0]
                if case == "development_role":
                    manifest["rows"][dev]["role"] = "holdout"
                    message = "protected row"
                elif case == "holdout_role":
                    manifest["rows"][hold]["role"] = "development"
                    message = "Holdout role"
                elif case == "shared_prompt":
                    manifest["rows"][hold]["prompt_id"] = manifest["rows"][dev]["prompt_id"]
                    message = "protected prompt"
                else:
                    manifest["development_indices"].append(hold)
                    message = "complete disjoint partition"
                self.split.write_text(json.dumps(manifest))
                config = {**self.config, "split_sha256": file_hash(self.split)}
                with self.assertRaisesRegex(ValueError, message):
                    self.load(config)

    def test_grouped_folds_cover_every_development_row_once_without_prompt_overlap(self):
        data = self.load()
        folds, fold_id = grouped_folds(data, 5)
        coverage = np.zeros(len(data["truth"]), dtype=int)
        all_rows = set(range(len(data["truth"])))
        self.assertEqual(len(folds), 5)
        for number, (training, validation) in enumerate(folds):
            self.assertFalse(set(training) & set(validation))
            self.assertEqual(set(training) | set(validation), all_rows)
            self.assertFalse(set(data["groups"][training]) & set(data["groups"][validation]))
            self.assertEqual(set(data["truth"][training]), set(range(18)))
            np.testing.assert_array_equal(fold_id[validation], np.full(len(validation), number))
            coverage[validation] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(coverage), dtype=int))
        for prompt in set(data["groups"]):
            self.assertEqual(len(set(fold_id[data["groups"] == prompt])), 1)

    def oof_fixture(self):
        data = self.load()
        _, fold_id = grouped_folds(data, 5)
        probabilities = np.eye(18)[data["truth"]]
        path = self.folder / "oof.npz"
        save_oof(path, probabilities, data, fold_id, "frozen-contract")
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        return path, data, fold_id, probabilities, payload

    def test_oof_identity_changes_are_rejected_including_label_order_and_fold(self):
        path, data, fold_id, probabilities, original = self.oof_fixture()
        np.testing.assert_array_equal(load_oof(path, data, fold_id, "frozen-contract"), probabilities)
        for field in ("row_index", "ids", "groups", "truth", "labels", "fold_id", "contract_sha256"):
            with self.subTest(field=field):
                payload = {name: value.copy() for name, value in original.items()}
                if field == "contract_sha256":
                    payload[field] = np.asarray("different-contract")
                elif field in ("truth", "fold_id"):
                    payload[field][0] += 1
                else:
                    payload[field][[0, 1]] = payload[field][[1, 0]]
                np.savez_compressed(path, **payload)
                with self.assertRaisesRegex(ValueError, f"OOF identity mismatch: {field}"):
                    load_oof(path, data, fold_id, "frozen-contract")

    def test_oof_missing_fields_and_invalid_probability_rows_are_rejected(self):
        path, data, fold_id, probabilities, original = self.oof_fixture()
        incomplete = {name: value for name, value in original.items() if name != "truth"}
        np.savez_compressed(path, **incomplete)
        with self.assertRaisesRegex(ValueError, "Unexpected OOF fields"):
            load_oof(path, data, fold_id, "frozen-contract")
        invalid = {}
        invalid["wrong_shape"] = probabilities[:, :-1]
        invalid["integer_dtype"] = probabilities.astype(np.int64)
        for name, replacement in (("nan", np.nan), ("negative", -0.1), ("over_one", 1.1)):
            value = probabilities.copy()
            value[0, 0] = replacement
            invalid[name] = value
        value = probabilities.copy()
        value[0] = 0
        invalid["uncovered_row"] = value
        value = probabilities.copy()
        value[0] *= .5
        invalid["wrong_row_sum"] = value
        for name, value in invalid.items():
            with self.subTest(case=name):
                np.savez_compressed(path, **{**original, "probabilities": value})
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    load_oof(path, data, fold_id, "frozen-contract")
        rejected = self.folder / "rejected.npz"
        with self.assertRaisesRegex(ValueError, "uncovered"):
            save_oof(rejected, invalid["uncovered_row"], data, fold_id, "frozen-contract")
        self.assertFalse(rejected.exists())

    def test_failed_claim_cannot_be_reclaimed_or_overwritten(self):
        path = self.folder / "claims" / "one-run.json"
        claim = claim_run(path, "first-contract")
        with self.assertRaises(FileExistsError):
            claim_run(path, "second-contract")
        claim.update(state="failed", attempted_fits=1, completed_fits=0)
        write_json(path, claim)
        failed_bytes = path.read_bytes()
        with self.assertRaises(FileExistsError):
            claim_run(path, "third-contract")
        self.assertEqual(path.read_bytes(), failed_bytes)
        self.assertEqual(json.loads(failed_bytes)["final_holdout_evaluations"], 0)

    def test_paired_intervals_match_hand_calculated_prediction_differences(self):
        data = self.load()
        correct = np.eye(18)[data["truth"]]
        wrong = np.eye(18)[(data["truth"] + 1) % 18]
        same_predictions = .9 * correct + .1 / 18
        # Each group contains all 18 classes, so every group resample has the
        # same BA differences: 0 for identical predictions, +1/-1 for all right/wrong.
        self.assertEqual(paired_interval(correct, same_predictions, data, self.config), [0., 0.])
        self.assertEqual(paired_interval(wrong, correct, data, self.config), [1., 1.])
        self.assertEqual(paired_interval(correct, wrong, data, self.config), [-1., -1.])

    def test_float32_softmax_roundoff_is_canonicalized_but_bad_values_are_rejected(self):
        logits = np.random.default_rng(42).normal(size=(100, 18)).astype(np.float32)
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        original = probabilities.copy()
        self.assertGreater(float(np.max(np.abs(probabilities.sum(axis=1) - 1))), 1e-8)
        validate_probabilities(probabilities, 100, 18)
        canonical = canonical_probabilities(probabilities, 100, 18)
        self.assertEqual(canonical.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(canonical.sum(axis=1), 1, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(canonical.argmax(axis=1), probabilities.argmax(axis=1))
        np.testing.assert_array_equal(probabilities, original)
        self.assertFalse(np.shares_memory(canonical, probabilities))
        for case in ("nan", "negative", "over_one", "uncovered", "gross_mass_error"):
            with self.subTest(case=case):
                bad = probabilities.copy()
                if case == "uncovered":
                    bad[0] = 0
                elif case == "gross_mass_error":
                    bad[0] *= .99
                else:
                    bad[0, 0] = {"nan": np.nan, "negative": -.001, "over_one": 1.001}[case]
                # Normalization must not conceal malformed probabilities.
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    canonical_probabilities(bad, 100, 18)

    def numerical_failure_fixture(self, name):
        directory = self.folder / name
        output = directory / "original_output"
        output.mkdir(parents=True)
        old_config = {"max_fits": 20, "folds": 5, "max_seconds": 1200,
                      "models": [{"name": "original_style", "C": 16}]}
        write_json(output / "config.json", old_config)
        source_hashes = {}
        config_relative = "name-that-llm/config/team_dev_20261021.json"
        for relative in ("name-that-llm/scripts/compare_team_cpu.py",
                         config_relative,
                         "name-that-llm/docs/TEAM_CONTINUATION_PLAN.md",
                         "name-that-llm/scripts/model_cpu.py"):
            snapshot = output / "source_snapshot" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if relative == config_relative:
                # Match the actual runner: source formatting differs from the
                # pretty-printed output copy although their JSON values agree.
                snapshot.write_text(json.dumps(old_config, separators=(",", ":")) + "\n")
            else:
                snapshot.write_text(f"Synthetic frozen source: {relative}\n")
            source_hashes[relative] = file_hash(snapshot)
        old_contract = {"config_sha256": source_hashes[config_relative],
                        "source_sha256": source_hashes,
                        "train_sha256": "unchanged-training-data", "split_sha256": "unchanged-split",
                        "fold_id_sha256": "unchanged-folds", "labels": list(range(18)),
                        "packages": {"numpy": np.__version__}}
        write_json(output / "contract.json", old_contract)
        claim_path = directory / "claim.json"
        claim = claim_run(claim_path, object_hash(old_contract))
        claim.update(state="failed", attempted_fits=1, completed_fits=0, wall_seconds=12.5,
                     current={"model": "original_style", "fold": 0},
                     error="ValueError: Invalid or uncovered probability rows")
        write_json(claim_path, claim)
        new_config = {**old_config, "max_fits": 21}
        new_contract = copy.deepcopy(old_contract)
        new_contract["config_sha256"] = object_hash(new_config)
        # Only the numerical guard, declared budget and plan are allowed to change.
        for relative in list(source_hashes)[:3]:
            new_contract["source_sha256"][relative] = hashlib.sha256(
                f"Synthetic numerical-repair source: {relative}".encode()).hexdigest()
        new_contract["source_sha256"][config_relative] = new_contract["config_sha256"]
        return claim_path, output, new_contract, new_config

    def test_numerical_recovery_preserves_failed_evidence_and_fit_cost_once(self):
        path, output, contract, config = self.numerical_failure_fixture("allowed-recovery")
        failed_claim = path.read_bytes()
        original_files = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        self.assertNotEqual(file_hash(output / "config.json"), file_hash(
            output / "source_snapshot/name-that-llm/config/team_dev_20261021.json"))
        resumed = resume_float32_failure(path, output, contract, config)
        self.assertEqual(resumed["state"], "running")
        self.assertEqual(resumed["attempted_fits"], 1)
        self.assertEqual(resumed["completed_fits"], 0)
        self.assertEqual(resumed["final_holdout_evaluations"], 0)
        self.assertEqual(resumed["recovery"]["discarded_first_fold_fit"], 1)
        self.assertEqual(resumed["recovery"]["discarded_first_fit_wall_seconds"], 12.5)
        self.assertEqual(resumed["recovery"]["original_error"],
                         "ValueError: Invalid or uncovered probability rows")
        self.assertNotIn("error", resumed)
        self.assertEqual((output / "failed_claim.json").read_bytes(), failed_claim)
        for relative, value in original_files.items():
            self.assertEqual((output / relative).read_bytes(), value)
        with self.assertRaises(ValueError):
            resume_float32_failure(path, output, contract, config)
        # Even another failure cannot remove the fact that recovery was consumed.
        resumed["state"] = "failed"
        write_json(path, resumed)
        failed_recovery = path.read_bytes()
        with self.assertRaises(ValueError):
            resume_float32_failure(path, output, contract, config)
        self.assertEqual(path.read_bytes(), failed_recovery)
        self.assertEqual((output / "failed_claim.json").read_bytes(), failed_claim)

    def test_numerical_recovery_rejects_drift_or_preexisting_scored_artifacts(self):
        for case in ("model_config", "fold_config", "fold_identity", "model_source",
                     "old_snapshot", "existing_oof", "existing_recovery", "extra_budget"):
            with self.subTest(case=case):
                path, output, contract, config = self.numerical_failure_fixture(case)
                if case == "model_config":
                    config = copy.deepcopy(config)
                    config["models"][0]["C"] = 8
                elif case == "fold_config":
                    config["folds"] = 4
                elif case == "fold_identity":
                    contract["fold_id_sha256"] = "different-folds"
                elif case == "model_source":
                    contract["source_sha256"]["name-that-llm/scripts/model_cpu.py"] = "different-model"
                elif case == "old_snapshot":
                    (output / "source_snapshot/name-that-llm/scripts/model_cpu.py").write_text("changed")
                elif case == "existing_oof":
                    np.savez_compressed(output / "original_style_oof.npz", probabilities=np.eye(18))
                elif case == "existing_recovery":
                    (output / "recovery_01").mkdir()
                else:
                    config["max_fits"] = 22
                before = path.read_bytes()
                expected_error = {
                    "model_config": "cannot change models", "fold_config": "cannot change models",
                    "extra_budget": "cannot change models", "fold_identity": "input, split, labels or runtime",
                    "model_source": "cannot change model implementation", "old_snapshot": "Original source snapshot",
                    "existing_oof": "prior scored artifacts", "existing_recovery": "Recovery already started",
                }[case]
                with self.assertRaisesRegex(ValueError, expected_error):
                    resume_float32_failure(path, output, contract, config)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse((output / "failed_claim.json").exists())


if __name__ == "__main__":
    unittest.main()
