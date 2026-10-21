"""Behavior tests for leakage protection and output integrity; synthetic only."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from model_cpu import checked_submission, make_features
from train_cpu import SEED, confirmed_improvement, finalize, freeze_split, grouped_folds, paired_group_bootstrap


class ModelSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ntl-model-synthetic-")
        self.root = Path(self.temporary.name)
        self.labels = [f"synthetic-label-{index}" for index in range(18)]
        self.rows = [{"id": f"row-{group}-{label}", "prompt_id": f"prompt-{group:02}",
                      "text": f"unique synthetic response group {group} label {label}",
                      "model": self.labels[label]}
                     for group in range(20) for label in range(18)]

    def tearDown(self):
        self.temporary.cleanup()

    def test_frozen_holdout_and_every_cv_split_are_prompt_disjoint_and_complete(self):
        manifest = self.root / "splits.json"
        dev, heldout, groups, y = freeze_split(self.rows, self.labels, manifest, "synthetic-sha")
        self.assertEqual(len(set(groups[heldout])), 4)
        self.assertFalse(set(groups[dev]) & set(groups[heldout]))
        for seed in (SEED, SEED + 1, SEED + 2):
            folds = grouped_folds(groups, dev, seed)
            self.assertEqual(sorted(np.concatenate([val for _, val in folds])), sorted(dev))
            for train, valid in folds:
                self.assertFalse(set(groups[train]) & set(groups[valid]))
                self.assertFalse(set(valid) & set(heldout))
                self.assertEqual(len(set(y[valid])), 18)
        frozen = manifest.read_bytes()
        other = freeze_split(self.rows, self.labels, manifest, "synthetic-sha")
        self.assertEqual(manifest.read_bytes(), frozen)
        np.testing.assert_array_equal(other[0], dev)
        with self.assertRaisesRegex(ValueError, "does not match"):
            freeze_split(self.rows, self.labels, manifest, "changed-sha")

    def test_cross_prompt_normalized_duplicate_requires_a_new_grouping_policy(self):
        self.rows[-1]["text"] = "  " + self.rows[0]["text"].replace(" ", "\t") + "  "
        with self.assertRaisesRegex(ValueError, "Cross-prompt duplicate"):
            freeze_split(self.rows, self.labels, self.root / "split.json", "synthetic")

    def test_unseen_validation_tokens_do_not_enter_training_vocabulary(self):
        features = make_features({"representation": "word", "min_df": 1})
        features.fit_transform(["training alpha", "training beta"])
        transformed = features.transform(["heldout_secret_token"])
        vocabulary = features.transformer_list[0][1].vocabulary_
        self.assertNotIn("heldout_secret_token", vocabulary)
        self.assertEqual(transformed.nnz, 0)

    def test_group_bootstrap_identity_and_unambiguous_improvement(self):
        y = np.tile(np.arange(18), 20)
        groups = np.repeat(np.arange(20), 18)
        identical = paired_group_bootstrap(y, y, y, groups, draws=100)
        self.assertEqual(identical["ci95_low"], 0)
        self.assertEqual(identical["ci95_high"], 0)
        improved = paired_group_bootstrap(y, np.zeros_like(y), y, groups, draws=100)
        self.assertAlmostEqual(improved["delta_balanced_accuracy"], 17 / 18)
        self.assertAlmostEqual(improved["ci95_low"], 17 / 18)
        self.assertEqual(improved["groups"], 20)

    def test_float_residue_at_zero_is_not_a_confirmed_positive_interval(self):
        comparison = {"delta_balanced_accuracy": .007, "ci95_low": 3.8e-19}
        self.assertFalse(confirmed_improvement(comparison, [.01, .01, .01, -.001, -.001]))
        comparison["ci95_low"] = .001
        self.assertTrue(confirmed_improvement(comparison, [.01, .01, .01, -.001, -.001]))

    def test_submission_keeps_arbitrary_row_count_and_rejects_invalid_predictions(self):
        rows = [{"id": f"id-{index}"} for index in range(7)]
        output = self.root / "submission.parquet"
        result = checked_submission(rows, self.labels[:7], self.labels, output)
        self.assertEqual(result["rows"], 7)
        self.assertEqual(pq.read_table(output).column("id").to_pylist(), [row["id"] for row in rows])
        with self.assertRaisesRegex(ValueError, "every test row"):
            checked_submission(rows, self.labels[:6], self.labels, output)
        with self.assertRaisesRegex(ValueError, "reference label"):
            checked_submission(rows, ["unknown"] * 7, self.labels, output)
        with self.assertRaisesRegex(ValueError, "unique"):
            checked_submission([{"id": "duplicate"}] * 7, self.labels[:7], self.labels, output)

    def test_final_holdout_cannot_be_opened_again(self):
        (self.root / "search_summary.json").write_text(json.dumps({}))
        (self.root / "holdout_opened.json").write_text(json.dumps({"opened": True}))
        with self.assertRaisesRegex(ValueError, "already opened"):
            finalize(argparse.Namespace(output=self.root, data_dir=self.root))

    def test_changed_split_is_rejected_before_final_holdout_is_opened(self):
        (self.root / "search_summary.json").write_text(json.dumps({"selected": {}, "split_sha256": "old-hash"}))
        (self.root / "split_manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Frozen split changed"):
            finalize(argparse.Namespace(output=self.root, data_dir=self.root))
        self.assertFalse((self.root / "holdout_opened.json").exists())


if __name__ == "__main__":
    unittest.main()
