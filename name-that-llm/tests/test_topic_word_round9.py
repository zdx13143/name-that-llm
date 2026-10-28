import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import topic_word_round9 as topic
import windows_topic_word_round9 as runner


class TopicWordTests(unittest.TestCase):
    def test_partial_center_matches_hand_calculation_and_preserves_other_rows(self):
        x = np.array([[2, 0], [0, 2], [3, 4]], dtype=np.float32)
        actual = topic.center_word(csr_matrix(x), np.array([4, 4, 9]), 0.5).toarray()
        np.testing.assert_allclose(actual, [[1.5, -.5], [-.5, 1.5], [3, 4]])
        np.testing.assert_array_equal(x, [[2, 0], [0, 2], [3, 4]])

    def test_large_groups_unchanged(self):
        x = csr_matrix(np.eye(20, dtype=np.float32))
        np.testing.assert_array_equal(topic.center_word(x, np.zeros(20)).toarray(), x.toarray())

    def test_cap_prevents_merging_large_cluster(self):
        s = np.full((6, 6), 0.8)
        groups = topic.capped_average(s, cap=2)
        self.assertEqual(sorted(np.unique(groups, return_counts=True)[1].tolist()), [2, 2, 2])
        self.assertEqual(len(np.unique(topic.capped_average(s, cap=6, threshold=.9))), 6)

    def test_text_only_grouping_and_order_invariance(self):
        self.assertEqual(list(inspect.signature(topic.infer_groups).parameters), ["texts", "ids", "settings"])
        texts = np.array([f"common topic{k} distinct{k}" for k in range(30) for _ in range(2)])
        ids = np.array([f"row{i:03}" for i in range(60)])
        cfg = {"min_df": 2, "max_df": .05, "cap": 18, "threshold": .15}
        groups = topic.infer_groups(texts, ids, cfg)
        np.testing.assert_array_equal(groups, topic.infer_groups(texts[::-1], ids[::-1], cfg)[::-1])
        self.assertEqual(sorted(np.unique(groups, return_counts=True)[1].tolist()), [2]*30)

    def test_tiny_and_empty_vocabulary_fallback_are_stable(self):
        cfg = {"min_df": 2, "max_df": .05, "cap": 18, "threshold": .15}
        for n in (3, 40):
            texts, ids = np.array(["the and of"]*n), np.array([str(i) for i in range(n)])
            g = topic.infer_groups(texts, ids, cfg)
            self.assertEqual(len(np.unique(g)), n)
            np.testing.assert_array_equal(g, topic.infer_groups(texts[::-1], ids[::-1], cfg)[::-1])

    def test_diagnostic_pair_counts_and_coverage(self):
        d = topic.group_diagnostics([0, 0, 1, 1], [3, 3, 4, 5])
        self.assertEqual(d["eligible_rows"], 4)
        self.assertEqual(d["pair_precision"], .5)
        self.assertEqual(d["pair_recall"], 1)

    def test_budget_persists_before_grouping_and_refuses_extra(self):
        with tempfile.TemporaryDirectory(dir=runner.MODULE / "outputs") as tmp:
            path = Path(tmp) / "claim.json"
            claim = {"attempted_groupings": 0, "completed_groupings": 0}
            runner.write_json(path, claim)
            cfg = {"budget": {"max_grouping_attempts": 1}, "method": {"grouping": {}}}
            with patch.object(runner, "CLAIM", path), patch.object(runner, "infer_groups", side_effect=ValueError("fail")):
                with self.assertRaises(ValueError):
                    runner.grouping(claim, cfg, [], [])
                self.assertTrue(path.exists())
                self.assertEqual(claim, {"attempted_groupings": 1, "completed_groupings": 0})
                with self.assertRaises(RuntimeError):
                    runner.grouping(claim, cfg, [], [])

    def test_existing_claim_blocks_before_loading_data(self):
        with tempfile.TemporaryDirectory(dir=runner.MODULE / "outputs") as tmp:
            path = Path(tmp) / "claim.json"
            path.touch()
            with patch.object(runner, "CLAIM", path), patch.object(runner, "load_inputs", side_effect=AssertionError("must not read")):
                self.assertEqual(runner.readiness({})[0]["state"], "blocked")

    def test_gate_rejects_transductive_gain_if_singletons_collapse(self):
        m = {"balanced_accuracy": .48, "fold_balanced_accuracy": [.48]*5}
        c = {"pooled_ba_gain": .03, "prompt_paired_gain_95ci": [.01, .05], "improved_folds": 5}
        cfg = {"min_pooled_balanced_accuracy": .45, "min_each_fold_balanced_accuracy": .35,
               "min_gain": .01, "min_improved_folds": 4, "min_half_batch_gain": -.01, "min_singleton_gain": -.02}
        self.assertTrue(topic.metric_gate(m, c, c, c, cfg)["metric_gate_passed"])
        self.assertFalse(topic.metric_gate(m, c, c, {"pooled_ba_gain": -.021}, cfg)["metric_gate_passed"])


if __name__ == "__main__":
    unittest.main()
