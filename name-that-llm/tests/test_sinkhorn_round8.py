import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sinkhorn_round8 import balance, paired_comparison, metrics, gate


class SinkhornTests(unittest.TestCase):
    def test_balances_mass_preserves_rows_input_and_order(self):
        source = np.random.default_rng(11).uniform(.01, 1, (72, 18))
        source *= np.linspace(1, 5, 18)
        source /= source.sum(1, keepdims=True)
        original = source.copy()
        result = balance(source)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_allclose(result.sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(result.sum(0), 4, atol=1e-10)
        np.testing.assert_allclose(balance(source[::-1])[::-1], result, atol=1e-12)

    def test_small_batch_bypasses_and_fifty_rows_uses_balance(self):
        row = np.arange(1, 19, dtype=float)
        row /= row.sum()
        np.testing.assert_allclose(balance(np.tile(row, (49, 1))), np.tile(row, (49, 1)))
        np.testing.assert_allclose(balance(np.tile(row, (50, 1))), np.full((50, 18), 1/18))

    def test_zero_entries_stay_finite(self):
        source = np.eye(18)[np.arange(72) % 18]
        result = balance(source)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result.argmax(1), source.argmax(1))

    def test_invalid_inputs_rejected(self):
        for value in (np.zeros((0, 18)), np.ones((4, 17)), np.full((4, 18), np.nan), np.full((4, 18), -1.0)):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    balance(value)

    def test_paired_metrics_known_perfect_improvement(self):
        truth = np.tile(np.arange(18), 4)
        groups = np.repeat(np.arange(4), 18)
        candidate = np.eye(18)[truth]
        baseline = np.eye(18)[(truth + 1) % 18]
        result = paired_comparison(candidate, baseline, truth, groups, resamples=100, seed=5)
        self.assertEqual(result["gain"], 1.0)
        self.assertEqual(result["descriptive_prompt_95ci"], [1.0, 1.0])
        self.assertEqual(result["corrected"], 72)
        self.assertEqual(result["harmed"], 0)
        self.assertEqual(metrics(candidate, truth, np.arange(18))["balanced_accuracy"], 1.0)

    def test_gate_is_conjunctive_and_requires_four_folds(self):
        settings = dict(min_pooled_balanced_accuracy=.42, min_each_fold_balanced_accuracy=.35,
                        min_gain_over_r7=.01, paired_prompt_95ci_lower_exceeds=0, min_improved_folds=4)
        folds = [dict(candidate_ba=.46 if i < 3 else .45, baseline_ba=.45) for i in range(5)]
        result = gate(dict(balanced_accuracy=.46), folds, dict(gain=.01, descriptive_prompt_95ci=[.001, .02]), settings)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["improved_folds"])


if __name__ == "__main__":
    unittest.main()
