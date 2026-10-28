import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from margin_center_round10 import center_margins, synthetic_inputs


class MarginTests(unittest.TestCase):
    def test_hand_calculation_and_isolation(self):
        x = np.tile(np.array([2., 6., 9.])[:, None], (1, 18))
        np.testing.assert_allclose(center_margins(x, [0, 0, 1])[:, 0], [0., 4., 9.])
        np.testing.assert_array_equal(x[:, 0], [2., 6., 9.])

    def test_reverse_and_oversize(self):
        x = np.random.default_rng(42).normal(size=(22, 18))
        g = np.array([0] * 19 + [1, 1, 2])
        out = center_margins(x, g)
        np.testing.assert_array_equal(out[:19], x[:19])
        np.testing.assert_allclose(center_margins(x[::-1], g[::-1])[::-1], out, atol=1e-12)

    def test_invalid(self):
        for x, g in [(np.zeros((2, 17)), [0, 0]), (np.full((2, 18), np.nan), [0, 0]), (np.zeros((2, 18)), [0])]:
            with self.assertRaises(ValueError):
                center_margins(x, g)

    def test_synthetic_grouping_exercises_correction(self):
        from topic_word_round9 import infer_groups
        texts, ids = synthetic_inputs()
        self.assertEqual(len(texts), 80)
        groups = infer_groups(texts, ids, dict(min_df=2, max_df=.05, cap=18, threshold=.15))
        self.assertEqual(sorted(np.unique(groups, return_counts=True)[1]), [2] * 40)
        np.testing.assert_array_equal(groups, infer_groups(texts[::-1], ids[::-1], dict(min_df=2, max_df=.05, cap=18, threshold=.15))[::-1])


if __name__ == "__main__":
    unittest.main()
