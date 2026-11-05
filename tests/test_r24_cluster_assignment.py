import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from r24_cluster import cluster, capped_agglo
from r24_assignment import objective, predict, scale_scores, solve


class ClusterAssignmentTest(unittest.TestCase):
    def test_matches_verbatim_v15(self):
        namespace = {'np': np}
        source = Path('scripts/r24_source/v15_cluster.py.txt').read_text(encoding='utf-8')
        exec(compile(source, 'v15_cluster.py.txt', 'exec'), namespace)
        texts = [f'commonword topic{i//3} unique{i} shared{i//3}' for i in range(120)]
        X = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=.05).fit_transform(texts)
        S = (X @ X.T).toarray()
        expected = namespace['capped_agglo'](S, 18, .15)
        actual, fallback = cluster(texts)
        self.assertFalse(fallback)
        np.testing.assert_array_equal(actual, expected)

    def test_cap_and_fallback(self):
        groups = capped_agglo(np.ones((40, 40)), 18, .15)
        self.assertLessEqual(max(np.unique(groups, return_counts=True)[1]), 18)
        groups, fallback = cluster(['the and'] * 50)
        self.assertTrue(fallback)
        np.testing.assert_array_equal(groups, np.arange(50))

    def test_brute_force(self):
        rng = np.random.default_rng(12)
        for n in range(1, 5):
            for _ in range(8):
                z = rng.integers(-3, 4, size=(n, 3)).astype(float)
                alpha = 1.
                possibilities = list(itertools.product(range(3), repeat=n))
                best = max(objective(z, p, alpha) for p in possibilities)
                expected = min(p for p in possibilities if objective(z, p, alpha) == best)
                np.testing.assert_array_equal(solve(z, alpha), expected)

    def test_zero_ties_and_invariance(self):
        z = np.zeros((12, 3)); z[:, 0:2] = 2
        np.testing.assert_array_equal(solve(z, 0), np.zeros(12, dtype=int))
        near = np.array([[0., 1e-11]])
        np.testing.assert_array_equal(solve(near, 1.), [1])
        rng = np.random.default_rng(31)
        raw = rng.normal(size=(24, 18)); folds = np.arange(24) % 2
        a, _ = scale_scores(raw, folds)
        b, _ = scale_scores(raw * 7 + np.arange(24)[:, None], folds)
        groups = np.repeat([1, 2], 12)
        np.testing.assert_array_equal(predict(a, groups, .5), predict(b, groups, .5))

    def test_only_eligible_clusters(self):
        groups = np.repeat(np.arange(4), [11, 12, 18, 19])
        scores = np.zeros((len(groups), 18)); scores[:, 0] = 1
        result = predict(scores, groups, 2)
        self.assertTrue(np.all(result[groups == 0] == 0))
        self.assertTrue(np.all(result[groups == 3] == 0))
        self.assertTrue(np.any(result[groups == 1] != 0))
        self.assertTrue(np.any(result[groups == 2] != 0))


if __name__ == '__main__':
    unittest.main()
