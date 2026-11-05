import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import r24_assignment as r24
import r25_assignment as r25


class AssignmentTest(unittest.TestCase):
    def test_zero_beta_and_zero_alpha(self):
        rng = np.random.default_rng(25)
        z = rng.normal(size=(30, 18)); groups = np.repeat([0, 1, 2], [12, 11, 7])
        for alpha in (0., .5, 2.):
            np.testing.assert_array_equal(r25.predict(z, groups, alpha, 0, 3, 4), r24.predict(z, groups, alpha))
        expected = r25.biased_scores(z, 1., 3, 4).argmax(axis=1)
        np.testing.assert_array_equal(r25.predict(z, groups, 0, 1., 3, 4), expected)

    def test_eligibility_and_ties(self):
        z = np.array([[0., 1., 2.], [3., 1., 2.], [0., 2., 2.], [0., 1., 2.]])
        biased = r25.biased_scores(z, 1., 1, 2)
        np.testing.assert_array_equal(biased - z, [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]])
        self.assertEqual(biased[0].argmax(), 1)  # equality follows first-index argmax
        np.testing.assert_array_equal(z, [[0, 1, 2], [3, 1, 2], [0, 2, 2], [0, 1, 2]])

    def test_small_clusters_and_brute_force(self):
        z = np.zeros((11, 3)); z[:, 2] = 1
        np.testing.assert_array_equal(r25.predict(z, np.zeros(11), 2, .5, 1, 2), np.full(11, 2))
        rng = np.random.default_rng(8)
        for n in range(1, 5):
            z = rng.integers(-2, 3, (n, 3)).astype(float)
            biased = r25.biased_scores(z, .5, 1, 2)
            candidates = list(itertools.product(range(3), repeat=n))
            best = max(r24.objective(biased, p, 1.) for p in candidates)
            expected = min(p for p in candidates if r24.objective(biased, p, 1.) == best)
            np.testing.assert_array_equal(r24.solve(biased, 1.), expected)


if __name__ == '__main__':
    unittest.main()
