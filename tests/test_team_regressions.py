"""Synthetic behavior checks; no competition data, downloads or model training."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from centered_stylegram import center_by_groups, cluster_texts
from run_embed import embedding_comparison_pools


def unit_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms == 0, 1, norms)


class CenteringRegressionTests(unittest.TestCase):
    def test_matches_dense_reference_without_mutating_input(self):
        values = np.array([[3., 4., 0.], [0., 0., 0.], [2., 0., 1.],
                           [3., 4., 0.], [0., 2., 3.], [0., 0., 5.]])
        groups = np.array(["a", "b", "b", "a", "b", "single"])
        expected = unit_rows(values)
        for group in np.unique(groups):
            mask = groups == group
            if mask.sum() > 1:
                expected[mask] -= expected[mask].mean(axis=0)
        expected = unit_rows(expected)
        matrix = sparse.csr_matrix(values)
        original = matrix.copy()
        actual = center_by_groups(matrix, groups)
        self.assertTrue(sparse.isspmatrix_csr(actual))
        np.testing.assert_allclose(actual.toarray(), expected, atol=1e-12)
        np.testing.assert_array_equal(matrix.toarray(), original.toarray())
        # Identical rows in a multi-row group must both vanish, not just its first row.
        np.testing.assert_array_equal(actual.toarray()[[0, 3]], np.zeros((2, 3)))

    def test_group_centering_is_equivariant_to_row_permutation(self):
        values = sparse.csr_matrix([[1., 0., 2.], [2., 1., 0.],
                                    [0., 1., 2.], [0., 3., 1.]])
        groups = np.array([2, 1, 2, 1])
        permutation = np.array([3, 0, 2, 1])
        expected = center_by_groups(values, groups).toarray()[permutation]
        actual = center_by_groups(values[permutation], groups[permutation]).toarray()
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_empty_and_singleton_groups_are_supported(self):
        empty = center_by_groups(sparse.csr_matrix((0, 3)), np.array([], dtype=int))
        self.assertEqual(empty.shape, (0, 3))
        singleton = center_by_groups(sparse.csr_matrix([[3., 4.]]), np.array([1]))
        np.testing.assert_allclose(singleton.toarray(), [[.6, .8]])
        with self.assertRaisesRegex(ValueError, "one entry"):
            center_by_groups(sparse.eye(2), np.array([1]))

    def test_clustering_tiny_and_empty_vocabulary_inputs(self):
        cases = [([], None), (["the"], None), (["the", "and"], 2),
                 (["uniquealpha", "uniquebeta", "uniquegamma"], 2),
                 (["repeat", "repeat", "repeat"], 2)]
        for texts, groups in cases:
            with self.subTest(texts=texts):
                actual = cluster_texts(texts, n_groups=groups)
                np.testing.assert_array_equal(actual, np.zeros(len(texts), dtype=int))


class EmbeddingPoolRegressionTests(unittest.TestCase):
    def test_current_embedding_is_excluded_from_baseline_and_added_once(self):
        with tempfile.TemporaryDirectory(prefix="ntl-pool-test-") as folder:
            directory = Path(folder)
            current = np.full((4, 3), .7)
            current_path = directory / "oof_emb_answerdotai__current.npy"
            np.save(current_path, current)
            np.save(directory / "pool_v2_char25.npy", np.full((4, 3), .2))
            np.save(directory / "oof_emb_answerdotai__historical.npy", np.full((4, 3), .3))
            np.save(directory / "oof_emb_wrong_shape.npy", np.zeros((2, 3)))
            baseline, augmented = embedding_comparison_pools(directory, current_path, current)
            self.assertEqual(set(baseline), {"char25", "emb:historical"})
            self.assertEqual(set(augmented), set(baseline) | {"emb"})
            self.assertFalse(any(np.array_equal(value, current) for value in baseline.values()))
            self.assertEqual(sum(np.array_equal(value, current) for value in augmented.values()), 1)
            self.assertIs(augmented["emb"], current)

    def test_current_file_alone_does_not_create_its_own_baseline(self):
        with tempfile.TemporaryDirectory(prefix="ntl-pool-test-") as folder:
            directory = Path(folder)
            current = np.full((2, 3), .5)
            current_path = directory / "oof_emb_current.npy"
            np.save(current_path, current)
            baseline, augmented = embedding_comparison_pools(directory, current_path, current)
            self.assertEqual(baseline, {})
            self.assertEqual(set(augmented), {"emb"})


if __name__ == "__main__":
    unittest.main()
