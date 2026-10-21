"""Synthetic risk checks; no competition text, fitted classifier or holdout use."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_prompt_centering as centering


class CenteringTests(unittest.TestCase):
    def test_matches_dense_oracle_and_does_not_mutate(self):
        original = np.array([[1., 2., 0.], [2., 0., 1.], [0., 0., 0.],
                             [0., 3., 4.], [0., 3., 4.]])
        groups = np.array(["a", "a", "b", "c", "c"])
        matrix = sparse.csr_matrix(original)
        for strength in (0., .5, 1.):
            expected = normalize(original)
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                if len(rows) > 1:
                    expected[rows] -= strength * expected[rows].mean(axis=0)
            expected = normalize(expected)
            result = centering.soft_center(matrix, groups, strength, 1_000_000)
            np.testing.assert_allclose(result.toarray(), expected, atol=1e-14)
            self.assertTrue(np.isfinite(result.data).all())
            np.testing.assert_array_equal(matrix.toarray(), original)

    def test_row_permutation_and_singletons(self):
        matrix = sparse.csr_matrix([[1., 2., 0.], [3., 1., 2.], [0., 0., 4.]])
        groups = np.array([4, 4, 9])
        order = np.array([2, 0, 1])
        original = centering.soft_center(matrix, groups, .5, 1_000_000)
        shuffled = centering.soft_center(matrix[order], groups[order], .5, 1_000_000)
        np.testing.assert_allclose(shuffled.toarray(), original.toarray()[order])
        np.testing.assert_allclose(
            centering.soft_center(matrix, np.arange(3), 1., 1_000_000).toarray(),
            normalize(matrix).toarray())

    def test_empty_and_zero_vectors(self):
        empty = centering.soft_center(sparse.csr_matrix((0, 3)), [], 1., 1000)
        self.assertEqual(empty.shape, (0, 3))
        zeros = centering.soft_center(sparse.csr_matrix((3, 2)), [0, 0, 1], 1., 1000)
        self.assertEqual(zeros.nnz, 0)

    def test_rejects_invalid_groups_strength_and_admission_budget(self):
        matrix = sparse.csr_matrix([[1., 0.], [0., 1.]])
        for groups in ([0], [0, float("nan")], [None, None], [[0], [1]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                centering.soft_center(matrix, groups, .5, 1000)
        for strength in (-.1, 1.1, float("nan")):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                centering.soft_center(matrix, [0, 0], strength, 1000)
        with self.assertRaises(MemoryError):
            centering.soft_center(matrix, [0, 0], .5, 1)


class InferenceBoundaryTests(unittest.TestCase):
    def test_cluster_partition_is_id_order_invariant(self):
        texts = np.array(["apple pear orchard", "apple pear orchard",
                          "quantum orbital physics", "quantum orbital physics",
                          "the and or"])
        ids = np.array(["d", "a", "c", "b", "e"])
        groups = centering.cluster_batch(texts, ids, .65, 100)
        order = np.array([4, 2, 0, 3, 1])
        shuffled = centering.cluster_batch(texts[order], ids[order], .65, 100)
        inverse = np.argsort(order)
        # Numeric labels need not match; the partition must match exactly.
        np.testing.assert_array_equal(groups[:, None] == groups[None, :],
            shuffled[inverse, None] == shuffled[inverse][None, :])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[0], groups[2])
        self.assertEqual(np.sum(groups == groups[4]), 1)

    def test_zero_texts_singletons_and_no_oracle_arguments(self):
        for texts in ([], ["the"], ["the and", "and the", "or"]):
            groups = centering.cluster_batch(texts, [str(i) for i in range(len(texts))], .65, 100)
            self.assertEqual(len(np.unique(groups)), len(texts))
        self.assertEqual(list(inspect.signature(centering.cluster_batch).parameters),
                         ["texts", "ids", "threshold", "max_features"])
        with self.assertRaises(ValueError):
            centering.cluster_batch(["apple", "pear"], ["same", "same"], .65, 100)

    def test_half_selection_uses_id_identity_not_input_position(self):
        ids = np.array([f"row-{i}" for i in range(7)])
        selected = centering.half_batch_indices(ids)
        order = np.array([4, 1, 6, 0, 3, 5, 2])
        shuffled = centering.half_batch_indices(ids[order])
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(ids[selected]), set(ids[order][shuffled]))
        self.assertEqual(centering.half_batch_indices([]).size, 0)

    def test_supervised_features_fit_training_text_only(self):
        training = np.array(["alpha beta alpha", "alpha gamma alpha", "beta gamma"])
        validation = np.array(["validationonlytoken zzzzz"])
        config = {"char_max_features": 1000, "word_max_features": 1000, "word_weight": .5}
        created = []
        def factory(**kwargs):
            vectorizer = TfidfVectorizer(**kwargs)
            created.append(vectorizer)
            return vectorizer
        with patch.object(centering, "TfidfVectorizer", side_effect=factory):
            train, valid = centering.feature_pair(training, validation, config)
        second_train, _ = centering.feature_pair(training, ["entirelydifferentvalue"], config)
        np.testing.assert_allclose(train.toarray(), second_train.toarray())
        self.assertEqual(train.dtype, np.float64)
        self.assertEqual(train.shape[1], valid.shape[1])
        self.assertNotIn("validationonlytoken", created[1].vocabulary_)
        self.assertTrue(all("z" not in token for token in created[0].vocabulary_))
        np.testing.assert_allclose(np.asarray(train.multiply(train).sum(axis=1)).ravel(), 1.)


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = {"adoption_min_delta": .005, "adoption_min_positive_folds": 4,
                       "stress_min_delta": -.005}
        self.result = {"delta_vs_original": .006, "paired_prompt_ci95": [.001, .01],
                       "positive_folds": 4, "stress": {
                           "half": {"delta_vs_original": -.005},
                           "singleton": {"delta_vs_original": 0.}}}

    def test_gate_rejects_any_failed_component(self):
        self.assertTrue(centering.gate(self.result, self.config))
        for key, value in (("delta_vs_original", .004), ("paired_prompt_ci95", [1e-12, .01]),
                           ("positive_folds", 3)):
            result = copy.deepcopy(self.result)
            result[key] = value
            self.assertFalse(centering.gate(result, self.config))
        for mode in ("half", "singleton"):
            result = copy.deepcopy(self.result)
            result["stress"][mode]["delta_vs_original"] = -.0051
            self.assertFalse(centering.gate(result, self.config))

    def test_gate_rejects_missing_stress_evidence(self):
        for modes in ({}, {"half": {"delta_vs_original": 0.}}):
            result = copy.deepcopy(self.result)
            result["stress"] = modes
            self.assertFalse(centering.gate(result, self.config))


if __name__ == "__main__":
    unittest.main()
