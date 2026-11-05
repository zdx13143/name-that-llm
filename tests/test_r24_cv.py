import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import r22_pairwise_cv as r22
import r24_cv as cv
from r19_cpu_cv import metrics


class CrossFitTest(unittest.TestCase):
    def setUp(self):
        self.truth = np.tile(np.arange(18), 10)
        self.folds = np.repeat(np.arange(5), 36)
        self.base = self.truth.copy()

    def test_selection_isolated_and_failure(self):
        predictions = {a: self.truth.copy() for a in cv.GRID}
        _, chosen = cv.crossfit(predictions, self.truth, self.base, self.folds)
        changed_truth = self.truth.copy(); changed_truth[self.folds == 0] = (changed_truth[self.folds == 0] + 1) % 18
        _, altered = cv.crossfit(predictions, changed_truth, self.base, self.folds)
        self.assertEqual(chosen[0], altered[0])
        bad = {a: (self.truth + 1) % 18 for a in cv.GRID}
        _, choices = cv.crossfit(bad, self.truth, self.base, self.folds)
        self.assertEqual(choices, [None] * 5)
        self.assertFalse(cv.adoption({'adopt': True, 'floor_pass': True}, {'pass': True}, choices))

    def test_label_mapping_and_r22_reproduction_path(self):
        rng = np.random.default_rng(2)
        scores = rng.normal(size=(180, 18))
        mapping = np.arange(18)[::-1]
        pred, lambdas = r22.crossfit(scores, self.truth, self.base, self.folds, mapping)
        saved = {'chosen_lambdas': lambdas, 'predictions': {'candidate': pred.tolist()},
                 'gate': {'candidate_ba': float(np.mean(pred == self.truth))}}
        with patch.object(cv, 'EXPECTED_LAMBDAS', lambdas), patch('builtins.round', return_value=.523284):
            reproduced = cv.reproduce_r22(scores, self.truth, self.base, self.folds, mapping, saved)
        np.testing.assert_array_equal(reproduced, pred)
        saved['predictions']['candidate'][0] = (saved['predictions']['candidate'][0] + 1) % 18
        with patch.object(cv, 'EXPECTED_LAMBDAS', lambdas):
            with self.assertRaises(ValueError):
                cv.reproduce_r22(scores, self.truth, self.base, self.folds, mapping, saved)

    def test_bootstrap_pairing(self):
        groups = np.repeat(np.arange(10), 18)
        result = metrics(self.truth, self.base, self.base.copy(), self.folds, groups)
        self.assertEqual(result['bootstrap_95'], [0., 0.])
        wrong = (self.truth + 1) % 18
        forward = metrics(self.truth, wrong, self.base, self.folds, groups)
        reverse = metrics(self.truth, self.base, wrong, self.folds, groups)
        np.testing.assert_allclose(forward['bootstrap_95'], -np.array(reverse['bootstrap_95'])[::-1])

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            cv.write_result(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                cv.write_result(path, {'value': 2})
            self.assertIn('1', path.read_text())


if __name__ == '__main__':
    unittest.main()
