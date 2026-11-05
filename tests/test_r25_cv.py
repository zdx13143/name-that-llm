import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import r25_cv as cv
import r24_cv as r24
from r19_cpu_cv import metrics


class CVTest(unittest.TestCase):
    def setUp(self):
        self.truth = np.tile(np.arange(18), 10)
        self.folds = np.repeat(np.arange(5), 36)
        self.base = self.truth.copy()

    def test_tiebreak_and_fold_isolation(self):
        predictions = {p: self.truth.copy() for p in cv.GRID}
        self.assertEqual(cv.select(predictions, self.truth, self.base, self.folds != 0), (0., 0.))
        predictions[(0., 0.)] = (self.truth + 1) % 18
        self.assertEqual(cv.select(predictions, self.truth, self.base, self.folds != 0), (0., .5))
        _, choices, _ = cv.crossfit(predictions, self.truth, self.base, self.folds)
        altered = self.truth.copy(); altered[self.folds == 0] = (altered[self.folds == 0] + 1) % 18
        _, choices2, _ = cv.crossfit(predictions, altered, self.base, self.folds)
        self.assertEqual(choices[0], choices2[0])

    def test_full_development_pair_matches_fixed_rule(self):
        rng = np.random.default_rng(91)
        z = rng.normal(size=(180, 18))
        clusters = np.repeat(np.arange(15), 12)  # clusters span the synthetic folds
        predictions = {pair: cv.predict(z, clusters, *pair, cv.r22.A, cv.r22.B) for pair in cv.GRID}
        baseline = predictions[(0., 0.)]
        pair = cv.select(predictions, self.truth, baseline, np.ones(len(self.truth), dtype=bool))
        self.assertIsNotNone(pair)
        np.testing.assert_array_equal(predictions[pair], cv.predict(z, clusters, *pair, cv.r22.A, cv.r22.B))

    def test_failure_has_no_concatenated_predictions(self):
        predictions = {p: (self.truth + 1) % 18 for p in cv.GRID}
        candidate, choices, failures = cv.crossfit(predictions, self.truth, self.base, self.folds)
        self.assertIsNone(candidate)
        self.assertEqual(choices, [None] * 5)
        self.assertEqual(failures, list(range(5)))

    def test_evaluate_failure_suppresses_primary_ba(self):
        n = 2159
        truth = np.arange(n) % 18; folds = np.arange(n) % 5; groups = np.arange(n) % 120
        mapping = np.arange(18); scores = np.zeros((n, 18))
        data = {'texts': ['synthetic'] * n, 'labels': cv.r22.R16_LABELS}
        bank = {'r16_balanced': scores, 'areeb_balanced': scores}
        saved = {'predictions': {'row_index': list(range(n)), 'candidate': [0] * n}}
        choices = [(0., 0.), None, (0., 0.), (0., 0.), (0., 0.)]
        with patch.object(cv.r22, 'validate', return_value=(truth, folds, groups, mapping)), \
             patch.object(cv, 'metrics', return_value={'delta_ba': .029147}), \
             patch.object(cv.r24, 'reproduce_r22', return_value=np.zeros(n, dtype=int)), \
             patch.object(cv, 'cluster', return_value=(np.arange(n), False)), \
             patch.object(cv, 'scale_scores', return_value=(scores, {})), \
             patch.object(cv, 'predict', return_value=np.zeros(n, dtype=int)), \
             patch.object(cv.r24, 'quality', return_value={}), \
             patch.object(cv, 'crossfit', return_value=(None, choices, [1])):
            result = cv.evaluate(data, bank, {}, saved, {}, lambda: None)
        self.assertFalse(result['adopted'])
        self.assertFalse(result['r16_gate']['adopt'])
        self.assertEqual(result['selection_failures'], [1])
        self.assertNotIn('candidate_ba', result['r16_gate'])
        self.assertNotIn('predictions', result)

    def test_hash_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / str(i) for i in range(4)]
            for path in paths:
                path.write_bytes(b'fixture')
            digest = hashlib.sha256(b'fixture').hexdigest()
            with patch.multiple(cv.r22, SCORES=paths[0], DEV=paths[1], SCORE_HASH=digest, DEV_HASH=digest), patch.multiple(cv.r24, R22_RESULT=paths[2], R22_HASH=digest), patch.multiple(cv, R24_RESULT=paths[3], R24_HASH=digest):
                self.assertEqual(len(cv.checked_inputs()), 4)
                for path in paths[2:]:
                    path.write_bytes(b'bad')
                    with self.assertRaises(ValueError): cv.checked_inputs()
                    path.write_bytes(b'fixture')

    def test_r24_saved_validation_and_exclusive_output(self):
        saved = {'predictions': {'row_index': [0, 1], 'candidate': [1, 2]}}
        np.testing.assert_array_equal(cv.saved_r24_predictions(saved, 2), [1, 2])
        saved['predictions']['row_index'] = [1, 0]
        with self.assertRaises(ValueError): cv.saved_r24_predictions(saved, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            r24.write_result(path, {'value': 1})
            with self.assertRaises(FileExistsError): r24.write_result(path, {'value': 2})
            self.assertEqual(json.loads(path.read_text()), {'value': 1})

    def test_bootstrap_pairing(self):
        groups = np.repeat(np.arange(10), 18)
        wrong = (self.truth + 1) % 18
        forward = metrics(self.truth, wrong, self.base, self.folds, groups)
        reverse = metrics(self.truth, self.base, wrong, self.folds, groups)
        np.testing.assert_allclose(forward['bootstrap_95'], -np.array(reverse['bootstrap_95'])[::-1])


if __name__ == '__main__':
    unittest.main()
