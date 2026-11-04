"""Synthetic-only tests for the frozen R22 rule."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import r22_pairwise_cv as r22


def fixture():
    truth = np.tile(np.arange(18), 5)
    folds = np.repeat(np.arange(5), 18)
    groups = np.array([f'g{f}' for f in folds])
    scores = np.full((90, 18), -2.0)
    scores[np.arange(90), truth] = 1.0
    data = {'labels': list(r22.R16_LABELS), 'truth': truth.tolist(), 'fold_id': folds.tolist(),
            'groups': groups.tolist(), 'row_index': list(range(90))}
    bank = {'row_index': np.arange(90), 'truth': truth.copy(), 'fold': folds.copy(), 'group': groups.copy()}
    for condition in ('r16', 'areeb'):
        for member in ('tfidf', 'distort', 'svc4', 'blend', 'balanced'):
            bank[f'{condition}_{member}'] = scores.copy()
    return data, bank


class R22Tests(unittest.TestCase):
    def test_identity_direction_and_strict_threshold(self):
        scores = np.full((4, 18), -5.0)
        scores[:, r22.B] = [1, 1, 1, 0]
        scores[:, r22.A] = [1, 0.5, 0, 0]
        scores[3, 0] = 2
        before = scores.argmax(axis=1)
        np.testing.assert_array_equal(r22.adjust(scores, 0), before)
        after = r22.adjust(scores, 1)
        np.testing.assert_array_equal(after, [r22.A, r22.A, r22.B, 0])
        self.assertTrue(np.all((before[after != before] == r22.B) & (after[after != before] == r22.A)))

    def test_tie_smallest_and_feasibility(self):
        data, bank = fixture()
        truth = np.array(data['truth']); scores = bank['areeb_balanced']; mapping = np.arange(18)
        baseline = truth.copy()
        self.assertEqual(r22.select(scores, truth, baseline, mapping), 0)
        # Improvement in BA is infeasible because it sacrifices all class 3.8 recall.
        idx = np.flatnonzero(truth == r22.A)
        scores[idx, r22.B] = 1.1
        scores[:, r22.A][truth == r22.B] = 0.9
        self.assertEqual(r22.select(scores, truth, baseline, mapping), 0)
        # If baseline already misses 3.8, adjusting can be feasible and improve BA.
        baseline[truth == r22.B] = r22.A
        self.assertEqual(r22.select(scores, truth, baseline, mapping), 0.25)

    def test_outer_labels_cannot_change_outer_lambda(self):
        data, bank = fixture(); truth = np.array(data['truth']); folds = np.array(data['fold_id'])
        scores = bank['areeb_balanced']; baseline = truth.copy(); mapping = np.arange(18)
        _, original = r22.crossfit(scores, truth, baseline, folds, mapping)
        changed = truth.copy(); changed[folds == 2] = (changed[folds == 2] + 1) % 18
        _, again = r22.crossfit(scores, changed, baseline, folds, mapping)
        self.assertEqual(original[2], again[2])

    def test_label_permutation(self):
        data, bank = fixture(); truth, folds, _, mapping = r22.validate(data, bank)
        baseline = mapping[bank['r16_balanced'].argmax(axis=1)]
        pred, lambdas = r22.crossfit(bank['areeb_balanced'], truth, baseline, folds, mapping)
        permutation = np.arange(18)[::-1]
        data['labels'] = [data['labels'][i] for i in permutation]
        data['truth'] = permutation[truth].tolist(); bank['truth'] = permutation[truth]
        new_truth, _, _, new_mapping = r22.validate(data, bank)
        new_pred, new_lambdas = r22.crossfit(bank['areeb_balanced'], new_truth, permutation[baseline], folds, new_mapping)
        np.testing.assert_array_equal(new_pred, permutation[pred])
        self.assertEqual(lambdas, new_lambdas)

    def test_bad_hash_group_leak_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input'; path.write_bytes(b'synthetic')
            with self.assertRaisesRegex(ValueError, 'hash'):
                r22.checked_hash(path, 'bad')
        data, bank = fixture()
        data['groups'][18] = data['groups'][0]; bank['group'][18] = bank['group'][0]
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            r22.validate(data, bank)
        data, bank = fixture(); bank['areeb_balanced'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            r22.validate(data, bank)


if __name__ == '__main__':
    unittest.main()
