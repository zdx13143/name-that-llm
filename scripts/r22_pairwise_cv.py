"""Frozen R22 one-shot score-bank evaluation; see docs/R22_PLAN_20261104.md."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

from r19_cpu_cv import metrics
from r19_svc_component import R16_LABELS

SCORES = Path('D:/kaggle/name_that_llm/outputs/worktrees/r21-scorebank-20261103/outputs/r21/kaggle_out_v4/r21_out/scores.npz')
DEV = Path('kaggle/r21_dataset/development.json')
OUT = Path('outputs/r22/result.json')
SCORE_HASH = '85a4a8b7faf931a9d02dd4e6b1a3082487bd8e5aabb96b0498358a835d0c2393'
DEV_HASH = '1c9446efa6c8c477bd68448f1de212aadcc8dc90224b76728ec2f70e2ec2334f'
GRID = (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FLOOR = 0.509118
A = R16_LABELS.index('gemini-3.7-flash')
B = R16_LABELS.index('gemini-3.8-flash')


def checked_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('incorrect input hash: ' + str(path))
    return actual


def validate(data, bank):
    labels = data['labels']
    if len(labels) != 18 or set(labels) != set(R16_LABELS):
        raise ValueError('label mismatch')
    truth, folds, groups = (np.asarray(data[k]) for k in ('truth', 'fold_id', 'groups'))
    n = len(truth)
    if n != len(folds) or n != len(groups) or not np.array_equal(bank['row_index'], np.arange(n)):
        raise ValueError('row alignment mismatch')
    # development row_index holds source-row ids (strictly increasing subset), not positions.
    source_rows = np.asarray(data['row_index'])
    if len(source_rows) != n or not np.all(np.diff(source_rows) > 0):
        raise ValueError('development row alignment mismatch')
    if not np.array_equal(bank['truth'], truth) or not np.array_equal(bank['fold'], folds) or not np.array_equal(bank['group'], groups):
        raise ValueError('fold, truth or group mismatch')
    if set(folds) != set(range(5)) or set(truth) != set(range(18)):
        raise ValueError('fold or class mismatch')
    for f in range(5):
        mask = folds == f
        if set(groups[mask]) & set(groups[~mask]):
            raise ValueError('group leakage')
        if set(truth[mask]) != set(range(18)) or set(truth[~mask]) != set(range(18)):
            raise ValueError('missing class in fold')
    for condition in ('r16', 'areeb'):
        for member in ('tfidf', 'distort', 'svc4', 'blend', 'balanced'):
            scores = bank[f'{condition}_{member}']
            if scores.shape != (n, 18) or not np.isfinite(scores).all():
                raise ValueError('invalid or nonfinite scores')
    return truth, folds, groups, np.array([labels.index(label) for label in R16_LABELS])


def adjust(scores, threshold):
    pred = scores.argmax(axis=1)
    mask = (pred == B) & ((scores[:, B] - scores[:, A]) < threshold)
    pred[mask] = A
    return pred


def select(scores, truth, baseline, mapping, grid=GRID):
    rec0 = recall_score(truth, baseline, labels=np.arange(18), average=None, zero_division=0)
    options = []
    for threshold in sorted(grid):
        pred = mapping[adjust(scores, threshold)]
        recall = recall_score(truth, pred, labels=np.arange(18), average=None, zero_division=0)
        feasible = bool(np.all(rec0 - recall <= 0.05))
        options.append((threshold, balanced_accuracy_score(truth, pred), feasible))
    eligible = [o for o in options if o[2]] or options
    return max(eligible, key=lambda o: (o[1], -o[0]))[0]


def crossfit(scores, truth, baseline, folds, mapping):
    result = np.empty(len(truth), dtype=int)
    lambdas = []
    for f in range(5):
        train, valid = folds != f, folds == f
        threshold = select(scores[train], truth[train], baseline[train], mapping)
        lambdas.append(threshold)
        result[valid] = mapping[adjust(scores[valid], threshold)]
    return result, lambdas


def evaluate(data, bank, hashes):
    truth, folds, groups, mapping = validate(data, bank)
    baseline = mapping[bank['r16_balanced'].argmax(axis=1)]
    scores = bank['areeb_balanced']
    unadjusted = mapping[scores.argmax(axis=1)]
    candidate, lambdas = crossfit(scores, truth, baseline, folds, mapping)
    diagnostic = metrics(truth, baseline, unadjusted, folds, groups)
    if round(float(diagnostic['delta_ba']), 6) != 0.029147:
        raise ValueError('R21 diagnostic reproduction failed')
    gate = metrics(truth, baseline, candidate, folds, groups)
    gate['r19_adopt'] = gate['adopt']
    gate['floor_pass'] = bool(gate['candidate_ba'] >= FLOOR)
    gate['adopt'] = bool(gate['r19_adopt'] and gate['floor_pass'])
    labels = data['labels']
    a, b = labels.index('gemini-3.7-flash'), labels.index('gemini-3.8-flash')
    counts = lambda p: {label: int(np.sum(p == i)) for i, label in enumerate(labels)}
    confusion = lambda p: int(np.sum((truth == a) & (p == b)))
    return {'input_hashes': hashes, 'labels_development': labels, 'labels_scores': R16_LABELS,
            'grid': GRID, 'chosen_lambdas': lambdas, 'gate': gate, 'diagnostic': diagnostic,
            'per_class_recall': {'baseline': gate['baseline_recall'], 'unadjusted': diagnostic['candidate_recall'], 'candidate': gate['candidate_recall']},
            'confusion_3_7_to_3_8': {'baseline': confusion(baseline), 'unadjusted': confusion(unadjusted), 'candidate': confusion(candidate)},
            'adjusted_row_count': int(np.sum(candidate != unadjusted)),
            'predicted_class_counts': {'before': counts(unadjusted), 'after': counts(candidate)},
            'predictions': {'row_index': list(range(len(truth))), 'baseline': baseline.tolist(), 'unadjusted': unadjusted.tolist(), 'candidate': candidate.tolist()}}


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run', action='store_true')
    action.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.synthetic:
        scores = np.zeros((2, 18)); scores[:, B] = 1; scores[:, A] = [0, 0.5]
        assert np.array_equal(adjust(scores, 0), scores.argmax(axis=1))
        assert np.array_equal(adjust(scores, 1), [B, A])
        print('synthetic smoke passed')
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    hashes = {'scores_sha256': checked_hash(SCORES, SCORE_HASH), 'development_sha256': checked_hash(DEV, DEV_HASH)}
    data = json.loads(DEV.read_text(encoding='utf-8'))
    with np.load(SCORES, allow_pickle=False) as bank:
        result = evaluate(data, bank, hashes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
