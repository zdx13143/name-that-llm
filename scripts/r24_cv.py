"""Frozen R24; --synthetic never opens the development inputs."""
import argparse
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, recall_score
import psutil

import r22_pairwise_cv as r22
from r19_cpu_cv import metrics
from r24_assignment import predict, scale_scores
from r24_cluster import cluster

GRID = (0., .25, .5, 1., 2.)
R22_RESULT = Path('outputs/r22/result.json')
R22_HASH = '78b734fc25a4ae43ced881906cd73fd31730baf1be63266fd111c4d02377eff9'
OUT = Path('outputs/r24/result.json')
EXPECTED_LAMBDAS = [1., 1.5, 1., 1., 1.]


def select(predictions, truth, baseline, mask):
    labels = np.arange(18)
    base = recall_score(truth[mask], baseline[mask], labels=labels, average=None, zero_division=0)
    candidates = []
    for alpha in GRID:
        rec = recall_score(truth[mask], predictions[alpha][mask], labels=labels, average=None, zero_division=0)
        if np.all(base - rec <= .05 + 1e-12):
            candidates.append((float(rec.mean()), alpha))
    return max(candidates, key=lambda x: (x[0], -x[1]))[1] if candidates else None


def crossfit(predictions, truth, baseline, folds):
    result = predictions[0.].copy()
    choices = []
    for f in range(5):
        alpha = select(predictions, truth, baseline, folds != f)
        choices.append(alpha)
        if alpha is not None:
            result[folds == f] = predictions[alpha][folds == f]
    return result, choices


def adoption(r16_gate, r22_comparison, choices):
    return bool(r16_gate['adopt'] and r16_gate['floor_pass'] and
                all(a is not None for a in choices) and r22_comparison['pass'])


def quality(groups, clusters):
    n = len(groups)
    purity = sum(Counter(groups[clusters == c]).most_common(1)[0][1] for c in np.unique(clusters)) / n
    completeness = sum(Counter(clusters[groups == g]).most_common(1)[0][1] for g in np.unique(groups)) / n
    return {'ari': float(adjusted_rand_score(groups, clusters)), 'purity': float(purity),
            'completeness': float(completeness)}


def reproduce_r22(scores, truth, baseline, folds, mapping, saved):
    pred, lambdas = r22.crossfit(scores, truth, baseline, folds, mapping)
    if lambdas != EXPECTED_LAMBDAS or saved['chosen_lambdas'] != lambdas:
        raise ValueError('R22 lambda reproduction failed')
    if not np.array_equal(pred, saved['predictions']['candidate']):
        raise ValueError('R22 prediction reproduction failed')
    recall = recall_score(truth, pred, labels=np.arange(18), average=None, zero_division=0)
    ba = float(recall.mean())
    if round(ba, 6) != .523284 or abs(ba - saved['gate']['candidate_ba']) > 1e-12:
        raise ValueError('R22 BA reproduction failed')
    return pred


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


def evaluate(data, bank, saved, hashes, check):
    truth, folds, groups, mapping = r22.validate(data, bank)
    n = len(truth)
    if n != 2159 or len(np.unique(groups)) != 120 or len(data['texts']) != n:
        raise ValueError('unexpected development dimensions')
    baseline = mapping[bank['r16_balanced'].argmax(axis=1)]
    scores = bank['areeb_balanced']
    unadjusted = mapping[scores.argmax(axis=1)]
    diagnostic = metrics(truth, baseline, unadjusted, folds, groups)
    if round(float(diagnostic['delta_ba']), 6) != .029147:
        raise ValueError('R21 diagnostic reproduction failed')
    r22_pred = reproduce_r22(scores, truth, baseline, folds, mapping, saved)
    check()
    clusters, fallback = cluster(data['texts'], check)
    z, scales = scale_scores(scores, folds)
    predictions = {a: mapping[predict(z, clusters, a, check)] for a in GRID}
    candidate, choices = crossfit(predictions, truth, baseline, folds)
    gate = metrics(truth, baseline, candidate, folds, groups)
    gate['floor_pass'] = bool(gate['candidate_ba'] >= r22.FLOOR)
    gate['selection_pass'] = all(a is not None for a in choices)
    gate['adopt'] = bool(gate['adopt'] and gate['floor_pass'] and gate['selection_pass'])
    comparison = metrics(truth, r22_pred, candidate, folds, groups)
    comparison['pass'] = bool(comparison['delta_ba'] >= .005 and comparison['bootstrap_95'][0] > 0)
    adopted = adoption(gate, comparison, choices)
    deploy = select(predictions, truth, baseline, np.ones(n, dtype=bool)) if adopted else None
    if adopted and deploy is None:
        adopted = False
    oracle = scores.argmax(axis=1)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        rr, cc = linear_sum_assignment(-scores[idx])
        oracle[idx[rr]] = cc
    oracle = mapping[oracle]
    hist = Counter(np.unique(clusters, return_counts=True)[1].tolist())
    fold_quality = []
    for f in range(5):
        mask = folds == f
        sub, sub_fallback = cluster(np.asarray(data['texts'], dtype=object)[mask], check)
        q = quality(groups[mask], sub)
        fold_quality.append({'fold': f, 'ari': q['ari'], 'purity': q['purity'], 'fallback': sub_fallback})
    check()
    labels = data['labels']
    counts = lambda p: {label: int(np.sum(p == i)) for i, label in enumerate(labels)}
    per_fold_ba = {str(f): {name: float(recall_score(truth[folds == f], p[folds == f], labels=np.arange(18), average=None, zero_division=0).mean())
                             for name, p in [('baseline', baseline), ('r22', r22_pred), ('candidate', candidate)]}
                   for f in range(5)}
    a, b = labels.index('gemini-3.7-flash'), labels.index('gemini-3.8-flash')
    result = {'input_hashes': hashes, 'labels_development': labels, 'labels_scores': r22.R16_LABELS,
            'grid': GRID, 'fold_scales': scales, 'chosen_alphas': choices, 'adopted': adopted,
            'r16_gate': gate, 'r22_comparison': comparison, 'per_fold_ba': per_fold_ba,
            'r21_reproduction': diagnostic, 'r22_lambdas': EXPECTED_LAMBDAS,
            'per_class_recall': {'baseline': gate['baseline_recall'], 'unadjusted': diagnostic['candidate_recall'],
                                 'r22': comparison['baseline_recall'], 'candidate': gate['candidate_recall']},
            'changed_rows_vs_unadjusted': int(np.sum(candidate != unadjusted)),
            'changed_rows_vs_r22': int(np.sum(candidate != r22_pred)),
            'predicted_class_counts': {'unadjusted': counts(unadjusted), 'r22': counts(r22_pred), 'candidate': counts(candidate)},
            'confusion_3_7_to_3_8': {name: int(np.sum((truth == a) & (p == b))) for name, p in [('baseline', baseline), ('r22', r22_pred), ('candidate', candidate)]},
            'cluster_size_histogram': {str(k): v for k, v in sorted(hist.items())},
            'cross_fold_cluster_count': int(sum(len(np.unique(folds[clusters == c])) > 1 for c in np.unique(clusters))),
            'primary_cluster_quality': dict(quality(groups, clusters), fallback=fallback),
            'per_fold_cluster_diagnostics': fold_quality,
            'oracle_true_group_hungarian_diagnostic_no_authority': {'ba': float(recall_score(truth, oracle, labels=np.arange(18), average=None, zero_division=0).mean())},
            'predictions': {'row_index': list(range(n)), 'baseline': baseline.tolist(), 'unadjusted': unadjusted.tolist(),
                            'r22': r22_pred.tolist(), 'candidate': candidate.tolist()}}
    if adopted:
        result['deployment_alpha'] = deploy
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.synthetic:
        rng = np.random.default_rng(24)
        scores = rng.normal(size=(24, 18)); folds = np.arange(24) % 5
        z, _ = scale_scores(scores, folds)
        p = predict(z, np.repeat([0, 1], 12), 1.)
        assert p.shape == (24,) and np.array_equal(predict(z, np.zeros(24), 0.), scores.argmax(axis=1))
        print('R24 synthetic smoke passed')
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic()
    process = psutil.Process()
    peak = [0]
    finished = threading.Event()
    def watch():
        while not finished.wait(.1):
            rss = process.memory_info().rss
            peak[0] = max(peak[0], rss)
            if rss > 4 * 1024**3 or time.monotonic() - start > 1800:
                os._exit(2)
    threading.Thread(target=watch, daemon=True).start()
    def check():
        if time.monotonic() - start > 1800 or process.memory_info().rss > 4 * 1024**3:
            raise RuntimeError('resource limit exceeded')
    hashes = {'scores_sha256': r22.checked_hash(r22.SCORES, r22.SCORE_HASH),
              'development_sha256': r22.checked_hash(r22.DEV, r22.DEV_HASH),
              'r22_result_sha256': r22.checked_hash(R22_RESULT, R22_HASH)}
    data = json.loads(r22.DEV.read_text(encoding='utf-8'))
    saved = json.loads(R22_RESULT.read_text(encoding='utf-8'))
    with np.load(r22.SCORES, allow_pickle=False) as bank:
        result = evaluate(data, bank, saved, hashes, check)
    check()
    result['runtime_seconds'] = time.monotonic() - start
    result['peak_rss_bytes'] = max(peak[0], process.memory_info().rss)
    write_result(OUT, result)
    finished.set()
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
