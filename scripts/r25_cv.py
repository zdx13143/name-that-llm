"""Frozen R25 decision procedure. --synthetic does not open development inputs."""
import argparse
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics import recall_score

import r22_pairwise_cv as r22
import r24_cv as r24
from r19_cpu_cv import metrics
from r24_assignment import scale_scores
from r24_cluster import cluster
from r25_assignment import predict

ALPHAS = (0., .5, 1., 2.)
BETAS = (0., .5, 1., 1.5)
GRID = tuple((a, b) for a in ALPHAS for b in BETAS)
R24_RESULT = Path('outputs/r24/result.json')
R24_HASH = 'edaed1c53b7d9a8a9c22db4de31e8e93312e34fec738e9459ec9ad89814e1a4c'
OUT = Path('outputs/r25/result.json')


def checked_inputs():
    return {'scores_sha256': r22.checked_hash(r22.SCORES, r22.SCORE_HASH),
            'development_sha256': r22.checked_hash(r22.DEV, r22.DEV_HASH),
            'r22_result_sha256': r22.checked_hash(r24.R22_RESULT, r24.R22_HASH),
            'r24_result_sha256': r22.checked_hash(R24_RESULT, R24_HASH)}


def select(predictions, truth, baseline, mask):
    labels = np.arange(18)
    if set(truth[mask]) != set(labels):
        raise ValueError('selection sample missing a class')
    base = recall_score(truth[mask], baseline[mask], labels=labels, average=None, zero_division=0)
    options = []
    for pair in GRID:
        rec = recall_score(truth[mask], predictions[pair][mask], labels=labels, average=None, zero_division=0)
        if np.all(base - rec <= .05 + 1e-12):
            options.append((float(rec.mean()), pair))
    return max(options, key=lambda x: (x[0], -x[1][0], -x[1][1]))[1] if options else None


def crossfit(predictions, truth, baseline, folds):
    choices = [select(predictions, truth, baseline, folds != f) for f in range(5)]
    failures = [f for f, pair in enumerate(choices) if pair is None]
    if failures:
        return None, choices, failures
    result = np.empty(len(truth), dtype=int)
    for f, pair in enumerate(choices):
        result[folds == f] = predictions[pair][folds == f]
    return result, choices, failures


def saved_r24_predictions(saved, n):
    p = saved['predictions']
    candidate = np.asarray(p['candidate'])
    if p['row_index'] != list(range(n)) or candidate.shape != (n,) or not np.issubdtype(candidate.dtype, np.integer) or np.any((candidate < 0) | (candidate >= 18)):
        raise ValueError('invalid R24 saved predictions')
    return candidate


def evaluate(data, bank, saved22, saved24, hashes, check):
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
    r22_pred = r24.reproduce_r22(scores, truth, baseline, folds, mapping, saved22)
    r24_pred = saved_r24_predictions(saved24, n)
    check()
    clusters, fallback = cluster(data['texts'], check)
    z, scales = scale_scores(scores, folds)
    a, b = r22.R16_LABELS.index('gemini-3.7-flash'), r22.R16_LABELS.index('gemini-3.8-flash')
    predictions = {pair: mapping[predict(z, clusters, *pair, a, b, check)] for pair in GRID}
    candidate, choices, failures = crossfit(predictions, truth, baseline, folds)
    hist = Counter(np.unique(clusters, return_counts=True)[1].tolist())
    result = {'input_hashes': hashes, 'labels_development': data['labels'], 'labels_scores': r22.R16_LABELS,
              'grid': GRID, 'fold_scales': scales, 'chosen_pairs': choices, 'selection_failures': failures,
              'selection_pass': not failures, 'adopted': False, 'r21_reproduction': diagnostic,
              'r22_lambdas': r24.EXPECTED_LAMBDAS,
              'cluster_size_histogram': {str(k): v for k, v in sorted(hist.items())},
              'cross_fold_cluster_count': int(sum(len(np.unique(folds[clusters == c])) > 1 for c in np.unique(clusters))),
              'cross_fold_cluster_row_count': int(sum(np.sum(clusters == c) for c in np.unique(clusters) if len(np.unique(folds[clusters == c])) > 1)),
              'primary_cluster_quality': dict(r24.quality(groups, clusters), fallback=fallback)}
    if failures:
        result['r16_gate'] = {'adopt': False, 'selection_pass': False}
        return result
    gate = metrics(truth, baseline, candidate, folds, groups)
    gate['floor_pass'] = bool(gate['candidate_ba'] >= r22.FLOOR)
    gate['adopt'] = bool(gate['adopt'] and gate['floor_pass'])
    comparison = metrics(truth, r22_pred, candidate, folds, groups)
    comparison['pass'] = bool(comparison['delta_ba'] >= .005 and comparison['bootstrap_95'][0] > 0)
    descriptive = metrics(truth, r24_pred, candidate, folds, groups)
    adopted = gate['adopt'] and comparison['pass']
    deployment = select(predictions, truth, baseline, np.ones(n, dtype=bool)) if adopted else None
    adopted = bool(adopted and deployment is not None)
    labels = data['labels']
    da, db = labels.index('gemini-3.7-flash'), labels.index('gemini-3.8-flash')
    variants = {'baseline': baseline, 'unadjusted': unadjusted, 'r22': r22_pred, 'r24': r24_pred, 'candidate': candidate}
    result.update({'adopted': adopted, 'r16_gate': gate, 'r22_comparison': comparison,
                   'r24_descriptive_comparison': descriptive,
                   'per_fold_ba': {str(f): {name: float(recall_score(truth[folds == f], p[folds == f], labels=np.arange(18), average=None, zero_division=0).mean()) for name, p in variants.items()} for f in range(5)},
                   'per_class_recall': {name: recall_score(truth, p, labels=np.arange(18), average=None, zero_division=0).tolist() for name, p in variants.items()},
                   'per_class_recall_drop_vs_r16': (np.asarray(gate['baseline_recall']) - np.asarray(gate['candidate_recall'])).tolist(),
                   'confusion_gemini': {name: {'3.7_to_3.8': int(np.sum((truth == da) & (p == db))), '3.8_to_3.7': int(np.sum((truth == db) & (p == da)))} for name, p in variants.items()},
                   'changed_rows_vs_unadjusted': int(np.sum(candidate != unadjusted)),
                   'changed_rows_vs_r22': int(np.sum(candidate != r22_pred)),
                   'predicted_class_counts': {name: {label: int(np.sum(p == i)) for i, label in enumerate(labels)} for name, p in variants.items()},
                   'predictions': {'row_index': list(range(n)), **{name: p.tolist() for name, p in variants.items()}}})
    if adopted:
        result['deployment_pair'] = deployment
    elif gate['adopt'] and comparison['pass']:
        result['deployment_selection_failed'] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.synthetic:
        rng = np.random.default_rng(25)
        raw = rng.normal(size=(24, 18)); folds = np.arange(24) % 5
        z, _ = scale_scores(raw, folds)
        groups = np.repeat([0, 1], 12)
        p = predict(z, groups, 0., .5, r22.A, r22.B)
        from r25_assignment import biased_scores
        assert np.array_equal(p, biased_scores(z, .5, r22.A, r22.B).argmax(axis=1))
        print('R25 synthetic smoke passed')
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic(); process = psutil.Process(); peak = [process.memory_info().rss]
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
    hashes = checked_inputs()
    data = json.loads(r22.DEV.read_text(encoding='utf-8'))
    saved22 = json.loads(r24.R22_RESULT.read_text(encoding='utf-8'))
    saved24 = json.loads(R24_RESULT.read_text(encoding='utf-8'))
    with np.load(r22.SCORES, allow_pickle=False) as bank:
        result = evaluate(data, bank, saved22, saved24, hashes, check)
    check()
    result['runtime_seconds'] = time.monotonic() - start
    result['peak_rss_bytes'] = max(peak[0], process.memory_info().rss)
    result['rss_sampling'] = 'process RSS sampled every 0.1 seconds, plus initial and final samples'
    r24.write_result(OUT, result)
    finished.set()
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
