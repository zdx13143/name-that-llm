"""One-shot R19 CPU CV. Do not run during preparation. See R19_PLAN_20261102.md."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
import pyarrow
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r19_overlap_audit import DEV, FILES, THRESHOLD, digest, similarities, read_pool
from r19_svc_component import fit_predict, R16_LABELS

OUT = Path('outputs/r19/cv')
LIMIT_SECONDS = 4 * 60 * 60


def validate_data():
    import hashlib
    actual_versions = (sys.version.split()[0], np.__version__, scipy.__version__, sklearn.__version__, pd.__version__, pyarrow.__version__)
    if actual_versions != ('3.11.6', '2.4.3', '1.17.1', '1.9.0', '3.0.5', '25.0.1'):
        raise ValueError('runtime package versions differ from preregistration')
    for name, (path, expected) in FILES.items():
        if digest(path) != expected:
            raise ValueError('incorrect input hash: ' + name)
    data = json.loads(DEV.read_text(encoding='utf-8'))
    folds = np.asarray(data['fold_id'])
    groups = np.asarray(data['groups'])
    truth = np.asarray(data['truth'])
    canonical = json.dumps(data['fold_id'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    if hashlib.sha256(canonical).hexdigest() != '96040086312dca7ce1d3d14bb3ee59640870ef35edba42c746c7f84ae7b827da':
        raise ValueError('fold identity mismatch')
    if len(folds) != 2159 or len(set(groups)) != 120 or len(data['labels']) != 18:
        raise ValueError('development shape mismatch')
    if set(folds) != set(range(5)) or set(truth) != set(range(18)):
        raise ValueError('fold or class mismatch')
    for fold in range(5):
        valid = folds == fold
        if len(set(groups[valid])) != 24 or set(groups[valid]) & set(groups[~valid]):
            raise ValueError('group leakage or fold mismatch')
        if set(truth[valid]) != set(range(18)) or set(truth[~valid]) != set(range(18)):
            raise ValueError('missing class')
    return data


def load_source(name):
    df, removed = read_pool(name)
    df['text'] = df['text'].fillna('').astype(str)
    return list(df.itertuples(index=False, name=None)), removed


def filter_source(rows, validation):
    scores = similarities(validation, [t for t, _ in rows])
    return [row for row, score in zip(rows, scores) if score < THRESHOLD], int((scores >= THRESHOLD).sum())


def metrics(truth, baseline, candidate, folds, groups):
    labels = np.arange(18)
    ba0 = balanced_accuracy_score(truth, baseline)
    ba1 = balanced_accuracy_score(truth, candidate)
    rec0 = recall_score(truth, baseline, labels=labels, average=None, zero_division=0)
    rec1 = recall_score(truth, candidate, labels=labels, average=None, zero_division=0)
    deltas = [balanced_accuracy_score(truth[folds == f], candidate[folds == f]) -
              balanced_accuracy_score(truth[folds == f], baseline[folds == f]) for f in range(5)]
    group_values = np.unique(groups)
    group_totals = np.zeros((len(group_values), 18), dtype=np.int64)
    group_delta = np.zeros_like(group_totals)
    for n, group in enumerate(group_values):
        mask = groups == group
        group_totals[n] = np.bincount(truth[mask], minlength=18)
        group_delta[n] = (np.bincount(truth[mask & (candidate == truth)], minlength=18) -
                          np.bincount(truth[mask & (baseline == truth)], minlength=18))
    rng = np.random.default_rng(19)
    draws = rng.integers(0, len(group_values), (10000, len(group_values)))
    multiplicities = np.zeros((10000, len(group_values)), dtype=np.int32)
    np.add.at(multiplicities, (np.arange(10000)[:, None], draws), 1)
    totals = multiplicities @ group_totals
    if (totals == 0).any():
        raise ValueError('bootstrap sample missing a class')
    boot = ((multiplicities @ group_delta) / totals).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {'baseline_ba': ba0, 'candidate_ba': ba1, 'delta_ba': ba1-ba0,
            'baseline_recall': rec0.tolist(), 'candidate_recall': rec1.tolist(),
            'fold_deltas': deltas, 'bootstrap_95': [float(lo), float(hi)],
            'adopt': bool(ba1-ba0 >= .01 and lo > 0 and sum(d > 0 for d in deltas) >= 4 and np.min(rec1-rec0) >= -.05)}


def worker():
    data = validate_data()
    folds, groups, truth = (np.asarray(data[k]) for k in ('fold_id', 'groups', 'truth'))
    texts, dev_labels = data['texts'], data['labels']
    labels = R16_LABELS
    if set(labels) != set(dev_labels):
        raise ValueError('R16 and development label sets differ')
    to_development = np.array([dev_labels.index(label) for label in labels], dtype=int)
    loaded = {name: load_source(name) for name in ('baseline', 'open', 'extra')}
    sources = {name: value[0] for name, value in loaded.items()}
    pred = {name: np.full(len(truth), -1, dtype=int) for name in ('baseline', 'open')}
    exclusion = []
    for f in range(5):
        valid_idx = np.flatnonzero(folds == f)
        train_idx = np.flatnonzero(folds != f)
        valid_text = [texts[i] for i in valid_idx]
        official = [(texts[i], dev_labels[truth[i]]) for i in train_idx]
        filtered = {}
        counts = {}
        for name in ('baseline', 'open', 'extra'):
            filtered[name], counts[name] = filter_source(sources[name], valid_text)
        exclusion.append(counts)
        for name in ('baseline', 'open'):
            pred[name][valid_idx] = to_development[fit_predict(official, filtered['extra'], filtered[name], valid_text, labels)]
    if (pred['baseline'] < 0).any() or (pred['open'] < 0).any():
        raise ValueError('incomplete OOF')
    result = metrics(truth, pred['baseline'], pred['open'], folds, groups)
    result.update({'labels': dev_labels, 'exclusions': exclusion, 'open_flag_removed': loaded['open'][1], 'seed': 19, 'bootstrap_samples': 10000})
    with (OUT / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)


def supervise():
    # Fail cheap checks before consuming the immutable claim.
    start = time.monotonic()
    supervisor = psutil.Process()
    parent_cpu_start = sum(supervisor.cpu_times()[:2])
    validate_data()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'claim.json').open('x', encoding='utf-8') as f:
        json.dump({'created_unix': time.time(), 'pid': os.getpid(), 'limit_seconds': LIMIT_SECONDS}, f)
    peak = 0
    cpu = 0.0
    status = 'failed'
    reason = None
    with (OUT / 'worker.log').open('x', encoding='utf-8') as log:
        env = os.environ.copy()
        env['R19_SUPERVISOR_PID'] = str(os.getpid())
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'], stdout=log, stderr=subprocess.STDOUT, env=env)
        process = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                tree = [process] + process.children(recursive=True)
                peak = max(peak, supervisor.memory_info().rss + sum(p.memory_info().rss for p in tree if p.is_running()))
                cpu = sum(supervisor.cpu_times()[:2]) - parent_cpu_start + sum(sum(p.cpu_times()[:2]) for p in tree if p.is_running())
                if time.monotonic() - start >= LIMIT_SECONDS or cpu >= LIMIT_SECONDS:
                    reason = 'four-hour budget exceeded'
                    proc.terminate()
                    proc.wait()
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(1)
        status = 'complete' if proc.returncode == 0 and (OUT / 'result.json').exists() else 'failed'
    with (OUT / 'receipt.json').open('x', encoding='utf-8') as f:
        json.dump({'status': status, 'reason': reason, 'returncode': proc.returncode,
                   'wall_seconds': time.monotonic()-start, 'sampled_cpu_seconds': cpu,
                   'peak_rss_bytes': peak}, f, indent=2)
    if status != 'complete':
        raise RuntimeError('CV failed or stopped; claim is retained')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if os.environ.get('R19_SUPERVISOR_PID') != str(os.getppid()) or not (OUT / 'claim.json').is_file():
            parser.error('worker must be launched by the claiming supervisor')
        worker()
    elif args.run:
        supervise()
    else:
        parser.error('use --run only after preparation and review')


if __name__ == '__main__':
    main()
