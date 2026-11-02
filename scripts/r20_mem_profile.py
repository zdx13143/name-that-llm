"""Bounded fold-0 union fit only. No predictions, scores, or CV claim."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from r19_cpu_cv import validate_data, load_source, filter_source
from r19_svc_component import R16_LABELS, build_fit_frame, shape, function
from r20_cpu_cv import union_pool, EXPECTED_UNION, kill_tree

OUT = Path(__file__).resolve().parent.parent / 'outputs' / 'r20'
CAP = 13 * 1024**3
WALL = 30 * 60


def fit_only(stage):
    stage('validate')
    data = validate_data()
    folds, truth = np.asarray(data['fold_id']), np.asarray(data['truth'])
    texts, labels = data['texts'], data['labels']
    if set(labels) != set(R16_LABELS):
        raise ValueError('label mismatch')
    valid = [texts[i] for i in np.flatnonzero(folds == 0)]
    official = [(texts[i], labels[truth[i]]) for i in np.flatnonzero(folds != 0)]
    stage('load_sources')
    baseline, _ = load_source('baseline')
    opened, _ = load_source('open')
    extra, _ = load_source('extra')
    stage('union')
    union, origins = union_pool(opened, baseline)
    if len(union) != EXPECTED_UNION or set(label for _, label in union) != set(R16_LABELS):
        raise ValueError('union identity mismatch')
    del baseline, opened, origins
    gc.collect()
    stage('filter_union')
    pool, _ = filter_source(union, valid)
    del union
    gc.collect()
    stage('filter_extra')
    filtered_extra, _ = filter_source(extra, valid)
    del extra
    stage('build_frame')
    frame = build_fit_frame(official, filtered_extra, pool, valid, R16_LABELS)
    del official, filtered_extra, pool, data, texts
    gc.collect()
    if set(frame.model) != set(R16_LABELS):
        raise ValueError('training class mismatch')
    x = frame.text.astype(str).tolist()
    y = frame.model.map({label: i for i, label in enumerate(R16_LABELS)}).to_numpy(dtype=int)
    weights = frame['_srcw'].to_numpy(dtype=np.float32)
    del frame
    train_matrices, test_matrices = [], []
    for view, lo, hi, maximum, weight in [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]:
        stage('view_' + view)
        prep = shape if view == 'shape' else function if view == 'function' else str
        analyzer = 'char' if view == 'shape' else 'word' if view == 'function' else view
        options = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=maximum, dtype=np.float32)
        if analyzer == 'word':
            options['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
        vec = TfidfVectorizer(**options)
        train_matrices.append(vec.fit_transform([prep(t) for t in x]) * weight)
        test_matrices.append(vec.transform([prep(t) for t in valid]) * weight)
    stage('stack_train')
    train = hstack(train_matrices, format='csr')
    del train_matrices
    gc.collect()
    stage('svc_fit')
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
    svc.fit(train, y, sample_weight=weights)
    stage('fit_complete_no_prediction')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stages_file = OUT / 'mem_profile_stages.jsonl'
    if '--child' in sys.argv:
        with stages_file.open('x', encoding='utf-8', buffering=1) as f:
            def stage(name):
                f.write(json.dumps({'stage': name, 'monotonic': time.monotonic()}) + '\n')
            fit_only(stage)
        return
    result_path = OUT / 'mem_profile.json'
    if result_path.exists() or stages_file.exists():
        raise FileExistsError('profile output already exists')
    start = time.monotonic()
    supervisor = psutil.Process()
    proc = subprocess.Popen([sys.executable, __file__, '--child'])
    child = psutil.Process(proc.pid)
    peaks, peak, current, reason = {}, 0, 'starting', None
    while proc.poll() is None:
        try:
            if stages_file.exists():
                lines = stages_file.read_text(encoding='utf-8').splitlines()
                if lines:
                    current = json.loads(lines[-1])['stage']
            tree = [child] + child.children(recursive=True)
            rss = supervisor.memory_info().rss + sum(p.memory_info().rss for p in tree if p.is_running())
            peak = max(peak, rss)
            peaks[current] = max(peaks.get(current, 0), rss)
            if rss >= CAP:
                reason = 'rss_cap'
            elif time.monotonic() - start >= WALL:
                reason = 'wall_cap'
            if reason:
                kill_tree(child)
                break
        except psutil.NoSuchProcess:
            pass
        time.sleep(.2)
    code = proc.wait()
    if stages_file.exists():
        lines = stages_file.read_text(encoding='utf-8').splitlines()
        if lines:
            current = json.loads(lines[-1])['stage']
    status = reason or ('complete' if code == 0 and current == 'fit_complete_no_prediction' else 'failed')
    result = {'status': status, 'last_stage': current, 'stage_peak_bytes': peaks, 'peak_bytes': peak,
              'seconds': time.monotonic() - start, 'cap_bytes': CAP, 'wall_cap_seconds': WALL,
              'returncode': code, 'predictions_or_metrics_computed': False}
    with result_path.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
