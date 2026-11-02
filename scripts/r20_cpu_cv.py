"""R20 one-shot CPU CV; --smoke uses synthetic data only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r19_cpu_cv import validate_data, load_source, filter_source, metrics, runtime_versions
from r19_overlap_audit import FILES
from r19_svc_component import R16_LABELS, norm, build_fit_frame, fit_predict

OUT = Path(os.environ.get('NTL_R20_OUT', str(Path(__file__).resolve().parent.parent / 'outputs' / 'r20' / 'cv')))
LIMIT_SECONDS = float(os.environ.get('NTL_R20_LIMIT_SECONDS', 4 * 60 * 60))
MIN_AVAILABLE = int(float(os.environ.get('NTL_R20_MIN_AVAILABLE_GIB', 24)) * 1024**3)
MAX_RSS = int(float(os.environ.get('NTL_R20_MAX_RSS_GIB', 26)) * 1024**3)
if not (0 < LIMIT_SECONDS <= 32400 and 22 * 1024**3 <= MIN_AVAILABLE and 0 < MAX_RSS <= 28 * 1024**3):
    raise ValueError('invalid R20 resource limits')
EXPECTED_UNION = 212889


def union_pool(open_rows, baseline_rows):
    seen, result, origins = set(), [], []
    for source, rows in (('open', open_rows), ('baseline', baseline_rows)):
        for row in rows:
            key = norm(row[0])
            if key not in seen:
                seen.add(key)
                result.append(row)
                origins.append(source)
    return result, origins


def final_counts(official, extra, pool, validation, labels, origins):
    frame = build_fit_frame(official, extra, pool, validation, labels)
    counts = {'official': int((frame._srcw == 1.0).sum()),
              'extra': int((frame._srcw == 0.5).sum()),
              'open': 0, 'baseline': 0}
    by_key = {norm(row[0]): source for row, source in zip(pool, origins)}
    for text in frame.loc[frame._srcw == 0.35, 'text']:
        counts[by_key[norm(text)]] += 1
    return counts


def worker():
    data = validate_data()
    folds, groups, truth = (np.asarray(data[k]) for k in ('fold_id', 'groups', 'truth'))
    texts, dev_labels = data['texts'], data['labels']
    if set(R16_LABELS) != set(dev_labels):
        raise ValueError('R16 and development label sets differ')
    mapping = np.array([dev_labels.index(label) for label in R16_LABELS], dtype=int)
    loaded = {name: load_source(name) for name in ('baseline', 'open', 'extra')}
    sources = {name: value[0] for name, value in loaded.items()}
    union, origins = union_pool(sources['open'], sources['baseline'])
    if len(union) != EXPECTED_UNION:
        raise ValueError('union input count mismatch')
    if set(label for _, label in union) != set(R16_LABELS):
        raise ValueError('union label set mismatch')
    predictions = {name: np.full(len(truth), -1, dtype=int) for name in ('baseline', 'union')}
    exclusions, training_counts = [], []
    for fold in range(5):
        valid_idx, train_idx = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        validation = [texts[i] for i in valid_idx]
        official = [(texts[i], dev_labels[truth[i]]) for i in train_idx]
        filtered, counts = {}, {}
        for name, rows in (('baseline', sources['baseline']), ('union', union), ('extra', sources['extra'])):
            filtered[name], counts[name] = filter_source(rows, validation)
        kept_keys = {norm(row[0]) for row in filtered['union']}
        counts['union_by_source'] = {source: sum(origin == source and norm(row[0]) not in kept_keys
                                                 for row, origin in zip(union, origins))
                                     for source in ('open', 'baseline')}
        exclusions.append(counts)
        union_origins = {norm(row[0]): origin for row, origin in zip(union, origins)}
        fold_counts = {}
        for name in ('baseline', 'union'):
            pool = filtered[name]
            pool_origins = ([union_origins[norm(row[0])] for row in pool] if name == 'union'
                            else ['baseline'] * len(pool))
            fold_counts[name] = final_counts(official, filtered['extra'], pool, validation, R16_LABELS, pool_origins)
            predictions[name][valid_idx] = mapping[fit_predict(official, filtered['extra'], pool, validation, R16_LABELS)]
        training_counts.append(fold_counts)
    if any(((p < 0) | (p >= len(dev_labels))).any() for p in predictions.values()):
        raise ValueError('incomplete or invalid OOF')
    result = metrics(truth, predictions['baseline'], predictions['union'], folds, groups)
    result.update({'labels': dev_labels, 'oof_predictions': {k: v.tolist() for k, v in predictions.items()},
                   'exclusions': exclusions, 'final_training_rows': training_counts,
                   'union_input_count': len(union), 'open_flag_removed': loaded['open'][1],
                   'input_sha256': {name: expected for name, (_, expected) in FILES.items()},
                   'fold_id_sha256': '96040086312dca7ce1d3d14bb3ee59640870ef35edba42c746c7f84ae7b827da',
                   'oof_row_order': 'development.json original row order; labels index development labels',
                   'seed': 19, 'bootstrap_samples': 10000, 'runtime_versions': runtime_versions()})
    with (OUT / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)


def kill_tree(process):
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for p in list(reversed(children)) + [process]:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=5)


def supervise():
    start = time.monotonic()
    if psutil.virtual_memory().available < MIN_AVAILABLE:
        raise RuntimeError(f'available physical memory below {MIN_AVAILABLE / 1024**3:g} GiB; no claim created')
    supervisor = psutil.Process()
    cpu_start = sum(supervisor.cpu_times()[:2])
    validate_data()
    if psutil.virtual_memory().available < MIN_AVAILABLE:
        raise RuntimeError(f'available physical memory below {MIN_AVAILABLE / 1024**3:g} GiB; no claim created')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'claim.json').open('x', encoding='utf-8') as f:
        json.dump({'created_unix': time.time(), 'pid': os.getpid(), 'limit_seconds': LIMIT_SECONDS,
                   'min_available_bytes': MIN_AVAILABLE, 'max_tree_rss_bytes': MAX_RSS}, f)
    peak, cpu, reason, returncode = 0, 0.0, None, None
    status = 'failed'
    try:
        with (OUT / 'worker.log').open('x', encoding='utf-8') as log:
            env = os.environ.copy()
            env['R20_SUPERVISOR_PID'] = str(os.getpid())
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'], stdout=log, stderr=subprocess.STDOUT, env=env)
            process = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    tree = [process] + process.children(recursive=True)
                    rss = supervisor.memory_info().rss + sum(p.memory_info().rss for p in tree if p.is_running())
                    peak = max(peak, rss)
                    cpu = sum(supervisor.cpu_times()[:2]) - cpu_start + sum(sum(p.cpu_times()[:2]) for p in tree if p.is_running())
                    if rss > MAX_RSS:
                        reason = f'sampled process tree RSS exceeded {MAX_RSS / 1024**3:g} GiB'
                    elif time.monotonic() - start >= LIMIT_SECONDS or cpu >= LIMIT_SECONDS:
                        reason = 'wall or CPU budget exceeded'
                    if reason:
                        kill_tree(process)
                        proc.wait()
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(1)
            returncode = proc.wait()
            if reason is None and returncode == 0 and (OUT / 'result.json').is_file():
                status = 'complete'
            elif reason is None:
                reason = 'worker failed or result missing'
    except Exception as exc:
        reason = f'supervisor error: {type(exc).__name__}: {exc}'
        if 'proc' in locals() and proc.poll() is None:
            kill_tree(psutil.Process(proc.pid))
            proc.wait()
        raise
    finally:
        with (OUT / 'receipt.json').open('x', encoding='utf-8') as f:
            json.dump({'status': status, 'reason': reason, 'returncode': returncode,
                       'wall_seconds': time.monotonic() - start, 'sampled_cpu_seconds': cpu,
                       'peak_rss_bytes': peak, 'limit_seconds': LIMIT_SECONDS,
                       'min_available_bytes': MIN_AVAILABLE, 'max_tree_rss_bytes': MAX_RSS}, f, indent=2)
    if status != 'complete':
        raise RuntimeError('CV failed or stopped; claim is retained')


def smoke():
    start = time.perf_counter()
    peak = [0]
    stop = threading.Event()
    process = psutil.Process()
    def sample():
        while not stop.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        labels = ['a', 'b']
        official = [(f'synthetic common text category {c} example {i}', c) for c in labels for i in range(10)]
        baseline = [(f'baseline synthetic words category {c} item {i}', c) for c in labels for i in range(80)]
        opened = [(f'open synthetic words category {c} item {i}', c) for c in labels for i in range(80)]
        union, origins = union_pool(opened, baseline)
        validation = ['synthetic held out passage unseen']
        for pool, source in ((baseline, ['baseline'] * len(baseline)), (union, origins)):
            kept, _ = filter_source(pool, validation)
            lookup = {norm(row[0]): origin for row, origin in zip(pool, source)}
            counts = final_counts(official, [], kept, validation, labels, [lookup[norm(row[0])] for row in kept])
            assert sum(counts.values()) == len(build_fit_frame(official, [], kept, validation, labels))
            assert fit_predict(official, [], kept, validation, labels).shape == (1,)
    finally:
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        thread.join()
    print(json.dumps({'kind': 'synthetic_smoke_no_score', 'union_rows': len(union),
                      'wall_seconds': time.perf_counter() - start, 'peak_rss_bytes': peak[0]}))


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run', action='store_true')
    choice.add_argument('--smoke', action='store_true')
    choice.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if os.environ.get('R20_SUPERVISOR_PID') != str(os.getppid()) or not (OUT / 'claim.json').is_file():
            parser.error('worker must be launched by the claiming supervisor')
        worker()
    elif args.smoke:
        smoke()
    else:
        supervise()


if __name__ == '__main__':
    main()
