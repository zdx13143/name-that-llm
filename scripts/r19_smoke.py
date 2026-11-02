"""Tiny public-data and synthetic-text pipeline smoke. No real CV or score."""
import json
from pathlib import Path
import threading
import time

import pandas as pd
import psutil
from r19_overlap_audit import FILES, digest, similarities, THRESHOLD, read_pool
from r19_svc_component import fit_predict


def main():
    start = time.perf_counter()
    for name in ('baseline', 'open', 'extra'):
        if digest(FILES[name][0]) != FILES[name][1]:
            raise ValueError('hash mismatch: ' + name)
    sources = {}
    for name in ('baseline', 'open', 'extra'):
        df, removed = read_pool(name)
        df = df.groupby('model', sort=True, group_keys=False).head(10 if name == 'extra' else 50)
        sources[name] = list(df[['text', 'model']].itertuples(index=False, name=None))
    labels = sorted({model for _, model in sources['baseline']})
    if len(labels) != 18:
        raise ValueError('expected 18 classes')
    official = [(f'The synthetic ordinary sample {i} for class {j}.', label)
                for j, label in enumerate(labels) for i in range(2)]
    validation = ['A fabricated validation sentence for the pipeline.']
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    durations = {}
    try:
        for name in ('baseline', 'open'):
            t = time.perf_counter()
            pool = sources[name]
            keep_pool = similarities(validation, [r[0] for r in pool]) < THRESHOLD
            keep_extra = similarities(validation, [r[0] for r in sources['extra']]) < THRESHOLD
            fit_predict(official, [r for r, keep in zip(sources['extra'], keep_extra) if keep],
                        [r for r, keep in zip(pool, keep_pool) if keep], validation, labels)
            durations[name] = time.perf_counter() - t
    finally:
        stop.set()
        monitor_thread.join()
    result = {'kind': 'tiny_smoke_no_score', 'pool_rows_each': 900, 'extra_rows': 180,
              'synthetic_training_rows': len(official), 'synthetic_validation_rows': len(validation),
              'seconds_by_condition': durations, 'elapsed_seconds_including_load': time.perf_counter()-start,
              'sampled_peak_rss_bytes': peak[0]}
    target = Path('outputs/r19/smoke_v3.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
