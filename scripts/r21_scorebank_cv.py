"""One-shot R21 score bank; --smoke operates exclusively on synthetic data."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r19_cpu_cv import validate_data, filter_source, metrics, runtime_versions
from r19_overlap_audit import FILES, read_pool
from r19_svc_component import R16_LABELS, norm
from r21_members import run_member
from r21_sinkhorn import balance

OUT = Path(os.environ.get('NTL_R21_OUT', str(Path(__file__).resolve().parent.parent / 'outputs/r21/cv')))
LIMIT = float(os.environ.get('NTL_R21_LIMIT_SECONDS', '32100'))
MIN_AVAIL = float(os.environ.get('NTL_R21_MIN_AVAILABLE_GIB', '24')) * 1024**3
MAX_RSS = float(os.environ.get('NTL_R21_MAX_RSS_GIB', '28')) * 1024**3
if not (0 < LIMIT <= 32400 and MIN_AVAIL >= 24 * 1024**3 and 0 < MAX_RSS <= 28 * 1024**3):
    raise ValueError('invalid resource limits')
WEIGHTS = {'r16': {'svc4': 1.5, 'tfidf': .25, 'distort': .1},
           'areeb': {'svc4': 12., 'tfidf': .25, 'distort': .1}}
MEMBERS = ('tfidf', 'distort', 'svc4')
CONDITIONS = ('r16', 'areeb')
PART_NAMES = {'r16': 'baseline', 'areeb': 'candidate'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def checkpoint_identity(data, sources, synthetic):
    root = Path(__file__).resolve().parent
    scripts = ('r21_scorebank_cv.py', 'r21_members.py', 'r21_sinkhorn.py',
               'r19_cpu_cv.py', 'r19_overlap_audit.py', 'r19_svc_component.py')
    versions = runtime_versions() | {'psutil': psutil.__version__}
    config = {'schema': 1, 'weights': WEIGHTS, 'members': MEMBERS, 'conditions': CONDITIONS,
              'labels': R16_LABELS, 'folds': list(range(5)), 'versions': versions,
              'scripts': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in scripts},
              'notebook_blocks': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root / 'r21_source').glob('*.txt'))}}
    inputs = ({'synthetic_data': canonical_hash(data), 'synthetic_sources': canonical_hash(sources)}
              if synthetic else {name: sha for name, (_, sha) in FILES.items()})
    return {'config_sha256': canonical_hash(config), 'input_sha256': inputs}


def unit_name(condition, fold, member):
    return f'{PART_NAMES[condition]}__f{fold}__{member}'


def atomic_json(path, value):
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.r21-', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def save_part(path, score, rows, identity, unit, stage):
    score = np.asarray(score, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    info = {'unit': unit, **identity, 'stage': stage,
            'scores_sha256': hashlib.sha256(score.tobytes(order='C')).hexdigest(),
            'rows_sha256': hashlib.sha256(rows.tobytes(order='C')).hexdigest()}
    if path.exists():
        raise FileExistsError('part already exists: ' + path.name)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.r21-', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            np.savez_compressed(f, scores=score, row_indices=rows, metadata=np.array(json.dumps(info, sort_keys=True)))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def read_part(path, identity, unit, rows):
    try:
        with np.load(path, allow_pickle=False) as part:
            if set(part.files) != {'scores', 'row_indices', 'metadata'}:
                raise ValueError('part fields mismatch')
            score = part['scores']
            indices = part['row_indices']
            info = json.loads(str(part['metadata'].item()))
        if (info['unit'] != unit or info['config_sha256'] != identity['config_sha256'] or
                info['input_sha256'] != identity['input_sha256']):
            raise ValueError('part identity mismatch')
        if score.dtype != np.dtype('float64') or indices.dtype != np.dtype('int64'):
            raise ValueError('part dtype mismatch')
        if score.shape != (len(rows), 18) or not np.isfinite(score).all() or not np.array_equal(indices, rows):
            raise ValueError('part scores or alignment mismatch')
        if (hashlib.sha256(score.tobytes(order='C')).hexdigest() != info['scores_sha256'] or
                hashlib.sha256(indices.tobytes(order='C')).hexdigest() != info['rows_sha256']):
            raise ValueError('part content hash mismatch')
        if info['stage']['status'] != 'ok':
            raise ValueError('part stage failed')
        return score, info['stage']
    except Exception as exc:
        raise ValueError(f'invalid checkpoint {path.name}: {exc}') from exc


def resume_parts(directory, identity, folds):
    if directory is None:
        return {}
    directory = Path(directory)
    directory = directory / 'parts' if (directory / 'parts').is_dir() else directory
    if not directory.is_dir():
        raise ValueError('resume directory missing')
    expected = {unit_name(condition, fold, member): np.flatnonzero(folds == fold)
                for fold in range(5) for condition in CONDITIONS for member in MEMBERS}
    result = {}
    for path in sorted(directory.glob('*.npz')):
        if path.stem not in expected:
            raise ValueError('unknown checkpoint: ' + path.name)
        result[path.stem] = read_part(path, identity, path.stem, expected[path.stem])
    if not result:
        raise ValueError('resume directory has no checkpoints')
    return result


def tree_rss(process):
    try:
        return sum(p.memory_info().rss for p in [process] + process.children(recursive=True) if p.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


class Stage:
    def __init__(self, name, records, log):
        self.name, self.records, self.log = name, records, log

    def __enter__(self):
        self.start = time.monotonic()
        self.cpu = time.process_time()
        self.peak = 0
        self.stop = threading.Event()
        with self.log.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'stage': self.name, 'event': 'start', 'unix': time.time()}) + '\n')
        proc = psutil.Process()
        def sample():
            while not self.stop.is_set():
                self.peak = max(self.peak, tree_rss(proc))
                self.stop.wait(.25)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, typ, value, tb):
        self.stop.set()
        self.thread.join()
        row = {'stage': self.name, 'event': 'finish', 'wall_seconds': time.monotonic() - self.start,
               'cpu_seconds': time.process_time() - self.cpu, 'sampled_peak_rss_bytes': self.peak,
               'status': 'ok' if typ is None else 'failed'}
        self.records.append(row)
        with self.log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')


def sample_v15(open_df, matched_df):
    """Exact cell 16 rule: global sampling, before fold-specific filtering."""
    counts = matched_df.model.value_counts()
    return pd.concat([g.sample(min(len(g), int(2 * counts.get(m, 0))),
                               weights='match_weight', random_state=0)
                      for m, g in open_df.groupby('model')])[['text', 'model']]


def load_sources():
    baseline, _ = read_pool('baseline')
    extra, _ = read_pool('extra')
    open_filtered, removed = read_pool('open')
    if removed != 0:
        raise ValueError('Open Pool flag removed nonzero rows')
    op = pd.read_parquet(FILES['open'][0], columns=['text', 'model', 'match_weight', 'flag_train_prompt_regen'])
    if op.flag_train_prompt_regen.isna().any() or op.flag_train_prompt_regen.astype(bool).any():
        raise ValueError('Open Pool flag invalid')
    if len(op) != len(open_filtered) or len(op) != 190588:
        raise ValueError('Open Pool row count mismatch')
    if op.match_weight.isna().any() or (op.match_weight < 0).any() or not np.isfinite(op.match_weight).all():
        raise ValueError('invalid Open Pool weights')
    sampled = sample_v15(op, extra)
    def rows(df):
        df = df[['text', 'model']].copy()
        df['text'] = df.text.fillna('').astype(str)
        return list(df.itertuples(index=False, name=None))
    return {'baseline': rows(baseline), 'extra': rows(extra), 'open': rows(open_filtered),
            'v15_extra': rows(sampled)}, {'open_flag_removed': removed, 'v15_extra_rows': len(sampled)}


def filter_fold(rows, validation):
    kept, cosine_removed = filter_source(rows, validation)
    held = {norm(t) for t in validation}
    filtered = [row for row in kept if norm(row[0]) not in held]
    return filtered, {'cosine': cosine_removed, 'exact_after_cosine': len(kept) - len(filtered)}


def calculate(data, sources, out, metadata, resume_from=None):
    labels = data['labels']
    if set(labels) != set(R16_LABELS):
        raise ValueError('label identity mismatch')
    folds = np.asarray(data['fold_id'], dtype=int)
    groups = np.asarray(data['groups'])
    truth = np.asarray(data['truth'], dtype=int)
    texts = data['texts']
    mapping = np.array([labels.index(label) for label in R16_LABELS])
    count = len(truth)
    arrays = {'row_index': np.arange(count), 'fold': folds, 'truth': truth,
              'group': groups.astype(str)}
    for condition in CONDITIONS:
        for name in (*MEMBERS, 'blend', 'balanced'):
            arrays[f'{condition}_{name}'] = np.full((count, 18), np.nan, dtype=np.float64)
    stage_records, exclusions = [], []
    stage_log = out / 'stages.jsonl'
    identity = checkpoint_identity(data, sources, bool(metadata.get('synthetic')))
    parts_dir = out / 'parts'
    if resume_from is not None and Path(resume_from).resolve() in (out.resolve(), parts_dir.resolve()):
        raise ValueError('resume source must differ from this run output')
    resumed = resume_parts(resume_from, identity, folds)
    parts_dir.mkdir(exist_ok=True)
    progress = {'schema': 1, **identity, 'total_units': 30, 'completed_units': [], 'resumed_units': []}
    atomic_json(out / 'progress.json', progress)
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        validation = [texts[i] for i in valid]
        official = [(texts[i], labels[truth[i]]) for i in train]
        y = np.array([R16_LABELS.index(label) for _, label in official], dtype=int)
        with Stage(f'fold{fold}/filter', stage_records, stage_log):
            filtered, counts = {}, {}
            for key in ('baseline', 'extra', 'open', 'v15_extra'):
                filtered[key], counts[key] = filter_fold(sources[key], validation)
            exclusions.append(counts)
        for condition in CONDITIONS:
            extra = filtered['extra' if condition == 'r16' else 'v15_extra']
            pool = filtered['baseline' if condition == 'r16' else 'open']
            for member in MEMBERS:
                unit = unit_name(condition, fold, member)
                if unit in resumed:
                    score, unit_stage = resumed[unit]
                    progress['resumed_units'].append(unit)
                else:
                    with Stage(f'fold{fold}/{condition}/{member}', stage_records, stage_log):
                        score = run_member(condition, member, official, extra, pool, validation, R16_LABELS)
                    unit_stage = stage_records[-1]
                save_part(parts_dir / (unit + '.npz'), score, valid, identity, unit, unit_stage)
                arrays[f'{condition}_{member}'][valid] = score
                progress['completed_units'].append({'unit': unit, 'resumed': unit in resumed,
                                                    'stage': unit_stage})
                atomic_json(out / 'progress.json', progress)
                del score
                gc.collect()
            weighted = {name: arrays[f'{condition}_{name}'][valid] * WEIGHTS[condition][name] for name in MEMBERS}
            with Stage(f'fold{fold}/{condition}/balance', stage_records, stage_log):
                raw, balanced = balance(weighted, y, condition)
                arrays[f'{condition}_blend'][valid] = raw
                arrays[f'{condition}_balanced'][valid] = balanced
            del weighted, raw, balanced
        del filtered
        gc.collect()
    if any(not np.isfinite(a).all() for a in arrays.values() if a.dtype.kind == 'f'):
        raise ValueError('incomplete or invalid score bank')
    def prediction(cond, key):
        return mapping[arrays[f'{cond}_{key}'].argmax(axis=1)]
    assessments = {key: metrics(truth, prediction('r16', key), prediction('areeb', key), folds, groups)
                   for key in ('balanced', 'blend', 'svc4')}
    # Only the balanced comparison has adoption authority.
    for key in ('blend', 'svc4'):
        assessments[key].pop('adopt', None)
    with (out / 'scores.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    score_hash = hashlib.sha256((out / 'scores.npz').read_bytes()).hexdigest()
    result = {'gate': assessments['balanced'], 'diagnostics': {'unbalanced_blend': assessments['blend'],
              'svc_alone': assessments['svc4']}, 'labels_development': labels, 'labels_scores': R16_LABELS,
              'weights': WEIGHTS, 'exclusions': exclusions, 'stages': stage_records,
              'input_sha256': {name: expected for name, (_, expected) in FILES.items()},
              'fold_id_sha256': '96040086312dca7ce1d3d14bb3ee59640870ef35edba42c746c7f84ae7b827da',
              'scores_sha256': score_hash, 'runtime_versions': runtime_versions(),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((Path(__file__).resolve().parent / 'r21_source').glob('*.txt'))},
              'notebook_source_sha256': {'r16': '38204e9f00cb7da783932a917929d5d7af8c146f02f5e487bb033d5fb8b48f37',
                                         'v15': 'adbde4ef757f3ae11d5bf9a748a10ce48d4e5d00f792ed22277f3b94449077a5'},
              'dataset_versions': {'open': 'crtproduction/name-that-llm-open-pool/11',
                                   'matched': 'dhruvtibarewal/nlm-matched-clean/3',
                                   'package': 'zephyrpong/ntl-r21-scorebank-package'},
              'score_format': 'unweighted member log probabilities in labels_scores order; blends weighted',
              'bootstrap_seed': 19, 'bootstrap_samples': 10000,
              'checkpoint_config_sha256': identity['config_sha256'],
              'resumed_units': progress['resumed_units'], **metadata}
    result['runtime_versions']['psutil'] = psutil.__version__
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)


def worker():
    data = validate_data()
    records = []
    with Stage('load_and_global_sample', records, OUT / 'stages.jsonl'):
        sources, metadata = load_sources()
    metadata['load_stage'] = records[0]
    calculate(data, sources, OUT, metadata, os.environ.get('R21_RESUME_FROM') or None)


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


def supervise(synthetic=False, resume_from=None):
    start = time.monotonic()
    if not synthetic and psutil.virtual_memory().available < MIN_AVAIL:
        raise RuntimeError('insufficient available memory; no claim')
    if not synthetic:
        validate_data()
    if not synthetic and psutil.virtual_memory().available < MIN_AVAIL:
        raise RuntimeError('insufficient available memory; no claim')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'claim.json').open('x', encoding='utf-8') as f:
        json.dump({'created_unix': time.time(), 'pid': os.getpid(), 'limit_seconds': LIMIT, 'synthetic': synthetic,
                   'min_available_bytes': MIN_AVAIL, 'max_rss_bytes': MAX_RSS}, f)
    supervisor = psutil.Process()
    parent_cpu = sum(supervisor.cpu_times()[:2])
    peak, cpu, reason, code, status = 0, 0., None, None, 'failed'
    proc = None
    try:
        with (OUT / 'worker.log').open('x', encoding='utf-8') as log:
            env = os.environ.copy()
            env['R21_SUPERVISOR_PID'] = str(os.getpid())
            env['NTL_R21_OUT'] = str(OUT)
            if resume_from is not None:
                env['R21_RESUME_FROM'] = str(Path(resume_from).resolve())
            else:
                env.pop('R21_RESUME_FROM', None)
            flag = '--synthetic-worker' if synthetic else '--worker'
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), flag], stdout=log, stderr=subprocess.STDOUT, env=env)
            process = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    tree = [process] + process.children(recursive=True)
                    rss = supervisor.memory_info().rss + sum(p.memory_info().rss for p in tree if p.is_running())
                    peak = max(peak, rss)
                    cpu = sum(supervisor.cpu_times()[:2]) - parent_cpu + sum(sum(p.cpu_times()[:2]) for p in tree if p.is_running())
                    if rss > MAX_RSS:
                        reason = 'sampled process tree RSS cap exceeded'
                    elif time.monotonic() - start >= LIMIT or cpu >= LIMIT:
                        reason = 'wall or CPU budget exceeded'
                    if reason:
                        kill_tree(process)
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(1)
            code = proc.wait()
            part_count = len(list((OUT / 'parts').glob('*.npz'))) if (OUT / 'parts').is_dir() else 0
            if reason is None and code == 0 and part_count == 30 and all((OUT / n).is_file() for n in ('result.json', 'scores.npz')):
                status = 'complete'
            elif reason is None:
                reason = 'worker failed or required output missing'
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
        if proc is not None and proc.poll() is None:
            kill_tree(psutil.Process(proc.pid))
            proc.wait()
    finally:
        with (OUT / 'receipt.json').open('x', encoding='utf-8') as f:
            completed_units = len(list((OUT / 'parts').glob('*.npz'))) if (OUT / 'parts').is_dir() else 0
            json.dump({'status': status, 'reason': reason, 'returncode': code,
                       'wall_seconds': time.monotonic() - start, 'sampled_cpu_seconds': cpu,
                       'peak_rss_bytes': peak, 'limit_seconds': LIMIT,
                       'min_available_bytes': MIN_AVAIL, 'max_rss_bytes': MAX_RSS,
                       'completed_units': completed_units, 'total_units': 30}, f, indent=2)
    if status != 'complete':
        raise RuntimeError('R21 failed; claim retained')


def synthetic_worker():
    # Deliberately use a different development order to exercise label mapping.
    data = {'labels': list(reversed(R16_LABELS)), 'texts': [], 'truth': [], 'fold_id': [], 'groups': []}
    for fold in range(5):
        for label in range(18):
            data['texts'].append(f'Synthetic group{fold} common prose classword{label} distinct{label} repeated repeated.')
            data['truth'].append(label)
            data['fold_id'].append(fold)
            data['groups'].append(f'group{fold}')
    sources = {k: [] for k in ('baseline', 'extra', 'open', 'v15_extra')}
    calculate(data, sources, OUT, {'synthetic': True})
    with np.load(OUT / 'scores.npz', allow_pickle=False) as scores:
        assert scores['r16_svc4'].shape == (90, 18)
        np.testing.assert_array_equal(scores['row_index'], np.arange(90))
        for condition in WEIGHTS:
            for fold in range(5):
                idx = scores['fold'] == fold
                weighted = {m: scores[f'{condition}_{m}'][idx] * WEIGHTS[condition][m] for m in MEMBERS}
                train_y = np.array([R16_LABELS.index(data['labels'][i]) for i in scores['truth'][~idx]])
                raw, balanced = balance(weighted, train_y, condition)
                np.testing.assert_allclose(scores[f'{condition}_blend'][idx], raw)
                np.testing.assert_allclose(scores[f'{condition}_balanced'][idx], balanced)
    result = json.loads((OUT / 'result.json').read_text())
    assert result['synthetic'] and len(result['stages']) == 45
    assert 'adopt' in result['gate'] and 'adopt' not in result['diagnostics']['unbalanced_blend']


def smoke():
    global OUT
    original = OUT
    try:
        with tempfile.TemporaryDirectory() as tmp:
            OUT = Path(tmp)
            supervise(synthetic=True)
            receipt = json.loads((OUT / 'receipt.json').read_text())
            claim = json.loads((OUT / 'claim.json').read_text())
            assert receipt['status'] == 'complete' and claim['synthetic']
    finally:
        OUT = original
    print(json.dumps({'kind': 'synthetic_smoke_no_score', 'status': 'complete'}))


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run', action='store_true')
    choice.add_argument('--smoke', action='store_true')
    choice.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    choice.add_argument('--synthetic-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--resume-from', help='verified checkpoints from a prior separately approved run')
    args = parser.parse_args()
    if args.resume_from and not args.run:
        parser.error('--resume-from requires --run')
    if args.worker or args.synthetic_worker:
        if os.environ.get('R21_SUPERVISOR_PID') != str(os.getppid()) or not (OUT / 'claim.json').is_file():
            parser.error('worker must be launched by claiming supervisor')
        if args.synthetic_worker:
            if not json.loads((OUT / 'claim.json').read_text())['synthetic']:
                parser.error('synthetic worker requires synthetic claim')
            synthetic_worker()
        else:
            if json.loads((OUT / 'claim.json').read_text())['synthetic']:
                parser.error('real worker requires real claim')
            worker()
    elif args.smoke:
        smoke()
    else:
        supervise(resume_from=args.resume_from)


if __name__ == '__main__':
    main()
