"""R19 preparation only: deterministic answer-similarity audit, not a CV runner."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

DEV = Path('D:/kaggle/name_that_llm/outputs/worktrees/team-e654-r15-20261029/outputs/r15_dev_export/development.json')
BASE = Path('D:/kaggle/name_that_llm/data/external/dhruvtibarewal__nlm-matched-clean__v3')
OPEN = Path('D:/kaggle/name_that_llm/data/external/crtproduction__name-that-llm-open-pool__v11')
FILES = {'development': (DEV, '1c9446efa6c8c477bd68448f1de212aadcc8dc90224b76728ec2f70e2ec2334f'),
         'baseline': (BASE / 'nlm_pool_v5.parquet', '040afb3609d7f6586d57c89ddb42d9bf4b58cc26426104b57390bdf6312b4650'),
         'open': (OPEN / 'ntl_open_pool.parquet', '38e48b40e1db0ab1bf6e4f29d3697e37307c49d1ae5020a7af3cd59c484657d5'),
         'extra': (BASE / 'nlm_matched_clean.parquet', '000f8d541f7c4c5942a18c4fcf557aa01756882931307414cf2530ac21980482')}
THRESHOLD = 0.90
DIAGNOSTIC = 0.85


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_pool(name):
    """Reproduce the public Open Pool setup, then return only text/model."""
    columns = ['text', 'model'] + (['flag_train_prompt_regen'] if name == 'open' else [])
    df = pd.read_parquet(FILES[name][0], columns=columns)
    removed = 0
    if name == 'open':
        if df['flag_train_prompt_regen'].isna().any():
            raise ValueError('missing train-prompt flag')
        flagged = df['flag_train_prompt_regen'].astype(bool)
        removed = int(flagged.sum())
        df = df.loc[~flagged]
    return df[['text', 'model']].copy(), removed


def vectorizer():
    return HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=2**20,
                             alternate_sign=False, norm='l2', lowercase=True,
                             dtype=np.float32)


def similarities(valid, candidates, batch=2000):
    """Maximum cosine to any held-out answer; empty vectors are unmatched."""
    v = vectorizer()
    held = v.transform(valid)
    result = np.zeros(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), batch):
        x = v.transform(candidates[start:start + batch])
        scores = x @ held.T
        result[start:min(start + batch, len(candidates))] = scores.max(axis=1).toarray().ravel()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='synthetic inputs only')
    p.add_argument('--audit', action='store_true', help='count overlaps only, never fit SVC')
    args = p.parse_args()
    start = time.perf_counter()
    if args.smoke:
        scores = similarities(['A small synthetic example'], ['A small synthetic example', 'completely different words'])
        assert scores[0] >= THRESHOLD and scores[1] < THRESHOLD
        print(json.dumps({'kind': 'synthetic_smoke', 'elapsed_seconds': time.perf_counter() - start}))
        return
    if not args.audit:
        p.error('choose --smoke or --audit; real CV is not implemented')
    for name, (path, expected) in FILES.items():
        if digest(path) != expected:
            raise RuntimeError(f'hash mismatch: {name}')
    data = json.loads(DEV.read_text(encoding='utf-8'))
    if len(data['texts']) != 2159 or len(set(data['groups'])) != 120:
        raise RuntimeError('unexpected development shape')
    folds = np.asarray(data['fold_id'])
    if hashlib.sha256(json.dumps(data['fold_id'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest() != '96040086312dca7ce1d3d14bb3ee59640870ef35edba42c746c7f84ae7b827da':
        raise RuntimeError('fold hash mismatch')
    sources = {name: read_pool(name) for name in ('baseline', 'open', 'extra')}
    result = {'threshold': THRESHOLD, 'diagnostic_threshold': DIAGNOSTIC, 'method': 'word 1-2 gram hashed count cosine',
              'open_flag_removed': sources['open'][1], 'folds': []}
    for f in range(5):
        valid = [t for t, fold in zip(data['texts'], folds) if fold == f]
        counts = {}
        for name in ('baseline', 'open', 'extra'):
            table = sources[name][0]
            s = similarities(valid, table['text'].fillna('').astype(str).tolist())
            counts[name] = {'excluded': int((s >= THRESHOLD).sum()), 'diagnostic_085_to_090': int(((s >= DIAGNOSTIC) & (s < THRESHOLD)).sum())}
        result['folds'].append(counts)
    result['elapsed_seconds'] = time.perf_counter() - start
    target = Path('outputs/r19/overlap_audit_v3.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
