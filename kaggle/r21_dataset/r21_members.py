"""Run frozen notebook CPU member blocks in isolated namespaces."""
import gc
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SOURCE = Path(__file__).resolve().parent / 'r21_source'


class FrameReader:
    def __init__(self, extra, pool):
        self.frames = {'extra.parquet': extra, 'pool.parquet': pool}

    def __getattr__(self, name):
        return getattr(pd, name)

    def read_parquet(self, path, **kwargs):
        return self.frames[path].copy()


def run_member(condition, member, official, extra, pool, validation, labels):
    """Run exact frozen source with weights 1, returning unweighted log scores."""
    if condition not in ('r16', 'areeb') or member not in ('tfidf', 'svc4', 'distort'):
        raise ValueError('unknown source/member')
    train = pd.DataFrame(official, columns=['text', 'model'])
    test = pd.DataFrame({'text': validation})
    if set(train.model) != set(labels) or len(labels) != 18:
        raise ValueError('training fold must contain all 18 classes')
    reader = FrameReader(pd.DataFrame(extra, columns=['text', 'model']),
                         pd.DataFrame(pool, columns=['text', 'model']))
    args = SimpleNamespace(extra='extra.parquet', pool='pool.parquet', extra_w=.5,
                           sim='1', perpassage=1, tfidf_w=1., svc_w=1., distort_w=1.)
    # Notebook checks os.path.exists before reading the in-memory pool frame.
    file_os = SimpleNamespace(path=SimpleNamespace(exists=lambda path: path == 'pool.parquet'))
    env = dict(pd=reader, np=np, sp=sp, re=re, os=file_os, gc=gc,
               train=train, test=test, LABELS=labels, L2I={v: i for i, v in enumerate(labels)},
               y=train.model.map({v: i for i, v in enumerate(labels)}).to_numpy(dtype=int),
               a=args, TfidfVectorizer=TfidfVectorizer, LogisticRegression=LogisticRegression,
               LP={}, log=lambda *args: None)
    for name in ('extra', member):
        path = SOURCE / f'{condition}_{name}.py.txt'
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), env)
    result = np.asarray(env['LP'][member], dtype=np.float64).copy()
    if result.shape != (len(validation), 18) or not np.isfinite(result).all():
        raise ValueError('invalid member output')
    return result
