"""R16 V2 four-view SVC; CV applies an additional prior overlap filter."""
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
R16_LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
              'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
              'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
              'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']


def shape(t):
    return re.sub(r'[^\W\d_]+', lambda m: 'X' if m[0].isupper() else 'x', t)


def function(t):
    return ' '.join(w.lower() if w.lower() in FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))


def norm(t):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(t or ''))).strip()


def build_fit_frame(official, extra, pool, validation, labels, extra_w=0.5, sim=True):
    """R16's EXTRA_DF preparation, then its SVC row construction, in order."""
    train = pd.DataFrame(official, columns=['text', 'model'])
    train['text'] = train.text.fillna('').astype(str)
    external = pd.DataFrame(extra, columns=['text', 'model'])
    external = external[external.model.isin(labels)].copy()
    external['text'] = external.text.fillna('').astype(str)
    if sim:
        held_keys = {norm(t) for t in validation}
        external = external[~external.text.map(norm).isin(held_keys)].copy()
    seen = {norm(t) for t in train.text.values}
    keep = []
    for text in external.text.values:
        key = norm(text)
        ok = bool(key) and key not in seen
        keep.append(ok)
        if ok:
            seen.add(key)
    extra_df = external.loc[keep, ['text', 'model']].reset_index(drop=True)

    official_df = train[['text', 'model']].copy()
    official_df['text'] = official_df.text.fillna('').astype(str)
    official_df['_srcw'] = 1.0
    parts = [official_df]
    seen = {norm(t) for t in official_df.text.values}
    if len(extra_df):
        e = extra_df[['text', 'model']].copy()
        e['_srcw'] = float(extra_w)
        keep = []
        for text in e.text.values:
            key = norm(text)
            ok = bool(key) and key not in seen
            keep.append(ok)
            if ok:
                seen.add(key)
        parts.append(e.loc[keep].copy())
    p = pd.DataFrame(pool, columns=['text', 'model'])
    p = p[p.model.isin(labels)][['text', 'model']].copy()
    p['text'] = p.text.fillna('').astype(str)
    p['_srcw'] = 0.35
    if sim:
        held_keys = {norm(t) for t in validation}
        p = p[~p.text.map(norm).isin(held_keys)].copy()
    keep = []
    for text in p.text.values:
        key = norm(text)
        ok = bool(key) and key not in seen
        keep.append(ok)
        if ok:
            seen.add(key)
    parts.append(p.loc[keep].copy())
    return pd.concat(parts, ignore_index=True)


def fit_predict(official, extra, pool, validation, labels):
    """CV uses sim=True, treating its validation texts as R16's held-out test."""
    frame = build_fit_frame(official, extra, pool, validation, labels, extra_w=0.5, sim=True)
    if set(frame.model) != set(labels):
        raise ValueError('missing or unexpected training class')
    x = frame.text.astype(str).tolist()
    y = frame.model.map({label: i for i, label in enumerate(labels)}).to_numpy(dtype=int)
    weights = frame['_srcw'].to_numpy(dtype=np.float32)
    train_matrices, test_matrices = [], []
    views = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
    for view, lo, hi, maximum, weight in views:
        prep = shape if view == 'shape' else function if view == 'function' else str
        analyzer = 'char' if view == 'shape' else 'word' if view == 'function' else view
        options = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False,
                       sublinear_tf=True, min_df=2, max_features=maximum, dtype=np.float32)
        if analyzer == 'word':
            options['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
        vec = TfidfVectorizer(**options)
        train_matrices.append(vec.fit_transform([prep(t) for t in x]) * weight)
        test_matrices.append(vec.transform([prep(t) for t in validation]) * weight)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
    svc.fit(hstack(train_matrices, format='csr'), y, sample_weight=weights)
    return svc.predict(hstack(test_matrices, format='csr'))
