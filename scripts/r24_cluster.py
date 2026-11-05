"""v15 content clustering, without labels or passage output."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def capped_agglo(S, cap=18, thr=0.15, check=lambda: None):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        check()
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab


def cluster(texts, check=lambda: None):
    if not len(texts):
        return np.array([], dtype=int), True
    try:
        v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
        X = v.fit_transform(texts)
    except ValueError:
        return np.arange(len(texts)), True
    check()
    return capped_agglo((X @ X.T).toarray(), 18, 0.15, check), False
