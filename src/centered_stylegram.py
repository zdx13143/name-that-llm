"""原创方案：prompt 中心化的稀疏 n-gram 分类器（centered stylegram）。

想法来源
--------
同一个 prompt 下 18 个模型的回答共享大量话题词。任何按 n-gram 训练的判别模型
都会先抓住这些话题词，而它们在测试集里**全部换成新话题**。这正是换题失效的根源。

公开方案里「remove the prompt」是在**神经特征**上减组均值。没人把它做在**稀疏
n-gram 空间**里。而这个组合有三个有利条件：

1. 稀疏 TF-IDF 的组均值就是话题方向，减掉它等于做了一次「话题投影消除」
2. 我们已验证 **TF-IDF 聚类恢复 prompt 分组纯度 0.986**，可以在测试集上无监督分组
3. 中心化不强制每组标签一一对应，但默认簇数仍按 n/18 估计，
   **这是待验证的组大小假设**，不能视为已知的测试集结构。

诚实的验证口径
--------------
训练集里中心化用真实 prompt_id（我们本来就知道）。
验证折里中心化必须用**该折内部无监督聚类**得到的分组 —— 这才模拟测试时的真实情形。
如果用真实 prompt_id 去中心化验证折，就等于偷看了分组，分数会虚高。

用法：
    python src/centered_stylegram.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL2IDX, LABELS, ROOT  # noqa: E402

K = len(LABELS)


def build_features(texts_tr, texts_te):
    """char_wb 2-5 + word 1-2，L2 归一化后加权拼接。"""
    out_tr, out_te = [], []
    for analyzer, ng, w in (("char_wb", (2, 5), 1.0), ("word", (1, 2), 0.5)):
        opts = dict(analyzer=analyzer, ngram_range=ng, lowercase=True, sublinear_tf=True,
                    min_df=2, max_features=300_000, dtype=np.float64)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**opts).fit(texts_tr)
        out_tr.append(normalize(v.transform(texts_tr)) * w)
        out_te.append(normalize(v.transform(texts_te)) * w)
    return sparse.hstack(out_tr).tocsr(), sparse.hstack(out_te).tocsr()


def center_by_groups(X: sparse.csr_matrix, groups: np.ndarray) -> sparse.csr_matrix:
    """先逐行归一化，再对多行组的每一行减组均值，最后逐行归一化。

    单行组没有可估计的共享话题，保留其归一化向量；不修改输入矩阵。
    用稀疏重复均值，避免把整个组变成稠密矩阵。
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != X.shape[0]:
        raise ValueError("groups must have one entry per input row")
    X = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    if X.shape[0] == 0:
        return X
    X = normalize(X, copy=False).tocsr()
    blocks, row_order = [], []
    for g in np.unique(groups):
        indices = np.flatnonzero(groups == g)
        block = X[indices]
        if len(indices) > 1:
            mean = sparse.csr_matrix(block.mean(axis=0))
            repeated_mean = sparse.vstack([mean] * len(indices), format="csr")
            block = block - repeated_mean
        blocks.append(block)
        row_order.extend(indices)
    if len(row_order) != X.shape[0]:
        raise ValueError("groups must not contain missing values")
    out = sparse.vstack(blocks, format="csr")[np.argsort(row_order)]
    return normalize(out, copy=False).tocsr()


def cluster_texts(texts, n_groups=None, dim=200, seed=0) -> np.ndarray:
    """仅从输入文本估计分组；默认 n/18 是组大小假设，不是已知题数。

    空输入返回空标签；单行、单簇或无可用词汇时返回一个组。
    n_groups 只应来自预先固定的设置，不能传验证集真实 prompt 数。
    """
    n = len(texts)
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    k = max(1, int(round(n / K))) if n_groups is None else int(n_groups)
    if k < 1:
        raise ValueError("n_groups must be positive")
    k = min(k, n)
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    try:
        X = TfidfVectorizer(ngram_range=(1, 1), min_df=2, stop_words="english",
                            sublinear_tf=True, max_features=60_000).fit_transform(texts)
    except ValueError as exc:
        if "empty vocabulary" in str(exc) or "After pruning, no terms remain" in str(exc):
            return np.zeros(n, dtype=np.int64)
        raise
    if X.shape[1] < 2:
        return np.zeros(n, dtype=np.int64)
    dim = min(dim, X.shape[1] - 1, n - 1)
    S = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(normalize(S))


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    y = np.array([LABEL2IDX[m] for m in df["model"]])
    groups = df["prompt_id"].to_numpy()
    texts = df["text"].astype(str).tolist()

    folds = list(GroupKFold(5).split(texts, y, groups))
    print("分组提示：验证侧仅看文本，默认簇数按 n/18 估计；这是待验证的假设。")

    for C in (2.0, 8.0, 16.0):
        oof_plain = np.zeros((len(y), K))
        oof_cent = np.zeros((len(y), K))
        t0 = time.time()
        for tr, va in folds:
            tr_t = [texts[i] for i in tr]
            va_t = [texts[i] for i in va]
            A, B = build_features(tr_t, va_t)

            # 1) 普通（不中心化）—— 基线
            clf = LogisticRegression(C=C, max_iter=2000,
                                     class_weight="balanced")
            clf.fit(A, y[tr])
            oof_plain[va] = clf.predict_proba(B)

            # 2) 训练侧用真实 prompt_id 中心化；验证侧用折内无监督聚类中心化
            A_c = center_by_groups(A, groups[tr])
            g_va = cluster_texts(va_t)
            B_c = center_by_groups(B, g_va)

            clf2 = LogisticRegression(C=C, max_iter=2000,
                                      class_weight="balanced")
            clf2.fit(A_c, y[tr])
            oof_cent[va] = clf2.predict_proba(B_c)

        s_plain = balanced_accuracy_score(y, oof_plain.argmax(1))
        s_cent = balanced_accuracy_score(y, oof_cent.argmax(1))
        print(f"C={C:<5} 普通 {s_plain:.4f}   中心化 {s_cent:.4f}   "
              f"差 {s_cent - s_plain:+.4f}   ({time.time() - t0:.0f}s)")

        np.save(ROOT / "outputs" / f"oof_cstyle_{C:g}.npy", oof_cent)
        np.save(ROOT / "outputs" / f"oof_plain_{C:g}.npy", oof_plain)

    print("\n对照：我们的 blend_v3（8 视图+嵌入）0.3978，公开集成 0.6")
    print("关键看两件事：① 中心化是否稳定优于普通 ② 与既有模型的相关性是否够低")


if __name__ == "__main__":
    main()
