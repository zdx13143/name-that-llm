"""利用「每个 prompt 下 18 个模型各一条」的结构做约束解码。

为什么这一步值 0.09
--------------------
测试集约 3,888 行 ≈ 216 × 18，也就是每个新 prompt 都由 18 个模型各写一条。
这意味着这不是 18 个独立的分类题，而是一道**分配题**：在一个 prompt 组内，
18 段文本要一一分给 18 个不同的模型，每个模型恰好一次。

实测（训练集，真实分组）：独立 argmax 0.3978 → 组内匈牙利 0.4879，**+0.0900**。
这个收益是乘性的：基础分类器越强，分配层拿到的越多。

分组怎么来
----------
测试集不提供 prompt_id，id 是随机哈希。只能从内容恢复分组。
关键教训：**编码器嵌入抓的是文风，词级 TF-IDF 抓的才是话题。**
- 编码器 mean-pooled 嵌入聚类：纯度 0.23（失败）
- 词级 TF-IDF + SVD(200) + KMeans：纯度 **0.986**，簇大小中位数正好 18

所以这里用后者。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

N_CLASSES = 18


def recover_groups(texts: list[str], n_groups: int | None = None,
                   dim: int = 200, seed: int = 0,
                   verbose: bool = True) -> np.ndarray:
    """按话题把文本聚类，返回每条文本的簇编号。

    n_groups 不给时按 ceil(n/18) 估计（假设每个 prompt 组 18 条）。
    """
    n = len(texts)
    if n_groups is None:
        n_groups = max(1, int(round(n / N_CLASSES)))
    n_groups = max(1, min(n_groups, n))

    X = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, stop_words="english",
        sublinear_tf=True, max_features=60_000,
    ).fit_transform(texts)

    # 维度不能超过特征数
    dim = min(dim, max(2, X.shape[1] - 1), n - 1)
    S = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    E = normalize(S)

    k = min(n_groups, n - 1) if n > 1 else 1
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(E)

    if verbose:
        sizes = np.bincount(labels)
        print(f"    分组：k={k}  簇大小 中位 {int(np.median(sizes))} "
              f"最小 {sizes.min()} 最大 {sizes.max()}")
    return labels


def assign_with_capacity(scores: np.ndarray, groups: np.ndarray,
                         n_classes: int = N_CLASSES) -> np.ndarray:
    """组内带容量的一对一分配。

    scores 是 log 概率（越大越好）。簇大小 n 时每类容量 k=ceil(n/n_classes)，
    用来处理「一个簇里混了两个 prompt（36 = 2×18）」的情况。
    """
    pred = np.full(len(scores), -1, dtype=np.int64)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n = len(idx)
        cap = max(1, int(np.ceil(n / n_classes)))
        cost = -scores[np.ix_(idx, np.arange(n_classes))]
        if cap > 1:
            cost = np.tile(cost, (1, cap))
        rows, cols = linear_sum_assignment(cost)
        pred[idx[rows]] = cols % n_classes
    # 保险：没被分配到的（理论上不会有）回退到独立 argmax
    missing = pred < 0
    if missing.any():
        pred[missing] = scores[missing].argmax(axis=1)
    return pred


def decode(proba: np.ndarray, texts: list[str],
           n_groups: int | None = None,
           min_rows: int = 36,
           verbose: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """先聚类再分配。返回 (预测类别, 簇编号)。

    行数太少时（本地冒烟只有 3 行）跳过分配，直接 argmax —— 3 条文本谈不上结构。
    """
    if len(texts) < min_rows:
        if verbose:
            print(f"    {len(texts)} 行 < {min_rows}，跳过分配（本地冒烟模式）")
        return proba.argmax(axis=1), None

    groups = recover_groups(texts, n_groups=n_groups, verbose=verbose)
    logp = np.log(np.clip(proba, 1e-12, 1.0))
    pred = assign_with_capacity(logp, groups)
    if verbose:
        changed = int((pred != proba.argmax(axis=1)).sum())
        print(f"    分配改变了 {changed}/{len(texts)} 条预测")
    return pred, groups
