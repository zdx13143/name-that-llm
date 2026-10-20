"""原创方案：prompt 中心化的稀疏 n-gram 分类器（centered stylegram）。

想法来源
--------
同一个 prompt 下 18 个模型的回答共享大量话题词。任何按 n-gram 训练的判别模型
都会先抓住这些话题词，而它们在测试集里**全部换成新话题**。这正是换题失效的根源。

公开方案里「remove the prompt」是在**神经特征**上减组均值。没人把它做在**稀疏
n-gram 空间**里。而这个组合有三个有利条件：

1. 稀疏 TF-IDF 的组均值就是话题方向，减掉它等于做了一次「话题投影消除」
2. 我们已验证 **TF-IDF 聚类恢复 prompt 分组纯度 0.986**，可以在测试集上无监督分组
3. 它**不依赖「测试集 18 平衡」这个已被证伪的假设**——中心化是软操作，
   分组不完美只是减了一个略微偏掉的均值，不会像硬分配那样把整组带崩

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
    """每个组减去组均值（组内 L2 归一化后再减，等价于去掉话题方向）。

    稀疏矩阵不支持广播减法，所以把组均值铺成一个同形的稀疏行重复矩阵再相减，
    避免把 block 变成稠密（300k 维 × 组内行数会爆内存）。
    """
    X = normalize(X).tocsr()
    out = X.copy().astype(np.float64)
    n_feat = out.shape[1]
    for g in np.unique(groups):
        m = groups == g
        n_rows = int(m.sum())
        if n_rows < 2:
            continue
        block = out[m]
        mean = np.asarray(block.mean(0)).ravel()
        nz = np.nonzero(mean)[0]
        if len(nz) == 0:
            continue
        rep = sparse.csr_matrix(
            (mean[nz], (np.zeros(len(nz), dtype=np.int64), nz)),
            shape=(n_rows, n_feat),
        )
        out[m] = block - rep
    return normalize(out).tocsr()


def cluster_texts(texts, n_groups=None, dim=200, seed=0) -> np.ndarray:
    """用词级 TF-IDF 聚类恢复 prompt 分组（已验证纯度 0.986）。"""
    n = len(texts)
    k = n_groups or max(1, int(round(n / K)))
    k = max(1, min(k, n - 1)) if n > 1 else 1
    X = TfidfVectorizer(ngram_range=(1, 1), min_df=2, stop_words="english",
                        sublinear_tf=True, max_features=60_000).fit_transform(texts)
    dim = min(dim, max(2, X.shape[1] - 1), n - 1)
    S = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(normalize(S))


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    y = np.array([LABEL2IDX[m] for m in df["model"]])
    groups = df["prompt_id"].to_numpy()
    texts = df["text"].astype(str).tolist()

    folds = list(GroupKFold(5).split(texts, y, groups))
    n_prompts = df["prompt_id"].nunique()

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
            g_va = cluster_texts(va_t, n_groups=len(np.unique(groups[va])))
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
