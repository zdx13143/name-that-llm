"""调结构约束解码的超参：分数变换、组数、簇大小策略。

分配层是乘性的 —— 基础概率只要稍微变准一点，分配后会被放大。
而解码器本身有几个没定死的选择：

1. **分数变换**：现在是直接用 log(p)。但我们的融合概率高度饱和
   （多数视图 train acc = 1.000），log(p) 的差距被拉得很大，
   分配会变得近乎"认死理"。温度缩放一下可能更稳。
2. **组数 k**：测试集行数只能估（约 3,888 → k≈216），估偏了会分错组。
   这里看 k 在真值附近抖动时分数掉多少，判断这个超参敏不敏感。
3. **簇大小策略**：簇恰好 18 条时是干净的一对一；混了 36 条时要靠容量。
   也许只在"干净簇"上分配、其余回退 argmax 更安全。

用法：
    python src/tune_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LABEL2IDX, ROOT  # noqa: E402

N_CLASSES = 18


def assign(scores: np.ndarray, groups: np.ndarray,
           only_clean: bool = False) -> np.ndarray:
    """组内带容量分配。only_clean=True 时只处理大小恰为 18 的簇。"""
    pred = np.full(len(scores), -1, dtype=np.int64)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n = len(idx)
        if only_clean and n != N_CLASSES:
            continue
        cap = max(1, int(np.ceil(n / N_CLASSES)))
        cost = -scores[np.ix_(idx, np.arange(N_CLASSES))]
        if cap > 1:
            cost = np.tile(cost, (1, cap))
        rows, cols = linear_sum_assignment(cost)
        pred[idx[rows]] = cols % N_CLASSES
    missing = pred < 0
    if missing.any():
        pred[missing] = scores[missing].argmax(axis=1)
    return pred


def transform(p: np.ndarray, mode: str) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    if mode == "log":
        return np.log(p)
    if mode == "p":
        return p
    if mode == "sqrt":
        return np.sqrt(p)
    if mode.startswith("T"):
        t = float(mode[1:])
        q = np.power(p, 1.0 / t)
        q /= q.sum(axis=1, keepdims=True)
        return np.log(q)
    if mode == "rank":
        return np.argsort(np.argsort(-p, axis=1), axis=1) * -1.0
    raise ValueError(mode)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    texts = df["text"].astype(str).tolist()
    y = np.array([LABEL2IDX[m] for m in df["model"]])
    oof = np.load(ROOT / "outputs" / "blend_v3_oof.npy")

    base = balanced_accuracy_score(y, oof.argmax(1))
    print(f"独立 argmax = {base:.4f}\n")

    # 聚类一次，后面反复用
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english",
                        sublinear_tf=True, max_features=60_000).fit_transform(texts)
    S = TruncatedSVD(n_components=200, random_state=0).fit_transform(X)
    E = normalize(S)

    print("=== 1) 分数变换（k=150）===")
    groups = KMeans(n_clusters=150, n_init=10, random_state=0).fit_predict(E)
    best_t, best_s = "log", -1.0
    for mode in ("log", "p", "sqrt", "rank", "T0.25", "T0.5", "T2", "T4", "T8"):
        s = balanced_accuracy_score(y, assign(transform(oof, mode), groups))
        flag = ""
        if s > best_s:
            best_s, best_t = s, mode
            flag = "  <-"
        print(f"  {mode:<6} {s:.4f}{flag}")
    print(f"  最佳变换 = {best_t}（{best_s:.4f}）")

    print("\n=== 2) 组数 k 敏感性（用最佳变换）===")
    for k in (135, 140, 145, 150, 155, 160, 175):
        g = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(E)
        s = balanced_accuracy_score(y, assign(transform(oof, best_t), g))
        print(f"  k={k:<4} {s:.4f}")

    print("\n=== 3) 簇大小策略（k=150）===")
    for only_clean in (False, True):
        s = balanced_accuracy_score(
            y, assign(transform(oof, best_t), groups, only_clean=only_clean))
        print(f"  only_clean={only_clean}  {s:.4f}")


if __name__ == "__main__":
    main()
