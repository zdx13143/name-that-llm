"""把微调模型的 OOF 和现有 sklearn 融合的 OOF 合到一起评。

两个来源的信息性质不同：
- sklearn 融合：8 个 TF-IDF 视图 + 2 个编码器冻结嵌入 → 抓的是**表层书写习惯**
- 微调模型：编码器整体微调 → 抓的是**上下文相关的写法**
两者互补的可能性很大，所以先看单用和合用各自的诚实分数。

诚实评估口径：拿 4 折的 OOF 选权重，在被留出的那折上评分（留一折），
避免"用验证集选超参"导致的乐观偏差。

用法：
    python src/blend_final.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LABEL2IDX, ROOT  # noqa: E402

N_CLASSES = 18


def greedy_weights(pool: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray,
                   rounds: int = 40, step: float = 0.04) -> dict[str, float]:
    """贪心往池子里加权重，只按给定 idx 上的分数选。"""
    names = list(pool)
    w = {k: 0.0 for k in names}
    cur = np.zeros((len(idx), N_CLASSES))
    best = -1.0
    acc = 0.0
    for _ in range(rounds):
        gains = []
        for k in names:
            cand = (cur + step * pool[k][idx])
            s = balanced_accuracy_score(y[idx], cand.argmax(1))
            gains.append((s, k))
        s, k = max(gains)
        if s <= best + 1e-6:
            break
        cur = cur + step * pool[k][idx]
        acc += step
        w[k] += step
        best = s
    if acc > 0:
        w = {k: v / acc for k, v in w.items()}
    return w


def score(pool: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray,
          w: dict[str, float]) -> float:
    blend = sum(w[k] * pool[k][idx] for k in pool if w.get(k, 0) > 0)
    return balanced_accuracy_score(y[idx], blend.argmax(1))


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    y = np.array([LABEL2IDX[m] for m in df["model"]])
    groups = df["prompt_id"].to_numpy()

    pool: dict[str, np.ndarray] = {}
    p_sk = ROOT / "outputs" / "blend_v3_oof.npy"
    if p_sk.exists():
        pool["sklearn"] = np.load(p_sk)
    p_ft = ROOT / "outputs" / "oof_clf.npy"
    if not p_ft.exists():
        p_ft = Path("oof_clf.npy")
    if p_ft.exists():
        pool["finetune"] = np.load(p_ft)

    if not pool:
        raise SystemExit("没有找到任何 OOF 文件")

    print("各来源单独分数：")
    for k, v in pool.items():
        print(f"  {k:<10} {balanced_accuracy_score(y, v.argmax(1)):.4f}")

    if len(pool) == 1:
        print("\n只有一个来源，无需融合")
        return

    # 全量 OOF 上选权重（乐观），再留一折（诚实）
    best_w = greedy_weights(pool, y, np.arange(len(y)))
    print(f"\n全量 OOF 上的权重（乐观）：{ {k: round(v, 3) for k, v in best_w.items()} }")
    print(f"  全量分数 {score(pool, y, np.arange(len(y)), best_w):.4f}")

    folds = list(GroupKFold(n_splits=5).split(y, y, groups))
    scores = []
    for i, (tr, va) in enumerate(folds, 1):
        w = greedy_weights(pool, y, tr)
        s = score(pool, y, va, w)
        scores.append(s)
        print(f"  留一折 fold {i}: {s:.4f}  权重 { {k: round(v, 2) for k, v in w.items()} }")

    print(f"\n>>> 留一折诚实分数 = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"    对照：纯 sklearn 融合 OOF 0.3978（LB 0.381）")


if __name__ == "__main__":
    main()
