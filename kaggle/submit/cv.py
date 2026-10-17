"""按 prompt 分组的交叉验证与评估。

为什么必须按 prompt 分组
------------------------
测试集的 prompt 全部是训练集里从未出现过的。如果按行随机切分，
同一个 prompt 下的 18 段文本会同时出现在训练侧与验证侧，
模型只要记住「这道题在聊什么」就能蒙对，验证分会虚高十几个点，
但排行榜上完全拿不到 —— 这是本赛最常见的翻车方式。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold


def _take(X, idx):
    """按行取子集，同时兼容 numpy 数组与 pandas DataFrame。

    坑：对 DataFrame 直接写 X[idx] 取的是「列」不是「行」，会抛 KeyError。
    """
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def oof_predictions(
    build_model: Callable,
    X,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_splits: int = 5,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """跑 prompt 分组 CV，返回 (oof_pred, oof_proba, 每折 balanced accuracy)。

    oof_proba 只有在模型支持 predict_proba 时才有值，否则全零。
    """
    gkf = GroupKFold(n_splits=n_splits)
    n = len(y)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = np.zeros((n, n_classes), dtype=np.float64)
    fold_scores: list[float] = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        model = build_model()
        model.fit(_take(X, tr), y[tr])
        oof_pred[va] = model.predict(_take(X, va))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(_take(X, va))
            # 某些模型在训练折里没见过某类时，predict_proba 的列会变窄，需对齐
            classes = getattr(model, "classes_", None)
            if classes is not None and proba.shape[1] == len(classes):
                oof_proba[np.ix_(va, classes)] = proba
            else:
                oof_proba[va] = proba
        score = balanced_accuracy_score(y[va], oof_pred[va])
        fold_scores.append(float(score))
        if verbose:
            print(
                f"  fold {fold}/{n_splits}  验证 {len(va):>4} 条  "
                f"balanced acc = {score:.4f}"
            )

    if verbose:
        arr = np.asarray(fold_scores)
        print(f"  平均 {arr.mean():.4f}  标准差 {arr.std():.4f}")

    return oof_pred, oof_proba, fold_scores


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray,
                     names: list[str]) -> pd.DataFrame:
    """每类召回率 —— 指标是 balanced accuracy，必须逐类看短板。"""
    rows = []
    for idx, name in enumerate(names):
        mask = y_true == idx
        support = int(mask.sum())
        correct = int((y_pred[mask] == idx).sum()) if support else 0
        rows.append(
            {
                "model": name,
                "support": support,
                "correct": correct,
                "recall": correct / support if support else float("nan"),
                "pred_count": int((y_pred == idx).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("recall")


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray,
                    names: list[str], top: int = 20) -> pd.DataFrame:
    """最容易混淆的 (真实 -> 预测) 组合，用于定位误差来源。"""
    from collections import Counter

    pairs = Counter(
        (names[t], names[p])
        for t, p in zip(y_true, y_pred)
        if t != p
    )
    total = max(len(y_true), 1)
    rows = [
        {"true": a, "pred": b, "count": c, "share": c / total}
        for (a, b), c in pairs.most_common(top)
    ]
    return pd.DataFrame(rows)


def lab_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                 names: list[str], lab_of: dict[str, str]) -> float:
    """机构层面的准确率 —— 这个数字高说明「哪家写的」已经认出来了，
    剩下的难度全在同机构内部的相近型号上。"""
    labs = sorted(set(lab_of.values()))
    lab_idx = {l: i for i, l in enumerate(labs)}
    mapped = np.array([lab_idx[lab_of[n]] for n in names])
    hit = 0
    for t, p in zip(y_true, y_pred):
        if p < 0:
            continue
        if mapped[t] == mapped[p]:
            hit += 1
    return hit / total_safe(len(y_true))


def total_safe(x: int) -> int:
    return x if x else 1


def summarize(y_true: np.ndarray, y_pred: np.ndarray, names: list[str],
              lab_of: dict[str, str], tag: str = "") -> dict:
    """一次性打印并返回完整评估摘要。"""
    ba = balanced_accuracy_score(y_true, y_pred)
    covered = int((y_pred >= 0).sum())
    print(f"\n{'=' * 66}")
    print(f"结果摘要 {tag}")
    print(f"{'=' * 66}")
    print(f"balanced accuracy (OOF) : {ba:.4f}")
    print(f"18 选 1 随机期望        : {1 / len(names):.4f}")
    print(f"机构层面准确率          : {lab_accuracy(y_true, y_pred, names, lab_of):.4f}")
    print(f"覆盖样本                : {covered}/{len(y_true)}")

    recall = per_class_recall(y_true, y_pred, names)
    print("\n各类召回率（由低到高，前 8 个是短板）：")
    print(recall.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n召回率最高的 3 类：")
    print(recall.tail(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    conf = confusion_table(y_true, y_pred, names, top=12)
    print("\n最常混淆的组合：")
    print(conf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    return {
        "balanced_accuracy": ba,
        "lab_accuracy": lab_accuracy(y_true, y_pred, names, lab_of),
        "recall": recall,
        "confusions": conf,
    }
