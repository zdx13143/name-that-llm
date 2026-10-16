"""分层建模：先判「哪家写的」，再在机构内部判「哪个型号」。

动机
----
误差分析显示：机构层面准确率 0.707，型号层面只有 0.343。
也就是说**大量错误是「认对了家、认错了型号」**。
扁平 18 分类器要在一次 argmax 里同时做两件难度完全不同的事：
  - 跨机构区分（容易，标点指纹几乎一眼可辨）
  - 同机构内部区分（极难，Anthropic/ZAI 那 7 个模型指标完全重合）

分层把这两件事拆开：第 1 步只需要学会「Google 从不用 em dash」这类强信号，
不必被第 2 步的噪声干扰；第 2 步则只在同机构的小样本里做精细区分，
搜索空间从 18 类降到 2~5 类。

用法：
    python src/run_hier.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv as cvmod  # noqa: E402
from common import LABELS, LAB_OF, LABEL2IDX, ensure_outputs, load_train  # noqa: E402
from features import extract_features  # noqa: E402
from run_sweep import build_folds, make_model, view_matrices  # noqa: E402

N_CLASSES = len(LABELS)
LABS = sorted(set(LAB_OF.values()))
LAB_IDX = {l: i for i, l in enumerate(LABS)}

# 每个全局类别所属的机构序号
CLASS_TO_LAB = np.array([LAB_IDX[LAB_OF[m]] for m in LABELS])
LAB_MEMBERS = {l: [i for i, m in enumerate(LABELS) if LAB_OF[m] == l] for l in LABS}


def run(mode: str, folds, y, view: str, kind: str, param: float):
    """mode: 'flat' 扁平 18 分类；'hier' 分层。

    返回 (oof_proba, 机构层面 OOF 预测)。
    """
    n = len(y)
    oof = np.zeros((n, N_CLASSES))
    lab_pred = np.full(n, -1, dtype=int)
    y_lab = CLASS_TO_LAB[y]

    for fold in folds:
        va = fold["va"]
        Atr, Ava = view_matrices(fold, view)
        ytr = fold["y_tr"]

        if mode == "flat":
            model = make_model(kind, param)
            model.fit(Atr, ytr)
            proba = model.predict_proba(Ava)
            cls = model.classes_
            full = np.zeros((Ava.shape[0], N_CLASSES))
            full[:, cls] = proba
            oof[va] = full
            lab_pred[va] = CLASS_TO_LAB[full.argmax(1)]
            continue

        # ---- 第 1 步：机构分类 ----
        lab_model = make_model(kind, param)
        lab_model.fit(Atr, CLASS_TO_LAB[ytr])
        p_lab = lab_model.predict_proba(Ava)  # 列对应 lab_model.classes_
        lab_classes = lab_model.classes_
        p_lab_full = np.zeros((Ava.shape[0], len(LABS)))
        p_lab_full[:, lab_classes] = p_lab

        # ---- 第 2 步：每个机构内部一个专家模型 ----
        p_within = np.zeros((Ava.shape[0], N_CLASSES))
        for lab in LABS:
            members = LAB_MEMBERS[lab]
            if len(members) == 1:
                p_within[:, members[0]] = 1.0
                continue
            mask = np.isin(ytr, members)
            sub_y = ytr[mask]
            if len(np.unique(sub_y)) < 2:
                p_within[:, members[0]] = 1.0
                continue
            sub = make_model(kind, param)
            sub.fit(Atr[mask], sub_y)
            sub_proba = sub.predict_proba(Ava)
            for j, c in enumerate(sub.classes_):
                p_within[:, c] = sub_proba[:, j]

        final = p_lab_full[:, CLASS_TO_LAB] * p_within
        oof[va] = final
        lab_pred[va] = CLASS_TO_LAB[final.argmax(1)]

    return oof, lab_pred, y_lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--views", default="charword,char13")
    args = ap.parse_args()

    out = ensure_outputs()
    df = load_train()
    texts = df["text"].tolist()
    y = np.array([LABEL2IDX[m] for m in df["model"]], dtype=np.int64)
    groups = df["prompt_id"].to_numpy()

    print("提取指纹特征 …")
    fp, _ = extract_features(texts)
    t0 = time.time()
    folds = build_folds(texts, fp, y, groups, n_splits=args.splits)
    print(f"折矩阵缓存完成 {time.time() - t0:.1f}s")

    y_lab = CLASS_TO_LAB[y]
    print(f"\n{'视图':<11}{'模式':<7}{'C':>6}{'型号层面':>10}{'机构层面':>10}")
    results = []
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        for mode in ("flat", "hier"):
            for C in (4.0, 16.0):
                oof, lab_pred, _ = run(mode, folds, y, view, "lr", C)
                ba = float(balanced_accuracy_score(y, oof.argmax(1)))
                la = float(balanced_accuracy_score(y_lab, lab_pred))
                results.append({"view": view, "mode": mode, "C": C,
                                "model_ba": ba, "lab_ba": la, "oof": oof})
                print(f"{view:<11}{mode:<7}{C:>6}{ba:>10.4f}{la:>10.4f}")

    best = max(results, key=lambda r: r["model_ba"])
    print(f"\n最佳：{best['view']} / {best['mode']} / C={best['C']} "
          f"-> 型号 {best['model_ba']:.4f}，机构 {best['lab_ba']:.4f}")

    flat_best = max((r for r in results if r["mode"] == "flat"),
                    key=lambda r: r["model_ba"])
    hier_best = max((r for r in results if r["mode"] == "hier"),
                    key=lambda r: r["model_ba"])
    print(f"  扁平最佳 {flat_best['model_ba']:.4f}  vs  "
          f"分层最佳 {hier_best['model_ba']:.4f}  "
          f"=> {hier_best['model_ba'] - flat_best['model_ba']:+.4f}")

    cvmod.summarize(y, best["oof"].argmax(1), LABELS, LAB_OF,
                    tag=f"分层/扁平最佳（{best['view']}/{best['mode']}）")
    np.save(out / "hier_best_oof.npy", best["oof"])
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
