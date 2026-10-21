"""在开发集比较预训练隐藏状态特征，并与已有视图融合。

流程：
  1. 冻结编码器，对全部训练文本算一次嵌入（缓存复用，不存在标签泄漏）
  2. 用同一套 prompt 分组 CV 评估「嵌入 + 线性分类器」
  3. 把嵌入视图加进 v2 的视图池，重新做加权融合，
     报告留一折权重比较。基础视图已用全量 OOF 选过超参，
     所以这仍是开发对照，不是完整嵌套验证或独立泛化增量。

用法：
    python src/run_embed.py --model microsoft/deberta-v3-small
    python src/run_embed.py --model microsoft/deberta-v3-base --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv as cvmod  # noqa: E402
from common import LABELS, LAB_OF, LABEL2IDX, ensure_outputs, load_train  # noqa: E402
from embeddings import cache_path, extract_embeddings  # noqa: E402
from run_sweep import ba_on, greedy_weights, honest_score, N_CLASSES  # noqa: E402


def oof_from_embeddings(X, y, groups, n_splits, C):
    n = len(y)
    oof = np.zeros((n, N_CLASSES))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                               random_state=0, n_jobs=-1),
        )
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])
    return oof


def embedding_comparison_pools(out_dir: Path, current_oof_path: Path,
                               current_oof: np.ndarray):
    """历史视图作开发对照，当前嵌入仅在增强池中加入一次。

    形状匹配仅用于兼容已有文件；不构成数据行序或模型版本身份校验。
    """
    baseline = {}
    for path in sorted(out_dir.glob("pool_v2_*.npy")):
        arr = np.load(path, allow_pickle=False)
        if arr.shape == current_oof.shape:
            baseline[path.stem.replace("pool_v2_", "")] = arr
    for path in sorted(out_dir.glob("oof_emb_*.npy")):
        if path.resolve() == current_oof_path.resolve():
            continue
        arr = np.load(path, allow_pickle=False)
        if arr.shape == current_oof.shape:
            short = path.stem.replace("oof_emb_", "").replace("answerdotai__", "")
            baseline[f"emb:{short}"] = arr
    augmented = dict(baseline)
    augmented["emb"] = current_oof
    return baseline, augmented


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-blend", action="store_true")
    args = ap.parse_args()

    out = ensure_outputs()
    df = load_train()
    texts = df["text"].tolist()
    y = np.array([LABEL2IDX[m] for m in df["model"]], dtype=np.int64)
    groups = df["prompt_id"].to_numpy()
    priors = np.bincount(y, minlength=N_CLASSES) / len(y)

    cp = Path(cache_path(out, args.model, args.tag))
    if cp.exists():
        X = np.load(cp)
        print(f"读取缓存嵌入 {cp.name}  {X.shape}")
    else:
        print(f"提取嵌入：{args.model}")
        t0 = time.time()
        X = extract_embeddings(
            texts, args.model, batch_size=args.batch_size,
            max_length=args.max_length, verbose=True,
        )
        np.save(cp, X)
        print(f"  耗时 {time.time() - t0:.1f}s，已缓存到 {cp.name}")

    print(f"\n评估嵌入视图（LR C={args.C}）")
    t0 = time.time()
    oof_emb = oof_from_embeddings(X, y, groups, args.splits, args.C)
    s_emb = float(balanced_accuracy_score(y, oof_emb.argmax(axis=1)))
    print(f"  嵌入视图 OOF balanced accuracy = {s_emb:.4f}  ({time.time() - t0:.1f}s)")
    cvmod.summarize(y, oof_emb.argmax(axis=1), LABELS, LAB_OF, tag="嵌入视图")

    # 保存当前 OOF；构建对照池时显式排除此文件。
    safe = args.model.replace("/", "__")
    current_oof_path = out / f"oof_emb_{safe}.npy"
    np.save(current_oof_path, oof_emb)

    # ---- 与 v2 视图池融合 ----
    if args.no_blend:
        return

    pool, pool2 = embedding_comparison_pools(out, current_oof_path, oof_emb)

    if not pool:
        print("\n未找到可融合的视图，跳过（先跑 python src/run_sweep.py）")
        return

    print(f"\n视图池：{list(pool)}")

    base_w, base_alpha, base_s = greedy_weights(pool, y, np.arange(len(y)), priors)
    base_dev, base_std = honest_score(pool, y, groups, priors, n_splits=args.splits)
    print(f"开发对照（不含当前嵌入，可含历史嵌入）：全量 {base_s:.4f} / 留一折 {base_dev:.4f} ± {base_std:.4f}")

    new_w, new_alpha, new_s = greedy_weights(pool2, y, np.arange(len(y)), priors)
    new_dev, new_std = honest_score(pool2, y, groups, priors, n_splits=args.splits)
    print(f"加入当前嵌入后的开发融合：全量 {new_s:.4f} / 留一折 {new_dev:.4f} ± {new_std:.4f}")
    print(f"\n>>> 当前嵌入的开发对照差值 = {new_dev - base_dev:+.4f}（不是独立泛化估计）")
    print(f"    权重：{ {k: round(v, 3) for k, v in new_w.items() if v} }")

    blend = sum(w * pool2[v] for v, w in new_w.items() if w)
    if new_alpha:
        blend = blend / np.power(priors, new_alpha)[None, :]
    cvmod.summarize(y, blend.argmax(axis=1), LABELS, LAB_OF, tag="融合（含嵌入）")

    np.save(out / "blend_v3_oof.npy", blend)
    np.save(out / "oof_emb.npy", oof_emb)
    pd.DataFrame(
        [{"view": k, "weight": new_w.get(k, 0.0)} for k in pool2]
    ).to_csv(out / "blend_v3_weights.csv", index=False)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
