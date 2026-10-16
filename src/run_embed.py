"""评估预训练模型隐藏状态特征的增量，并与已有视图融合。

流程：
  1. 冻结编码器，对全部训练文本算一次嵌入（缓存复用，不存在标签泄漏）
  2. 用同一套 prompt 分组 CV 评估「嵌入 + 线性分类器」
  3. 把嵌入视图加进 v2 的视图池，重新做加权融合，
     用留一折评估判断它到底带来了多少真实增量

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

    # ---- 与 v2 视图池融合 ----
    pool_files = sorted(out.glob("pool_v2_*.npy"))
    if args.no_blend or not pool_files:
        print("\n未找到 v2 视图池，跳过融合（先跑 python src/run_sweep.py）")
        np.save(out / f"embemb_{args.model.replace('/', '__')}.npy", oof_emb)
        return

    pool = {}
    for f in pool_files:
        name = f.stem.replace("pool_v2_", "")
        arr = np.load(f)
        if arr.shape == oof_emb.shape:
            pool[name] = arr
    print(f"\n视图池：{list(pool)}")

    base_w, base_alpha, base_s = greedy_weights(pool, y, np.arange(len(y)), priors)
    base_honest, base_std = honest_score(pool, y, groups, priors, n_splits=args.splits)
    print(f"原有融合（不含嵌入）：全量 {base_s:.4f} / 留一折 {base_honest:.4f} ± {base_std:.4f}")

    pool2 = dict(pool)
    pool2["emb"] = oof_emb
    new_w, new_alpha, new_s = greedy_weights(pool2, y, np.arange(len(y)), priors)
    new_honest, new_std = honest_score(pool2, y, groups, priors, n_splits=args.splits)
    print(f"加入嵌入后融合：全量 {new_s:.4f} / 留一折 {new_honest:.4f} ± {new_std:.4f}")
    print(f"\n>>> 嵌入带来的真实增量 = {new_honest - base_honest:+.4f}")
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
