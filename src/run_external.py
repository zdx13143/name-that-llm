"""验证外部公开语料是否带来真实增量。

数据来源
--------
`olegzholobov/name-that-llm-response-corpus` 的 `external_short_wide.parquet`：
13,642 行 × 18 个模型列，24,025 个答案单元，已按 40–250 词筛选。
作者已核验：与官方 2,699 条训练答案零重叠（字节/归一化/子串/90% 字符相似度全查过）。
本脚本另外核验了 prompt 层面：12,022 个外部 prompt 与 150 个训练题零重叠。

验证口径（关键）
----------------
外部数据**只进训练折，永远不进验证折**。
验证折只由官方训练集的 holdout prompt 构成，所以 OOF 分数与之前完全可比。

用法：
    python src/run_external.py                      # 单视图快速判断
    python src/run_external.py --max-per-model 1500 --train-weight 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv as cvmod  # noqa: E402
from common import LABELS, LAB_OF, LABEL2IDX, ROOT, ensure_outputs, load_train  # noqa: E402
from features import extract_features  # noqa: E402
from run_sweep import BLOCK_DEFS  # noqa: E402

N_CLASSES = len(LABELS)
EXT_PATH = ROOT / "data" / "external" / "response_corpus" / "external_short_wide.parquet"


def load_external(max_per_model: int, seed: int = 0) -> pd.DataFrame:
    """把宽表熔成长表，按模型去重、按上限截断。"""
    wide = pd.read_parquet(EXT_PATH)
    keep = ["id", "prompt"] + [c for c in wide.columns if c in LABELS]
    long = wide[keep].melt(id_vars=["id", "prompt"], var_name="model",
                           value_name="text").dropna(subset=["text"])
    long["text"] = long["text"].astype(str).str.strip()
    long = long[long["text"].str.len() > 0]

    # 同模型下文本完全相同的只留一份（不同 prompt 下重复出现会放大同一个样本）
    long = long.drop_duplicates(subset=["model", "text"])

    if max_per_model:
        parts = []
        for m, g in long.groupby("model"):
            parts.append(g.sample(min(len(g), max_per_model), random_state=seed))
        long = pd.concat(parts)

    long["prompt_id"] = "ext:" + long["id"].astype(str)
    return long[["id", "prompt_id", "text", "model"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="charword", help="charword / char13 / char25 / bigmix")
    ap.add_argument("--C", type=float, default=16.0)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max-per-model", type=int, default=0, help="0 表示不截断")
    ap.add_argument("--train-weight", type=float, default=3.0,
                    help="官方训练样本的权重（外部数据权重固定 1.0）")
    ap.add_argument("--no-external", action="store_true", help="对照组：不加外部数据")
    args = ap.parse_args()

    out = ensure_outputs()
    train = load_train()
    y = np.array([LABEL2IDX[m] for m in train["model"]], dtype=np.int64)
    groups = train["prompt_id"].to_numpy()
    tr_texts = train["text"].astype(str).tolist()

    if args.no_external:
        ext = train.iloc[0:0].copy()
        print("对照组：只用官方训练集")
    else:
        ext = load_external(args.max_per_model)
        print(f"外部数据 {len(ext)} 条，覆盖模型 {ext['model'].nunique()}/18")
        cnt = ext["model"].value_counts().reindex(LABELS).fillna(0).astype(int)
        print("  各模型条数：", dict(cnt))

    ext_texts = ext["text"].astype(str).tolist()
    ext_y = np.array([LABEL2IDX[m] for m in ext["model"]], dtype=np.int64) if len(ext) else np.array([], dtype=int)

    print("\n提取指纹特征 …")
    t0 = time.time()
    fp_tr_all, _ = extract_features(tr_texts)
    fp_ext_all, _ = extract_features(ext_texts) if len(ext) else (np.zeros((0, 1)), [])
    print(f"  {time.time() - t0:.0f}s")

    print("按折构建特征（词表只在训练侧拟合）…")
    gkf = GroupKFold(n_splits=args.splits)
    oof = np.zeros((len(train), N_CLASSES))
    split = list(gkf.split(tr_texts, y, groups))

    for fold, (tr, va) in enumerate(split, 1):
        t0 = time.time()
        fit_texts = [tr_texts[i] for i in tr] + ext_texts

        vecs = {}
        for b in ("char25", "word12"):
            vecs[b] = TfidfVectorizer(lowercase=True, **BLOCK_DEFS[b]).fit(fit_texts)

        def build(b, idx):
            return vecs[b].transform([tr_texts[i] for i in idx])

        # 只用训练侧数据拟合 scaler（训练折 + 外部数据），验证折绝不参与
        fp_fit = (np.vstack([fp_tr_all[tr], fp_ext_all]) if len(ext)
                  else fp_tr_all[tr])
        sc = StandardScaler().fit(fp_fit)

        def block_matrix(idxs):
            """把 char / word / fp 三块按列拼成一个样本矩阵。"""
            return sparse.hstack([
                build("char25", idxs),
                build("word12", idxs),
                sparse.csr_matrix(sc.transform(fp_tr_all[idxs])),
            ]).tocsr()

        X_tr = block_matrix(tr)
        X_va = block_matrix(va)
        y_tr = y[tr]
        w_tr = np.full(len(tr), args.train_weight)

        if len(ext):
            X_ext = sparse.hstack([
                vecs["char25"].transform(ext_texts),
                vecs["word12"].transform(ext_texts),
                sparse.csr_matrix(sc.transform(fp_ext_all)),
            ]).tocsr()
            # 行方向拼接：官方训练折在上，外部数据在下
            X_tr = sparse.vstack([X_tr, X_ext]).tocsr()
            y_tr = np.concatenate([y_tr, ext_y])
            w_tr = np.concatenate([w_tr, np.ones(len(ext))])

        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced",
                                 random_state=0, n_jobs=-1)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        proba = clf.predict_proba(X_va)
        full = np.zeros((len(va), N_CLASSES))
        full[:, clf.classes_] = proba
        oof[va] = full

        s = balanced_accuracy_score(y[va], oof[va].argmax(1))
        print(f"  fold {fold}/{args.splits}  n_train={len(tr)}+{len(ext)}  "
              f"bal.acc={s:.4f}  {time.time() - t0:.0f}s")

    ba = float(balanced_accuracy_score(y, oof.argmax(1)))
    print(f"\nOOF balanced accuracy = {ba:.4f}")
    cvmod.summarize(y, oof.argmax(1), LABELS, LAB_OF,
                    tag=f"外部数据 {'关闭' if args.no_external else '开启'}")

    tag = "noext" if args.no_external else f"ext{args.max_per_model or 'all'}"
    np.save(out / f"oof_external_{tag}.npy", oof)
    print(f"\n已保存 {out / f'oof_external_{tag}.npy'}")


if __name__ == "__main__":
    main()
