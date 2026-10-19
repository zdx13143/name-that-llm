"""验证「Gemini 家族门控」在本地 CV 上是否真有收益。

背景
----
公开集成里做了 Claude 门控（5 类）和 GPT 门控（5 类），但**没有 Gemini 门控**。
而 Gemini 的三个 flash 模型（3.6/3.7/3.8）是我们错误最多的簇。

但有个硬约束：**gemini-3.7-flash 在所有公开生成数据里一条都没有**
（3.1-pro 160 / 3.6-flash 424 / 3.8-flash 430 / 3.7-flash 0）。
所以四类门控里有一类只能靠官方训练集的 150 条。

这个脚本就是回答：这样的门控到底值不值。
做法与原作者验证 Claude 门控的方式一致——按 prompt 分组 CV，
在 18 路基础概率之上叠加族内 SVC 的 log 证据，比较族内召回与总体 balanced accuracy。

用法：
    python src/gemini_gate_cv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL2IDX, LABELS, ROOT  # noqa: E402

K = len(LABELS)


def stylegram(texts_tr, texts_te):
    """近似原作的 stylegram：char_wb 2-5 + word 1-2，加权拼接。"""
    blocks_tr, blocks_te = [], []
    for analyzer, ng, w in (("char_wb", (2, 5), 1.0), ("word", (1, 2), 0.5)):
        opts = dict(analyzer=analyzer, ngram_range=ng, lowercase=True, sublinear_tf=True,
                    min_df=2, max_features=300_000, dtype=np.float64)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**opts).fit(texts_tr)
        blocks_tr.append(v.transform(texts_tr) * w)
        blocks_te.append(v.transform(texts_te) * w)
    return sparse.hstack(blocks_tr).tocsr(), sparse.hstack(blocks_te).tocsr()


def load_external_gemini() -> pd.DataFrame:
    parts = []
    for f in ROOT.glob("data/external/*/*.csv"):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        col = "model" if "model" in d.columns else ("label" if "label" in d.columns else None)
        if not col or "text" not in d.columns:
            continue
        m = d[col].astype(str).str.startswith("gemini-")
        if m.any():
            parts.append(d.loc[m, [col, "text"]].rename(columns={col: "model"}))
    if not parts:
        return pd.DataFrame(columns=["model", "text"])
    out = pd.concat(parts)
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 0].drop_duplicates(subset=["model", "text"])
    return out.reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    y = np.array([LABEL2IDX[m] for m in df["model"]])
    groups = df["prompt_id"].to_numpy()
    texts = df["text"].astype(str).tolist()

    GEM_IDX = np.array([i for i, l in enumerate(LABELS) if l.startswith("gemini")])
    GEM_NAMES = [LABELS[i] for i in GEM_IDX]
    print("Gemini 家族:", GEM_NAMES)

    ext = load_external_gemini()
    print(f"外部 Gemini 样本 {len(ext)}：{ext['model'].value_counts().to_dict()}")
    missing = [n for n in GEM_NAMES if n not in set(ext["model"])]
    if missing:
        print(f"⚠️ 无外部数据的类: {missing}（只能靠官方训练集的样本）")

    base = np.load(ROOT / "outputs" / "blend_v3_oof.npy")  # 我们的 18 路基础概率
    pred_base = base.argmax(1)

    def gem_recall(pred):
        m = np.isin(y, GEM_IDX)
        return (pred[m] == y[m]).mean()

    print(f"\n基础: 总体 {balanced_accuracy_score(y, pred_base):.4f}  "
          f"Gemini 族内召回 {gem_recall(pred_base):.4f}  "
          f"（Gemini 行数 {np.isin(y, GEM_IDX).sum()}）")

    folds = list(GroupKFold(5).split(df, y, groups))
    ext_all = ext["text"].tolist()
    ext_y = np.array([LABEL2IDX[m] for m in ext["model"]]) if len(ext) else np.array([], int)

    for lam in (0.5, 1.0, 1.5):
        pred = pred_base.copy()
        for tr, va in folds:
            # 训练门控：训练折的 Gemini 行 + 全部外部 Gemini
            m_tr = np.isin(y[tr], GEM_IDX)
            tr_texts = [texts[i] for i in tr[m_tr]] + ext_all
            tr_y = np.concatenate([y[tr][m_tr], ext_y]) if len(ext_all) else y[tr][m_tr]
            if len(np.unique(tr_y)) < 2:
                continue
            va_gem = va[np.isin(pred[va], GEM_IDX)]     # 只重判「被预测为 Gemini」的行
            if len(va_gem) == 0:
                continue
            A, B = stylegram(tr_texts, [texts[i] for i in va_gem])
            svc = LinearSVC(C=0.5, random_state=42, max_iter=4000,
                            class_weight="balanced").fit(A, tr_y)
            d = svc.decision_function(B)
            # 族内重排：log P[gem] + lam * decision
            s = np.log(np.clip(base[np.ix_(va_gem, GEM_IDX)], 1e-12, 1)) + lam * d
            pred[va_gem] = GEM_IDX[s.argmax(1)]
            if va_gem[0] == va[0]:
                pass
        changed = int((pred != pred_base).sum())
        print(f"\nlam={lam}: 总体 {balanced_accuracy_score(y, pred):.4f}  "
              f"Gemini 族内召回 {gem_recall(pred):.4f}  改动 {changed} 行"
              f"  (对照 {balanced_accuracy_score(y, pred_base):.4f} / {gem_recall(pred_base):.4f})")


if __name__ == "__main__":
    main()
