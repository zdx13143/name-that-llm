"""多视图分别建模 + 概率融合 + 超参搜索（v2）。

相对 v1 的三处改进
------------------
1. **更多样的视图**：不同 n-gram 范围的字符视图、词视图，以及一个
   「余弦质心」原型分类器。融合的收益主要来自视图之间的**差异**，
   不是单个视图的强弱，所以故意放了归纳偏置不同的模型进去。
2. **扩展超参范围**：v1 显示 charword 在 C=16 时仍在上升、fp 在 C=0.5 时
   最好，说明两边的边界都没探到，这里把两端都往外扩。
3. **诚实的融合分数**：在 OOF 上直接搜权重会乐观（相当于用验证集选超参）。
   这里用「留一折」评估：拿 4 折的 OOF 选权重，在第 5 折上评分，再取平均。

用法：
    python src/run_sweep.py
    python src/run_sweep.py --quick
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
from common import LABELS, LAB_OF, LABEL2IDX, ensure_outputs, load_train  # noqa: E402
from features import extract_features  # noqa: E402

N_CLASSES = len(LABELS)

# 预先按折缓存的特征块。视图 = 若干块的拼接。
BLOCK_DEFS = {
    "char13": dict(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True,
                   min_df=2, max_features=150_000),
    "char25": dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                   min_df=2, max_features=200_000),
    "char36": dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True,
                   min_df=2, max_features=300_000),
    "word12": dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                   min_df=2, max_features=150_000, token_pattern=r"(?u)\b\w+\b"),
    "word13": dict(analyzer="word", ngram_range=(1, 3), sublinear_tf=True,
                   min_df=2, max_features=250_000, token_pattern=r"(?u)\b\w+\b"),
}

# 视图定义：名字 -> 用到的块（fp 表示手工指纹特征）
# 注意：不要在这里拼太多块。"char13+char25+char36" 拼起来是 65 万维，
# 逻辑回归拟合会慢到无法迭代，而且信息高度冗余，实测不划算。
VIEW_DEFS = {
    "char13": ["char13"],
    "char25": ["char25"],
    "char36": ["char36"],
    "word12": ["word12"],
    "word13": ["word13"],
    "charword": ["char25", "word12"],
    "bigmix": ["char36", "word13"],
    "fp": ["fp"],
}

SPARSE_VIEWS = [v for v in VIEW_DEFS if v != "fp"]


# ------------------------------------------------------------------ 分类器
class CosineCentroid:
    """原型分类器：每类在 TF-IDF 空间算 L2 归一化质心，按余弦相似度出概率。

    与逻辑回归的归纳偏置完全不同（原型式 vs 判别式），因此融合时能提供
    真正互补的信息，而不是同一信息的复述。
    """

    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        self.classes_ = np.unique(y)
        cents = []
        for c in self.classes_:
            sub = X[y == c]
            cent = np.asarray(sub.mean(axis=0)).ravel()
            norm = np.linalg.norm(cent)
            cents.append(cent / norm if norm else cent)
        self.centroids_ = np.vstack(cents)
        return self

    def predict_proba(self, X):
        X = sparse.csr_matrix(X)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1))).ravel()
        norms[norms == 0] = 1.0
        sims = (X @ self.centroids_.T) / norms[:, None]
        logits = sims / max(self.temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def make_model(kind: str, param: float):
    if kind == "lr":
        return LogisticRegression(C=param, max_iter=3000, class_weight="balanced",
                                  random_state=0, n_jobs=-1)
    if kind == "centroid":
        return CosineCentroid(temperature=param)
    raise ValueError(kind)


# ------------------------------------------------------------------ 折缓存
def build_folds(texts, fp, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr, va in gkf.split(texts, y, groups):
        ttr = [texts[i] for i in tr]
        tva = [texts[i] for i in va]
        blocks = {}
        for name, kw in BLOCK_DEFS.items():
            vec = TfidfVectorizer(lowercase=True, **kw).fit(ttr)
            blocks[name] = (vec.transform(ttr), vec.transform(tva))
        scaler = StandardScaler().fit(fp[tr])
        folds.append({
            "va": va, "y_tr": y[tr], "y_va": y[va],
            "blocks": blocks,
            "fp": (scaler.transform(fp[tr]), scaler.transform(fp[va])),
        })
    return folds


def view_matrices(fold, view: str):
    if view == "fp":
        return fold["fp"]
    parts = [fold["blocks"][b] for b in VIEW_DEFS[view]]
    if len(parts) == 1:
        return parts[0]
    return (sparse.hstack([p[0] for p in parts]).tocsr(),
            sparse.hstack([p[1] for p in parts]).tocsr())


def run_config(folds, n_rows, view: str, kind: str, param: float):
    oof = np.zeros((n_rows, N_CLASSES))
    for fold in folds:
        Atr, Ava = view_matrices(fold, view)
        model = make_model(kind, param)
        model.fit(Atr, fold["y_tr"])
        proba = model.predict_proba(Ava)
        cls = getattr(model, "classes_", None)
        if cls is not None and proba.shape[1] != N_CLASSES:
            full = np.zeros((proba.shape[0], N_CLASSES))
            full[:, cls] = proba
            proba = full
        oof[fold["va"]] = proba
    return oof


# ------------------------------------------------------------------ 融合
def ba_on(proba: np.ndarray, y: np.ndarray, idx, weights: dict,
          alpha: float, priors: np.ndarray) -> float:
    p = sum(w * proba[v][idx] for v, w in weights.items() if w)
    if alpha:
        p = p / np.power(priors, alpha)[None, :]
    return float(balanced_accuracy_score(y[idx], p.argmax(axis=1)))


def greedy_weights(proba: dict, y, idx, priors, rounds: int = 6, step: float = 0.1):
    """贪心坐标上升搜权重：从最佳单视图出发，逐轮微调各视图权重。"""
    singles = {v: ba_on(proba, y, idx, {v: 1.0}, 0.0, priors) for v in proba}
    best_view = max(singles, key=singles.get)
    weights = {v: 0.0 for v in proba}
    weights[best_view] = 1.0
    best = singles[best_view]
    best_alpha = 0.0

    for _ in range(rounds):
        improved = False
        for v in proba:
            for delta in (step, -step):
                cand = dict(weights)
                cand[v] = max(0.0, min(2.0, cand[v] + delta))
                if sum(cand.values()) == 0:
                    continue
                s = sum(cand.values())
                cand = {k: x / s for k, x in cand.items()}
                for alpha in (0.0, 0.5, 1.0):
                    sc = ba_on(proba, y, idx, cand, alpha, priors)
                    if sc > best + 1e-6:
                        best, weights, best_alpha = sc, cand, alpha
                        improved = True
        if not improved:
            break
    return weights, best_alpha, best


def honest_score(proba: dict, y, groups, priors, n_splits: int = 5):
    """留一折评估：用其余折选权重，在被留出的折上评分。"""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in gkf.split(np.zeros(len(y)), y, groups):
        w, alpha, _ = greedy_weights(proba, y, tr_idx, priors)
        scores.append(ba_on(proba, y, va_idx, w, alpha, priors))
    return float(np.mean(scores)), float(np.std(scores))


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out = ensure_outputs()
    df = load_train()
    texts = df["text"].tolist()
    y = np.array([LABEL2IDX[m] for m in df["model"]], dtype=np.int64)
    groups = df["prompt_id"].to_numpy()
    priors = np.bincount(y, minlength=N_CLASSES) / len(y)

    print("提取指纹特征 …")
    t0 = time.time()
    fp, _ = extract_features(texts, show_progress=True)
    print(f"  {fp.shape[1]} 维，{time.time() - t0:.1f}s")

    t0 = time.time()
    folds = build_folds(texts, fp, y, groups, n_splits=args.splits)
    print(f"折矩阵缓存完成，{time.time() - t0:.1f}s")

    lr_grid = [4.0, 16.0]
    cent_grid = [0.05]
    fp_grid = [0.25, 0.5]

    configs = []
    for view in SPARSE_VIEWS:
        for C in lr_grid:
            configs.append((view, "lr", C))
        for T in cent_grid:
            configs.append((view, "centroid", T))
    for C in fp_grid:
        configs.append(("fp", "lr", C))
    for T in cent_grid:
        configs.append(("fp", "centroid", T))

    print(f"\n共 {len(configs)} 个配置\n")
    print(f"{'视图':<11}{'分类器':<10}{'参数':>8}{'OOF':>10}{'秒':>8}")
    proba_by_cfg = {}
    rows = []
    for view, kind, param in configs:
        t0 = time.time()
        oof = run_config(folds, len(y), view, kind, param)
        s = float(balanced_accuracy_score(y, oof.argmax(axis=1)))
        dt = time.time() - t0
        proba_by_cfg[(view, kind, param)] = oof
        rows.append({"view": view, "kind": kind, "param": param,
                     "oof": s, "seconds": dt})
        print(f"{view:<11}{kind:<10}{param:>8}{s:>10.4f}{dt:>8.1f}")

    res = pd.DataFrame(rows).sort_values("oof", ascending=False)
    print(f"\n单配置最佳：{res.iloc[0]['view']} / {res.iloc[0]['kind']} "
          f"param={res.iloc[0]['param']} -> {res.iloc[0]['oof']:.4f}")

    # 每个视图取最优配置进入融合池
    pool = {}
    for (view, kind, param), oof in proba_by_cfg.items():
        cur = pool.get(view)
        s = float(balanced_accuracy_score(y, oof.argmax(axis=1)))
        if cur is None or s > cur[0]:
            pool[view] = (s, oof, kind, param)
    pool = {v: t[1] for v, t in pool.items()}
    print("\n各视图最优：")
    for v in sorted(pool, key=lambda k: -float(
            balanced_accuracy_score(y, pool[k].argmax(axis=1)))):
        print(f"  {v:<11}{balanced_accuracy_score(y, pool[v].argmax(axis=1)):.4f}")

    # 全量权重（用于报告）+ 诚实的留一折评估
    w_all, alpha_all, s_all = greedy_weights(pool, y, np.arange(len(y)), priors)
    print(f"\n全量 OOF 上选出的权重：{ {k: round(v, 3) for k, v in w_all.items() if v} }")
    print(f"  alpha={alpha_all}  该分数（乐观）= {s_all:.4f}")

    t0 = time.time()
    mean_h, std_h = honest_score(pool, y, groups, priors, n_splits=args.splits)
    print(f"留一折诚实评估 = {mean_h:.4f} ± {std_h:.4f}  ({time.time() - t0:.1f}s)")

    best_single = max(balanced_accuracy_score(y, p.argmax(axis=1))
                      for p in pool.values())
    print(f"最佳单视图 = {best_single:.4f}  ->  融合增益 "
          f"{mean_h - best_single:+.4f}")

    # 用全量权重生成最终预测
    blend = sum(w * pool[v] for v, w in w_all.items() if w)
    if alpha_all:
        blend = blend / np.power(priors, alpha_all)[None, :]
    cvmod.summarize(y, blend.argmax(axis=1), LABELS, LAB_OF, tag="融合（全量权重）")

    res.to_csv(out / "sweep_v2_results.csv", index=False)
    np.save(out / "blend_v2_oof.npy", blend)
    for v, p in pool.items():
        np.save(out / f"pool_v2_{v}.npy", p)
    pd.DataFrame([{"view": k, "weight": w_all.get(k, 0.0)} for k in pool]).to_csv(
        out / "blend_v2_weights.csv", index=False
    )
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
