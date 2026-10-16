"""基线：三视图对比 + prompt 分组 CV。

对比三条线，判断「手工指纹特征」到底值多少分：
  1. fp    —— 只用 src/features.py 的手工指纹特征
  2. tfidf —— 只用字符 + 词 n-gram（复现公开 baseline 的思路）
  3. all   —— 指纹特征 + n-gram 拼接

所有视图共用同一套 GroupKFold(prompt_id) 切分，分数可直接比较。

用法：
    python src/run_baseline.py                 # 跑三条线
    python src/run_baseline.py --views all     # 只跑其中一条
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv as cvmod  # noqa: E402
from common import LABELS, LAB_OF, LABEL2IDX, ensure_outputs, load_train  # noqa: E402
from features import extract_features  # noqa: E402


class MultiViewClassifier(BaseEstimator, ClassifierMixin):
    """把「手工指纹特征」「字符 n-gram」「词 n-gram」拼在一起送进线性分类器。

    关键点：所有拟合步骤（标准化、IDF、词表）都发生在 fit() 内部，
    因此配合 GroupKFold 使用时，验证折的信息不会泄漏进训练折。
    """

    def __init__(
        self,
        use_fp: bool = True,
        use_char: bool = True,
        use_word: bool = True,
        C: float = 4.0,
        class_weight: str | None = "balanced",
        seed: int = 0,
    ):
        self.use_fp = use_fp
        self.use_char = use_char
        self.use_word = use_word
        self.C = C
        self.class_weight = class_weight
        self.seed = seed

    def _numeric(self, X: pd.DataFrame) -> np.ndarray:
        return X.drop(columns=["text"]).to_numpy(dtype=np.float64)

    def fit(self, X: pd.DataFrame, y):
        texts = X["text"].astype(str).tolist()
        blocks = []

        if self.use_fp:
            self.scaler_ = StandardScaler().fit(self._numeric(X))
            blocks.append(sparse.csr_matrix(self.scaler_.transform(self._numeric(X))))

        if self.use_char:
            self.char_ = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                min_df=2, max_features=200_000, lowercase=True,
            )
            blocks.append(self.char_.fit_transform(texts))

        if self.use_word:
            self.word_ = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                min_df=2, max_features=150_000, lowercase=True,
                token_pattern=r"(?u)\b\w+\b",
            )
            blocks.append(self.word_.fit_transform(texts))

        self.clf_ = LogisticRegression(
            C=self.C, max_iter=3000, class_weight=self.class_weight,
            random_state=self.seed, n_jobs=-1,
        )
        self.clf_.fit(sparse.hstack(blocks).tocsr(), y)
        self.classes_ = self.clf_.classes_
        return self

    def _transform(self, X: pd.DataFrame):
        texts = X["text"].astype(str).tolist()
        blocks = []
        if self.use_fp:
            blocks.append(sparse.csr_matrix(self.scaler_.transform(self._numeric(X))))
        if self.use_char:
            blocks.append(self.char_.transform(texts))
        if self.use_word:
            blocks.append(self.word_.transform(texts))
        return sparse.hstack(blocks).tocsr()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict(self._transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(self._transform(X))


def run_view(name: str, df: pd.DataFrame, n_splits: int, verbose: bool):
    y = np.array([LABEL2IDX[m] for m in df["model"]], dtype=np.int64)
    groups = df["prompt_id"].to_numpy()

    # 所有视图都保留 text 列：分类器内部需要它来拟合/应用 n-gram 词表，
    # 只用指纹特征的视图也会拿到 text，只是不用它。
    X = df[["text"]].reset_index(drop=True).copy()
    if name in {"fp", "all"}:
        Xf, names = extract_features(df["text"].tolist(), show_progress=verbose)
        X = pd.concat([X, pd.DataFrame(Xf, columns=names)], axis=1)

    def build():
        return MultiViewClassifier(
            use_fp=(name in {"fp", "all"}),
            use_char=(name in {"tfidf", "all"}),
            use_word=(name in {"tfidf", "all"}),
        )

    print(f"\n--- 视图 {name} ---")
    t0 = time.time()
    oof_pred, oof_proba, folds = cvmod.oof_predictions(
        build, X, y, groups, n_classes=len(LABELS), n_splits=n_splits, verbose=verbose
    )
    elapsed = time.time() - t0

    summary = cvmod.summarize(y, oof_pred, LABELS, LAB_OF, tag=f"[{name}]")
    return {
        "view": name,
        "mean_bal_acc": float(np.mean(folds)),
        "std": float(np.std(folds)),
        "oof_bal_acc": summary["balanced_accuracy"],
        "lab_acc": summary["lab_accuracy"],
        "seconds": elapsed,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "recall": summary["recall"],
        "confusions": summary["confusions"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--views", default="fp,tfidf,all")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    out = ensure_outputs()

    df = load_train()
    print(f"训练集 {len(df)} 行 / {df['prompt_id'].nunique()} 个 prompt / "
          f"{df['model'].nunique()} 个模型")
    print(f"每个 prompt 的文本数：{df.groupby('prompt_id').size().unique().tolist()}")

    results = []
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        res = run_view(view, df, args.splits, verbose)
        results.append(res)

    print(f"\n{'=' * 66}")
    print("三视图对比（同一套 prompt 分组 CV 切分）")
    print(f"{'=' * 66}")
    print(f"{'视图':<8}{'折均值':>10}{'折标准差':>10}{'OOF':>10}{'机构准确率':>12}{'耗时(s)':>10}")
    for r in results:
        print(f"{r['view']:<8}{r['mean_bal_acc']:>10.4f}{r['std']:>10.4f}"
              f"{r['oof_bal_acc']:>10.4f}{r['lab_acc']:>12.4f}{r['seconds']:>10.1f}")

    best = max(results, key=lambda r: r["oof_bal_acc"])
    print(f"\n最佳视图：{best['view']}  OOF balanced accuracy = {best['oof_bal_acc']:.4f}")

    # 保存结果，供后续分析/融合
    for r in results:
        np.save(out / f"oof_pred_{r['view']}.npy", r["oof_pred"])
        np.save(out / f"oof_proba_{r['view']}.npy", r["oof_proba"])
        r["recall"].to_csv(out / f"recall_{r['view']}.csv", index=False)
        r["confusions"].to_csv(out / f"confusions_{r['view']}.csv", index=False)

    pd.DataFrame(
        [{k: v for k, v in r.items()
          if k not in {"oof_pred", "oof_proba", "recall", "confusions"}}
         for r in results]
    ).to_csv(out / "baseline_summary.csv", index=False)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
