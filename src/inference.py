"""端到端推理：读 test.csv，产出 submission.parquet。

流程
----
1. 读 train.csv（有标签，用于拟合）与 test.csv（待预测）
2. 只用 **train** 拟合所有视图（TF-IDF 词表、标准化、分类器）
3. 冻结编码器对 train+test 一起算嵌入（纯前向，不涉及训练，因此无标签泄漏）
4. 按 configs/blend_v1.json 里的权重融合各视图概率
5. 写 submission.parquet

Kaggle 兼容性
-------------
比赛是 Code Competition：断网、只能从 /kaggle/input 读数据。
所以脚本会自动探测 Kaggle 路径，模型从挂载的 Dataset 里读。

用法：
    python src/inference.py                                   # 本地（用 data/raw）
    python src/inference.py --no-embed                        # 快速跑一遍纯 TF-IDF，验证管道
    python src/inference.py --out submission.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LABELS, LABEL2IDX, ROOT  # noqa: E402
from features import extract_features  # noqa: E402
from run_sweep import BLOCK_DEFS  # noqa: E402
import assign as assign_mod  # noqa: E402

N_CLASSES = len(LABELS)

# Kaggle 环境下的固定路径
KAGGLE_COMP_DIR = Path("/kaggle/input/competitions/name-that-llm")


def resolve_dirs() -> tuple[Path, Path]:
    """返回 (数据目录, 模型根目录)。

    模型根目录下直接是各个模型文件夹（ModernBERT-base / ModernBERT-large），
    这样本地（name-that-llm/models）与 Kaggle 数据集挂载点结构一致，
    配置里只写相对名字即可。
    """
    if KAGGLE_COMP_DIR.exists():
        data_dir = KAGGLE_COMP_DIR
        for cand in ("name-that-llm-models", "ntl-models", "models"):
            p = Path("/kaggle/input") / cand
            if p.exists():
                return data_dir, p
        return data_dir, Path("/kaggle/input")
    return ROOT / "data" / "raw", ROOT / "models"


def resolve_model_dir(name: str, model_root: Path) -> Path | None:
    """找到某个嵌入模型的目录（含 config.json 的那一层）。

    三种来源，按优先级：
    1. 本地/自建数据集：<model_root>/<name>
    2. Kaggle 官方模型挂载：/kaggle/input/*/pytorch/{base,large}
    3. 上面两种再往下搜一层（Kaggle Models 有时会多套一级目录）
    """
    def locate(base: Path) -> Path | None:
        if not base.exists():
            return None
        if (base / "config.json").exists():
            return base
        for cfg in sorted(base.rglob("config.json")):
            return cfg.parent
        return None

    hit = locate(model_root / name)
    if hit:
        return hit

    inp = Path("/kaggle/input")
    if not inp.exists():
        return None

    variation = "large" if "large" in name.lower() else "base"

    # 先试浅层固定模式
    for pat in (f"*/transformers/{variation}", f"*/pytorch/{variation}",
                f"*/models/{variation}", f"*/*/{variation}", f"*/{name}"):
        for cand in sorted(inp.glob(pat)):
            hit = locate(cand)
            if hit:
                return hit

    # 再递归找：Kaggle 把挂载的 Model 放在 /kaggle/input/models/<owner>/... 这种深层路径
    for cand in sorted(inp.rglob(variation)):
        if not cand.is_dir() or cand.parent.name not in {"transformers", "pytorch"}:
            continue
        hit = locate(cand)
        if hit:
            return hit
    return None


def find_config(explicit: str | None) -> Path:
    """按优先级找配置：显式参数 > src/configs > 仓库 configs > 当前目录。"""
    here = Path(__file__).resolve().parent
    if explicit:
        return Path(explicit)
    for cand in (here / "configs" / "blend_v1.json",
                 here.parent / "configs" / "blend_v1.json",
                 here / "blend_v1.json",
                 Path.cwd() / "blend_v1.json"):
        if cand.exists():
            return cand
    raise SystemExit("找不到 blend_v1.json")


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def build_embedding_view(texts, model_path: Path, max_length: int, C: float,
                         y: np.ndarray, n_train: int, verbose: bool = True):
    """对 train+test 一起编码，用 train 拟合 scaler+LR，返回 (test 概率, train 概率)。"""
    from embeddings import extract_embeddings

    t0 = time.time()
    X = extract_embeddings(
        texts, str(model_path), max_length=max_length, batch_size=16, verbose=verbose
    )
    Etr, Ete = X[:n_train], X[n_train:]

    scaler = StandardScaler().fit(Etr)
    clf = LogisticRegression(
        C=C, max_iter=5000, class_weight="balanced", random_state=0, n_jobs=-1
    )
    clf.fit(scaler.transform(Etr), y)

    p_tr = np.zeros((n_train, N_CLASSES))
    p_te = np.zeros((len(Ete), N_CLASSES))
    p_tr[:, clf.classes_] = clf.predict_proba(scaler.transform(Etr))
    p_te[:, clf.classes_] = clf.predict_proba(scaler.transform(Ete))
    if verbose:
        print(f"    {model_path.name} 完成，{time.time() - t0:.0f}s，"
              f"维度 {X.shape[1]}")
    return p_te, p_tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="submission.parquet")
    ap.add_argument("--no-embed", action="store_true", help="只用 TF-IDF 视图，快速验证")
    ap.add_argument("--no-assign", action="store_true",
                    help="关闭聚类+匈牙利的结构约束解码，回退到独立 argmax")
    ap.add_argument("--n-groups", type=int, default=0,
                    help="测试集 prompt 组数，0 表示自动按 ceil(n/18) 估计")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    cfg = json.loads(find_config(args.config).read_text(encoding="utf-8"))
    data_dir, model_root = resolve_dirs()
    print(f"数据目录 {data_dir}")
    print(f"模型根目录 {model_root}")

    train, test = load_data(data_dir)
    print(f"train {len(train)} 行 / test {len(test)} 行")

    train_texts = train["text"].astype(str).tolist()
    test_texts = test["text"].astype(str).tolist()
    y = np.array([LABEL2IDX[m] for m in train["model"]], dtype=np.int64)
    n_train, n_test = len(train_texts), len(test_texts)
    all_texts = train_texts + test_texts

    # ---------------- 手工指纹特征 ----------------
    t0 = time.time()
    fp_all, _ = extract_features(all_texts)
    scaler_fp = StandardScaler().fit(fp_all[:n_train])
    fp_tr = scaler_fp.transform(fp_all[:n_train])
    fp_te = scaler_fp.transform(fp_all[n_train:])
    if verbose:
        print(f"指纹特征 {fp_all.shape[1]} 维，{time.time() - t0:.0f}s")

    # ---------------- TF-IDF 特征块（只在 train 上拟合词表） ----------------
    t0 = time.time()
    blocks = {}
    for name, kw in BLOCK_DEFS.items():
        vec = TfidfVectorizer(lowercase=True, **kw).fit(train_texts)
        blocks[name] = (vec.transform(train_texts), vec.transform(test_texts))
    if verbose:
        print(f"TF-IDF 词表拟合完成 {time.time() - t0:.0f}s")

    def view_mats(view: str):
        if view == "fp":
            return fp_tr, fp_te
        from run_sweep import VIEW_DEFS
        parts = [blocks[b] for b in VIEW_DEFS[view]]
        if len(parts) == 1:
            return parts[0]
        return (sparse.hstack([p[0] for p in parts]).tocsr(),
                sparse.hstack([p[1] for p in parts]).tocsr())

    # ---------------- 逐视图训练 + 融合 ----------------
    blend_te = np.zeros((n_test, N_CLASSES))
    blend_tr = np.zeros((n_train, N_CLASSES))
    total_w = 0.0

    print("\nTF-IDF 视图：")
    for spec in cfg["tfidf_views"]:
        view, w = spec["view"], float(spec["weight"])
        Atr, Ate = view_mats(view)
        clf = LogisticRegression(
            C=float(spec["param"]), max_iter=3000, class_weight="balanced",
            random_state=0, n_jobs=-1,
        )
        clf.fit(Atr, y)
        p_tr = np.zeros((n_train, N_CLASSES))
        p_te = np.zeros((n_test, N_CLASSES))
        p_tr[:, clf.classes_] = clf.predict_proba(Atr)
        p_te[:, clf.classes_] = clf.predict_proba(Ate)
        blend_te += w * p_te
        blend_tr += w * p_tr
        total_w += w
        if verbose:
            acc = (p_tr.argmax(1) == y).mean()
            print(f"  {view:<10} w={w:.3f}  C={spec['param']:<5}  train acc={acc:.3f}")

    if not args.no_embed:
        print("\n嵌入视图：")
        for spec in cfg["embedding_views"]:
            path = resolve_model_dir(spec["path"], model_root)
            if path is None:
                print(f"  跳过 {spec['path']} —— 找不到模型目录")
                inp = Path("/kaggle/input")
                if inp.exists():
                    print(f"    /kaggle/input 下现有："
                          f"{[p.name for p in sorted(inp.iterdir())][:12]}")
                    for p in sorted(inp.rglob("config.json")):
                        print(f"    发现 config.json: {p.parent}")
                continue
            w = float(spec["weight"])
            p_te, p_tr = build_embedding_view(
                all_texts, path, int(spec["max_length"]), float(spec["param"]),
                y, n_train, verbose=verbose,
            )
            blend_te += w * p_te
            blend_tr += w * p_tr
            total_w += w

    if total_w == 0:
        raise SystemExit("没有任何视图参与融合")

    blend_te /= total_w
    blend_tr /= total_w

    # 训练集上的自评（仅供参考，不是泛化指标）
    print(f"\n融合后训练集拟合率 {np.mean(blend_tr.argmax(1) == y):.4f}"
          f"（这是拟合率不是泛化分，别当成绩看）")

    # ---------------- 结构约束解码 ----------------
    # 测试集约 3,888 行 ≈ 216 个 prompt × 18 个模型各一条，
    # 所以组内是一对一分配，而不是独立分类。实测值 +0.09。
    if args.no_assign:
        print("\n跳过结构约束解码（--no-assign）")
        pred = blend_te.argmax(axis=1)
    else:
        print("\n结构约束解码（TF-IDF 聚类 + 带容量匈牙利）：")
        pred, _ = assign_mod.decode(blend_te, test_texts,
                                    n_groups=args.n_groups or None, verbose=True)

    labels = [LABELS[i] for i in pred]

    submission = pd.DataFrame({"id": test["id"], "model": labels})
    out_path = Path(args.out)
    submission.to_parquet(out_path, index=False)

    # ---------------- 提交前自检 ----------------
    assert len(submission) == len(test), "行数与 test 不一致"
    assert submission["id"].is_unique, "id 有重复"
    assert submission["model"].notna().all(), "存在空值"
    assert set(submission["model"]).issubset(set(LABELS)), "存在非法标签"

    print(f"\n已写出 {out_path}  ({len(submission)} 行)")
    print("预测分布（前 6）：")
    print(submission["model"].value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
