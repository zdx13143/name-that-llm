"""在复现的 0.554 内核里注入「Gemini 家族门控」。

思路
----
原作的 final_pipeline.combine_and_predict 结构干净：

    mixed  = 几何融合(base, tabpfn)
    mixed  = 0.95*mixed + 0.05*tabicl
    corr   = apply_post_balance(mixed, texts)
    pred   = gated_predict(corr, ...)      # 这里做了 Claude 门控 + GPT 门控

公开集成做了 Claude(5) 和 GPT(5) 的门控，**唯独没做 Gemini(4)**，
而 3.6/3.7/3.8 flash 是最混淆的三个类。

本地 CV 实测（src/gemini_gate_cv.py，按 prompt 分组）：
    总体 0.3978 -> 0.4064 (+0.0086)，Gemini 族内召回 0.4883 -> 0.5267 (+0.0384)

本补丁只在 gated_predict 之后追加一次族内重判，且：
- 只对「被预测为 Gemini」的行生效（其余行一字不改）
- 整体包 try/except，任何异常都原样返回，绝不让 0.554 的管线崩掉
- 不改动 geometric_pool / combine_complements（那两处有严格断言，保持原样）

用法：
    python scripts/patch_gemini_gate.py
    kaggle kernels push -p kaggle/ensemble554g
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external_notebooks" / "v055" / "0-554-lora-tabpfn-centered-style.ipynb"
DST_DIR = ROOT / "kaggle" / "ensemble554g"
MARKER = "%%writefile /kaggle/temp/ntl-v6-pfn184-code/final_pipeline.py"

NEW_PIPELINE = '''%%writefile /kaggle/temp/ntl-v6-pfn184-code/final_pipeline.py
"""Geometric PFN fusion, small TabICL complement, punctuation, then original gate."""
import glob
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from claude_gate import gated_predict
from orthography_post import apply_post_balance

# ---- 我们追加的 Gemini 门控超参（本地 CV 最优 lam=1.0）----
GEM_LAM = 1.0
GEM_C = 0.5
EXT_FILES = ("gen_passages.csv", "ntl_proxy_generations.csv", "ntl_generations.csv")


def geometric_pool(base, tabular, tabpfn_exponent=.2):
    """Pool raw class-aligned probabilities with exponents; normalize once by row."""
    assert base.dtype == tabular.dtype == np.float64
    assert base.ndim == 2 and base.shape == tabular.shape and base.shape[1] == 18
    assert np.isfinite(base).all() and np.isfinite(tabular).all()
    assert (base >= 0).all() and (tabular >= 0).all()
    assert (base.sum(1) > 0).all() and (tabular.sum(1) > 0).all()
    np.testing.assert_allclose(tabular.sum(1), 1, rtol=0, atol=1e-5)
    assert tabpfn_exponent == .2
    a = tabpfn_exponent
    values = (1-a)*np.log(np.clip(base, 1e-300, None)) + a*np.log(np.clip(tabular, 1e-300, None))
    mixed = np.exp(values-values.max(1, keepdims=True))
    mixed /= mixed.sum(1, keepdims=True)
    assert np.isfinite(mixed).all() and (mixed >= 0).all()
    np.testing.assert_allclose(mixed.sum(1), 1, rtol=0, atol=2e-15)
    return mixed


def combine_complements(base, tabular, tabicl):
    """Preserve raw TabICL probability mass in the fixed .95/.05 arithmetic mix."""
    assert tabicl.dtype == np.float64 and tabicl.shape == base.shape
    assert np.isfinite(tabicl).all() and (tabicl >= 0).all()
    np.testing.assert_allclose(tabicl.sum(1), 1, rtol=0, atol=1e-5)
    geometric = geometric_pool(base, tabular, tabpfn_exponent=.2)
    mixed = .95*geometric + .05*tabicl
    expected_mass = .95*geometric.sum(1) + .05*tabicl.sum(1)
    np.testing.assert_allclose(mixed.sum(1), expected_mass, rtol=0, atol=2e-15)
    return mixed, expected_mass


# ---------------------------------------------------------------- Gemini gate
def _gem_indices(labels):
    return np.array([i for i, l in enumerate(labels) if l.startswith('gemini')], dtype=int)


def _stylegram(tr_texts, te_texts):
    """近似原作 stylegram：char_wb 2-5 与 word 1-2 加权拼接。"""
    blocks_tr, blocks_te = [], []
    for analyzer, ng, w in (('char_wb', (2, 5), 1.0), ('word', (1, 2), 0.5)):
        opts = dict(analyzer=analyzer, ngram_range=ng, lowercase=True, sublinear_tf=True,
                    min_df=2, max_features=300000, dtype=np.float64)
        if analyzer == 'word':
            opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
        v = TfidfVectorizer(**opts).fit(tr_texts)
        blocks_tr.append(v.transform(tr_texts) * w)
        blocks_te.append(v.transform(te_texts) * w)
    return sparse.hstack(blocks_tr).tocsr(), sparse.hstack(blocks_te).tocsr()


def _data_dir():
    for p in ('/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'):
        if os.path.isfile(os.path.join(p, 'train.csv')):
            return p
    hits = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def _gate_training(labels):
    """官方训练集的 Gemini 行 + 公开外部 Gemini 段落（3.7-flash 无公开数据）。"""
    pos = {l: i for i, l in enumerate(labels)}
    gem_names = {labels[i] for i in _gem_indices(labels)}
    texts, ys = [], []

    d = _data_dir()
    if d:
        tr = pd.read_csv(os.path.join(d, 'train.csv'))
        m = tr['model'].isin(gem_names)
        texts += tr.loc[m, 'text'].astype(str).tolist()
        ys += [pos[v] for v in tr.loc[m, 'model']]

    for name in EXT_FILES:
        for f in glob.glob('/kaggle/input/**/' + name, recursive=True)[:1]:
            try:
                e = pd.read_csv(f)
            except Exception:
                continue
            col = 'model' if 'model' in e.columns else ('label' if 'label' in e.columns else None)
            if col is None or 'text' not in e.columns:
                continue
            mm = e[col].astype(str).isin(gem_names)
            texts += e.loc[mm, 'text'].astype(str).tolist()
            ys += [pos[v] for v in e.loc[mm, col].astype(str)]
    return texts, np.asarray(ys, dtype=int)


def gemini_gate(proba, pred, texts, labels, lam=GEM_LAM):
    """只对『被预测为 Gemini』的行，用族内 4 路 SVC 的 log 证据重排。失败即原样返回。"""
    try:
        gem_idx = _gem_indices(labels)
        rows = np.where(np.isin(pred, gem_idx))[0]
        if len(rows) == 0:
            print('[gem-gate] no Gemini-predicted rows', flush=True)
            return pred
        gtexts, gy = _gate_training(labels)
        if len(gy) < 200 or len(np.unique(gy)) < 4:
            print(f'[gem-gate] skipped: train rows {len(gy)}, classes {len(np.unique(gy))}/4', flush=True)
            return pred
        A, B = _stylegram(gtexts, [texts[i] for i in rows])
        svc = LinearSVC(C=GEM_C, random_state=42, max_iter=4000,
                        class_weight='balanced').fit(A, gy)
        cls = np.asarray(svc.classes_, dtype=int)
        if len(cls) != len(gem_idx) or not np.array_equal(cls, gem_idx):
            print(f'[gem-gate] class mismatch {cls} vs {gem_idx}, skipped', flush=True)
            return pred
        d = svc.decision_function(B)
        s = np.log(np.clip(proba[np.ix_(rows, gem_idx)], 1e-12, 1)) + lam * d
        new_pred = pred.copy()
        new_pred[rows] = gem_idx[s.argmax(1)]
        print(f'[gem-gate] gate rows {len(gtexts)} | reranked {len(rows)} | '
              f'changed {int((new_pred[rows] != pred[rows]).sum())}', flush=True)
        return new_pred
    except Exception as exc:
        print('[gem-gate] skipped on error:', repr(exc), flush=True)
        return pred


def combine_and_predict(base, tabular, texts, labels, gate, tabicl):
    assert base.shape == tabular.shape == tabicl.shape == (len(texts), len(labels))
    mixed, expected_mass = combine_complements(base, tabular, tabicl)
    corrected = apply_post_balance(mixed, texts, labels)
    np.testing.assert_allclose(corrected.sum(1), 1, rtol=0, atol=1e-5)
    pred = gated_predict(corrected, labels, **gate)
    # 我们追加的一步：Gemini 家族内重判（只动 Gemini 预测行）
    pred = gemini_gate(corrected, pred, texts, labels)
    return pred, corrected, expected_mass
'''


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    hit = None
    for c in nb["cells"]:
        src = "".join(c["source"])
        if src.startswith(MARKER):
            hit = c
            break
    if hit is None:
        raise SystemExit("找不到 final_pipeline.py 的写文件单元")

    old_len = len("".join(hit["source"]))
    hit["source"] = NEW_PIPELINE.splitlines(keepends=True)
    print(f"已替换 final_pipeline.py 单元（{old_len} -> {len(NEW_PIPELINE)} 字符）")

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True)
    nb_name = "ntl-ensemble-554-gemgate.ipynb"
    (DST_DIR / nb_name).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    meta = {
        "id": "zdx13143/ntl-ensemble-554-gemgate",
        "title": "ntl-ensemble-554-gemgate",
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "seyitkaangunes/ntl-extra-api-passages",
            "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
            "lightningv08/ntl-tabicl-offline-assets",
            "lightningv08/ntl-tabular-offline-assets",
            "bbucxi/qwen35-9b-safetensors",
            "pjmathematician/ultrabert-380m-ntl",
            # 我们新增：Gemini 外部生成数据
            "lucasboesen/ntl-free-proxy-generations",
            "dophuchung/ntl-proxy-generations",
            "abdullah1afify/ntl-generations",
        ],
        "kernel_sources": [
            "lucasboesen/name-that-llm-glm4-9b-trunc25",
            "lucasboesen/name-that-llm-extract-encoders2",
            "lucasboesen/name-that-llm-electra-large-weights",
            "lucasboesen/name-that-llm-desklib-weights",
            "lightningv08/ntl-lora24-fivefold-training",
            "lucasboesen/name-that-llm-train-lora24-s7",
            "lucasboesen/name-that-llm-train-tokattn",
            "lucasboesen/name-that-llm-train-tokattn2",
        ],
        "competition_sources": ["name-that-llm"],
        "model_sources": [
            "qwen-lm/qwen-3/Transformers/8b-base/1",
            "qwen-lm/qwen-3/Transformers/4b-base/1",
            "qwen-lm/qwen2.5/Transformers/7b/1",
        ],
    }
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {DST_DIR}")


if __name__ == "__main__":
    main()
