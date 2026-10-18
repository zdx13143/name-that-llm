"""在 Kaggle GPU 上微调编码器做 18 分类。

为什么值得做
------------
冻结嵌入 + 逻辑回归的 OOF 是 0.3882。线性探针吃不下 19k 样本的容量，
微调可以让模型自己学「哪些 token 组合是某个模型的写法」。

数据
----
官方 train 2,699 + 公开外部语料（挂载 olegzholobov/name-that-llm-response-corpus）。
外部数据**只进训练折**，验证折永远只用官方 holdout prompt，
所以 OOF 与既有 0.3882 完全可比。

工程要点（第一版踩过的坑）
--------------------------
1. **必须预分词。** 第一版在 Dataset 里逐条 tokenize，18k 样本每轮都要重做，
   一个 epoch 410s，其中一大截是分词开销。
2. **一轮学不会。** 第一版只跑 1 轮，fold-1 只有 0.1778（对照：融合模型 0.3815）。
   现在逐轮验证、保留最佳轮。
3. **外部数据要降权。** 它来自不同系统提示与解码参数，标签真但分布脏；
   样本里 80%+ 是外部数据，不降权会把官方分布冲掉。

产出：OOF 概率 + N 折模型对 test 的平均预测 → submission.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_CLASSES = 18


def find_inputs() -> tuple[Path, Path | None]:
    comp = Path("/kaggle/input/competitions/name-that-llm")
    if not comp.exists():
        comp = Path(__file__).resolve().parents[1] / "data" / "raw"
    ext = None
    for p in sorted(Path("/kaggle/input").rglob("external_short_wide.parquet")):
        ext = p
        break
    return comp, ext


def load_external(path: Path, labels: list[str], max_per_model: int) -> pd.DataFrame:
    wide = pd.read_parquet(path)
    cols = [c for c in wide.columns if c in labels]
    long = wide[["id", "prompt"] + cols].melt(
        id_vars=["id", "prompt"], var_name="model", value_name="text"
    ).dropna(subset=["text"])
    long["text"] = long["text"].astype(str).str.strip()
    long = long[long["text"].str.len() > 0].drop_duplicates(subset=["model", "text"])
    if max_per_model:
        parts = []
        for _, g in long.groupby("model"):
            parts.append(g.sample(min(len(g), max_per_model), random_state=0))
        long = pd.concat(parts)
    return long.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/kaggle/input/models/answer-ai/modernbert/transformers/base/2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-per-model", type=int, default=800)
    ap.add_argument("--ext-weight", type=float, default=0.4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=0)
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--out", default="submission.parquet")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import LABELS, LABEL2IDX  # noqa: E402

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}")
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    comp, ext_path = find_inputs()
    train = pd.read_csv(comp / "train.csv")
    test = pd.read_csv(comp / "test.csv")
    print(f"train {len(train)}  test {len(test)}")

    y = np.array([LABEL2IDX[m] for m in train["model"]], dtype=np.int64)
    groups = train["prompt_id"].to_numpy()
    texts = train["text"].astype(str).tolist()
    test_texts = test["text"].astype(str).tolist()

    ext_texts, ext_y = [], np.array([], dtype=np.int64)
    if not args.no_external and ext_path is not None:
        ext = load_external(ext_path, LABELS, args.max_per_model)
        ext_texts = ext["text"].tolist()
        ext_y = np.array([LABEL2IDX[m] for m in ext["model"]], dtype=np.int64)
        print(f"外部样本 {len(ext_texts)}（每模型上限 {args.max_per_model}），"
              f"覆盖 {ext['model'].nunique()}/18，权重 {args.ext_weight}")

    # ---------------- 预分词（只做一次）----------------
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    all_texts = texts + ext_texts + test_texts
    enc = tok(all_texts, truncation=True, max_length=args.max_len,
              padding=False, add_special_tokens=True)
    ids_all = enc["input_ids"]
    print(f"预分词 {len(all_texts)} 条，{time.time() - t0:.0f}s，"
          f"长度中位 {int(np.median([len(x) for x in ids_all]))}")

    n_tr, n_ext = len(texts), len(ext_texts)
    te_lo, te_hi = n_tr + n_ext, n_tr + n_ext + len(test_texts)

    class DS(Dataset):
        def __init__(self, indices):
            self.idx = list(indices)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            return ids_all[self.idx[i]]

    def collate(items):
        maxlen = max(len(x) for x in items)
        ids = torch.zeros(len(items), maxlen, dtype=torch.long)
        mask = torch.zeros(len(items), maxlen, dtype=torch.long)
        for i, x in enumerate(items):
            n = len(x)
            ids[i, :n] = torch.tensor(x, dtype=torch.long)
            mask[i, :n] = 1
        return ids, mask

    class TrainDS(Dataset):
        """训练集要带标签和权重，所以索引到 (id, label, weight)。"""

        def __init__(self, indices, labels, weights):
            self.idx = list(indices)
            self.lab = labels
            self.w = weights

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            k = self.idx[i]
            return ids_all[k], int(self.lab[i]), float(self.w[i])

    def train_collate(items):
        ids, mask = collate([x[0] for x in items])
        lab = torch.tensor([x[1] for x in items], dtype=torch.long)
        w = torch.tensor([x[2] for x in items], dtype=torch.float32)
        return ids, mask, lab, w

    @torch.no_grad()
    def predict(model, indices, bs=64):
        model.eval()
        out = []
        loader = DataLoader(DS(indices), batch_size=bs, shuffle=False, collate_fn=collate)
        for ids, mask in loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, N_CLASSES), dtype=np.float32)

    oof = np.zeros((len(train), N_CLASSES), dtype=np.float32)
    test_sum = np.zeros((len(test), N_CLASSES), dtype=np.float32)
    fold_scores = []
    done = 0

    splits = list(GroupKFold(n_splits=args.folds).split(texts, y, groups))

    for fold, (tr, va) in enumerate(splits, 1):
        if args.only_fold and fold != args.only_fold:
            continue
        t_fold = time.time()

        tr_idx = list(tr) + list(range(n_tr, n_tr + n_ext))
        tr_labels = np.concatenate([y[tr], ext_y]) if n_ext else y[tr]
        tr_w = np.concatenate([np.ones(len(tr), dtype=np.float32),
                               np.full(n_ext, args.ext_weight, dtype=np.float32)]) \
            if n_ext else np.ones(len(tr), dtype=np.float32)

        cnt = np.bincount(tr_labels, minlength=N_CLASSES).astype(np.float64)
        cw = 1.0 / np.sqrt(np.maximum(cnt, 1))
        class_w = torch.tensor((cw / cw.mean()).astype(np.float32), device=device)

        loader = DataLoader(TrainDS(tr_idx, tr_labels, tr_w), batch_size=args.bs,
                            shuffle=True, collate_fn=train_collate, drop_last=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=N_CLASSES).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total = max(1, len(loader) * args.epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=total, pct_start=0.1)

        print(f"\nfold {fold}: n_train={len(tr_idx)}  steps/epoch={len(loader)}")
        best_va, best_va_proba, best_te_proba = -1.0, None, None
        step = 0
        for ep in range(1, args.epochs + 1):
            model.train()
            tot_loss, n_seen = 0.0, 0
            for ids, mask, lab, ww in loader:
                ids, mask = ids.to(device), mask.to(device)
                lab, ww = lab.to(device), ww.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    per = F.cross_entropy(logits.float(), lab, weight=class_w,
                                          reduction="none")
                    loss = (per * ww).sum() / ww.sum().clamp(min=1e-6)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step < total - 1:
                    sched.step()
                step += 1
                tot_loss += float(loss.detach())
                n_seen += 1

            va_proba = predict(model, list(va))
            s = balanced_accuracy_score(y[va], va_proba.argmax(1))
            print(f"  epoch {ep}  loss={tot_loss / max(1, n_seen):.4f}  "
                  f"val bal.acc={s:.4f}  ({time.time() - t_fold:.0f}s)")
            if s > best_va:
                best_va, best_va_proba = s, va_proba
                best_te_proba = predict(model, list(range(te_lo, te_hi)))

        oof[va] = best_va_proba
        fold_scores.append(best_va)
        if best_te_proba is not None:
            test_sum += best_te_proba
        done += 1
        print(f"  fold {fold} 最佳 val = {best_va:.4f}")

        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------------- 结果 ----------------
    if done == args.folds:
        ba = balanced_accuracy_score(y, oof.argmax(1))
        print(f"\n>>> OOF balanced accuracy = {ba:.4f}   （各折 "
              f"{[round(s, 3) for s in fold_scores]}）")
        print("    对照：融合模型 OOF 0.3978（LB 0.381）")
        np.save("oof_clf.npy", oof)

    if len(test) and done:
        proba = test_sum / done
        np.save("test_proba_clf.npy", proba)
        pred = proba.argmax(1)
        sub = pd.DataFrame({"id": test["id"], "model": [LABELS[i] for i in pred]})
        sub.to_parquet(args.out, index=False)
        print(f"\n已写出 {args.out}（{len(sub)} 行，{done} 折平均）")


if __name__ == "__main__":
    main()
