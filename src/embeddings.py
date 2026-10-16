"""用预训练语言模型的隐藏状态做特征。

为什么这条路可能有用
--------------------
前面 8 个 TF-IDF 视图全部卡在 0.28~0.32，说明 n-gram 能提供的风格信息已经
接近榨干。预训练模型在大量文本上训练过，它的中间层表示对「这段话是怎么写的」
比 n-gram 敏感得多 —— 这正是我们区分同机构相近型号所缺的信号。

两个必须注意的点
----------------
1. **编码器必须冻结**。不在这份比赛数据上做任何训练，只做前向推理，
   所以对全部文本算一次嵌入不存在标签泄漏，可以整体缓存复用。
   （以后做 LoRA 微调就必须按折重算，不能这么干。）
2. hf.co 在这台机器上被墙，必须走 hf-mirror.com 镜像。
"""

from __future__ import annotations

import os

# 必须在导入 transformers / huggingface_hub 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402


def pick_device(prefer: str | None = None) -> str:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_embeddings(
    texts: list[str],
    model_name: str,
    layers: list[int] | None = None,
    pooling: str = "mean",
    batch_size: int = 16,
    max_length: int = 256,
    device: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """返回 (n_texts, dim) 的嵌入矩阵。

    layers 指定取哪些层的隐藏状态并拼接；None 表示取最后 4 层。
    取多层拼接通常比只取最后一层更能捕捉风格。
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device(device)
    if verbose:
        print(f"  模型 {model_name}  设备 {device}")
        print(f"  HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device).eval()

    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(max(0, n_layers - 3), n_layers + 1))
    if verbose:
        print(f"  共 {n_layers} 层，取层 {layers}")

    chunks: list[np.ndarray] = []
    use_amp = device == "cuda"

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(**enc)
            else:
                out = model(**enc)

        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        feats = []
        for L in layers:
            h = out.hidden_states[L].float()  # (B, T, H)
            if pooling == "mean":
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            elif pooling == "last":
                idx = mask.sum(1).long().clamp(min=1) - 1
                pooled = h[torch.arange(h.size(0)), idx.squeeze(-1)]
            else:
                raise ValueError(pooling)
            feats.append(pooled.cpu().numpy())
        chunks.append(np.concatenate(feats, axis=1).astype(np.float32))

        if verbose and (start // batch_size) % 20 == 0:
            done = min(start + batch_size, len(texts))
            print(f"    已编码 {done}/{len(texts)}")

    del model
    import torch as _t
    if device == "cuda":
        _t.cuda.empty_cache()

    X = np.vstack(chunks)
    if verbose:
        print(f"  嵌入矩阵 {X.shape}")
    return X


def cache_path(out_dir, model_name: str, tag: str = "") -> str:
    safe = model_name.replace("/", "__")
    suffix = f"_{tag}" if tag else ""
    return str(out_dir / f"emb_{safe}{suffix}.npy")
