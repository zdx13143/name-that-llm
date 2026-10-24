"""Fixed supervised-training mechanics, with no data paths or experiment entry point."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def update_batches(order, micro_batch_size, accumulation):
    """Each update sees <= effective-batch samples, including every final sample."""
    if micro_batch_size < 1 or accumulation < 1:
        raise ValueError("Batch sizes must be positive")
    for offset in range(0, len(order), micro_batch_size * accumulation):
        window = order[offset:offset + micro_batch_size * accumulation]
        yield [window[i:i + micro_batch_size] for i in range(0, len(window), micro_batch_size)]


def learning_rate_factor(step, total_steps, warmup_fraction):
    """Zero-based step; linear nonzero warmup, then remaining-step decay."""
    if not 0 <= step < total_steps or not 0 <= warmup_fraction < 1:
        raise ValueError("Invalid fixed scheduler position")
    warmup = max(1, math.ceil(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    return (total_steps - step) / max(1, total_steps - warmup)


def optimizer_for(model, method):
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError("This protocol requires all model parameters to be trainable")
        lr = method["encoder_learning_rate"] if name.startswith("model.") else method["head_learning_rate"]
        decay = method["weight_decay"] if parameter.ndim > 1 and not name.endswith("bias") else 0.0
        buckets.setdefault((lr, decay), []).append(parameter)
    groups = [{"params": params, "lr": lr, "initial_lr": lr, "weight_decay": decay}
              for (lr, decay), params in buckets.items()]
    return torch.optim.AdamW(groups, foreach=False, fused=False)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def train_epoch(model, encoded, truth, optimizer, method, order, step, total_steps,
                hooks, device, *, phase="train"):
    """Only training tensors enter; no validation data, labels, or epoch selection."""
    if (len(truth) != len(encoded["input_ids"]) or len(order) != len(truth)
            or not np.array_equal(np.sort(order), np.arange(len(truth)))):
        raise ValueError("Training order must cover each permitted training row exactly once")
    model.train()
    total_loss, seen, microbatches, steps = 0.0, 0, 0, 0
    for chunks in update_batches(order, method["micro_batch_size"], method["gradient_accumulation"]):
        examples = sum(len(chunk) for chunk in chunks)
        factor = learning_rate_factor(step, total_steps, method["warmup_fraction"])
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor
        optimizer.zero_grad(set_to_none=True)
        for chunk in chunks:
            selection = torch.as_tensor(chunk, dtype=torch.long)
            batch = {key: value[selection].to(device) for key, value in encoded.items()}
            targets = truth[selection].to(device)
            hooks.before(phase + "_forward")
            with autocast(device):
                logits = model(**batch).logits
            loss_sum = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
            if not torch.isfinite(loss_sum).item():
                raise FloatingPointError("Non-finite training loss")
            hooks.after(phase + "_forward")
            hooks.before(phase + "_backward")
            (loss_sum / examples).backward()
            hooks.after(phase + "_backward")
            total_loss += float(loss_sum.detach())
            seen += len(chunk)
            microbatches += 1
            del logits, loss_sum, batch, targets
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), method["gradient_clip_norm"], error_if_nonfinite=True)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("Non-finite gradient norm")
        hooks.before(phase + "_step")
        optimizer.step()
        hooks.after(phase + "_step")
        step += 1
        steps += 1
    optimizer.zero_grad(set_to_none=True)
    return {"mean_training_loss": total_loss / seen, "examples_seen": seen,
            "microbatches": microbatches, "optimizer_steps": steps,
            "order_sha256": hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()}, step


@torch.inference_mode()
def predict(model, encoded, batch_size, hooks, device, *, phase="valid"):
    model.eval()
    result = []
    for start in range(0, len(encoded["input_ids"]), batch_size):
        batch = {key: value[start:start + batch_size].to(device) for key, value in encoded.items()}
        hooks.before(phase + "_forward")
        with autocast(device):
            logits = model(**batch).logits
        probability = torch.softmax(logits.float(), dim=-1)
        if not torch.isfinite(probability).all().item():
            raise FloatingPointError("Non-finite prediction")
        result.append(probability.cpu().numpy())
        hooks.after(phase + "_forward")
    if not result:
        raise ValueError("Empty validation set")
    return np.concatenate(result)
