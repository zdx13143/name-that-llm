"""Pinned ModernBERT download and synthetic-only CUDA probe; never loads data.

Run download first, then smoke. Both refuse to overwrite successful receipts.
A failed/incomplete smoke claim blocks another attempt, regardless of outputs.
Use an external process timeout (900 s download / 300 s smoke), per the plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

MODULE = Path(__file__).resolve().parents[1]
CONFIG_PATH = MODULE / "config/windows_modernbert_20261023.json"
OUTPUT = MODULE / "outputs/windows_4060"
CLAIM = MODULE / "outputs/continuation_claims/windows-modernbert-synthetic-20261023.json"
FILES = ("README.md", "config.json", "model.safetensors", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json")
TEXTS = (
    "The small boat reached the quiet shore before sunset.",
    'A careful writer asks: "What changed?" Then answers with three precise examples.',
    "Although the experiment is small, its records should preserve order, punctuation, and every relevant detail.",
    "This deliberately long synthetic sentence checks truncation and padding. " * 80,
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value, *, exclusive=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def validate_files(directory, manifest):
    """Bind local bytes to the public fixed-revision Git blob / LFS identity."""
    if {item["rfilename"] for item in manifest} != set(FILES):
        raise ValueError("Unexpected model file set")
    if {path.name for path in directory.iterdir() if path.name != ".cache"} != set(FILES):
        raise ValueError("Unexpected model directory contents")
    records = []
    for item in manifest:
        path = directory / item["rfilename"]
        if path.stat().st_size != item["size"]:
            raise ValueError("Model file size mismatch")
        sha = digest(path)
        if "lfs" in item:
            if sha != item["lfs"]["sha256"]:
                raise ValueError("Model LFS SHA256 mismatch")
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
            if blob != item["blobId"]:
                raise ValueError("Model Git blob identity mismatch")
        records.append({"file": item["rfilename"], "bytes": item["size"], "sha256": sha})
    return records


def download(config):
    receipt_path = OUTPUT / "modernbert_files.json"
    if receipt_path.exists():
        raise FileExistsError("Download receipt exists; inspect it instead of downloading again")
    endpoint = config["source"]
    if endpoint != "https://huggingface.co":
        raise ValueError("Only the preregistered official endpoint is allowed")
    os.environ["HF_HOME"] = str(OUTPUT / "hf_home")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import snapshot_download

    url = f'{endpoint}/api/models/{config["model_id"]}/revision/{config["revision"]}?blobs=true'
    with urllib.request.urlopen(url, timeout=30) as response:
        upstream = json.load(response)
    if upstream["sha"] != config["revision"]:
        raise ValueError("Upstream revision mismatch")
    selected = [item for item in upstream["siblings"] if item["rfilename"] in FILES]
    identities = []
    for item in selected:
        identity = {key: item[key] for key in ("rfilename", "size", "blobId")}
        if "lfs" in item:
            identity["lfs"] = {"sha256": item["lfs"]["sha256"]}
        identities.append(identity)
    if identities != config["model_files"]:
        raise ValueError("Upstream file identities differ from preregistered manifest")
    weights = next(item for item in selected if item["rfilename"] == "model.safetensors")
    if weights["lfs"]["sha256"] != config["weights_sha256"] or weights["size"] != config["weights_bytes"]:
        raise ValueError("Preregistered weight identity mismatch")
    directory = MODULE / "models/modernbert-base" / config["revision"]
    started = time.perf_counter()
    snapshot_download(config["model_id"], revision=config["revision"],
                      local_dir=str(directory), allow_patterns=list(FILES),
                      endpoint=endpoint, token=False, max_workers=2)
    records = validate_files(directory, config["model_files"])
    receipt = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
               "model_id": config["model_id"], "revision": config["revision"],
               "source": endpoint, "upstream_manifest": config["model_files"], "verified_files": records,
               "download_and_verification_seconds": time.perf_counter() - started,
               "model_forward_calls": 0, "training_fits": 0}
    write_json(receipt_path, receipt)
    print(json.dumps({"status": "verified", "files": len(records), "revision": config["revision"]}))


def masked_pool(hidden, attention_mask, layers):
    import torch
    mask = attention_mask.unsqueeze(-1).float()
    if not bool((mask.sum(1) > 0).all()):
        raise ValueError("Empty attention mask")
    return torch.cat([(hidden[layer].float() * mask).sum(1) / mask.sum(1)
                      for layer in layers], dim=1)


def smoke(config):
    receipt_path = OUTPUT / "modernbert_smoke.json"
    if receipt_path.exists() or CLAIM.exists():
        raise FileExistsError("Synthetic probe already claimed; preserve the previous result")
    claim = {"state": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "config_sha256": hashlib.sha256(canonical(config).encode()).hexdigest(),
             "script_sha256": digest(__file__), "attempted_forward_calls": 0,
             "completed_forward_calls": 0, "training_fits": 0,
             "competition_records": 0, "final_holdout_evaluations": 0}
    write_json(CLAIM, claim)
    started = time.perf_counter()
    try:
        downloaded = json.loads((OUTPUT / "modernbert_files.json").read_text(encoding="utf-8"))
        if downloaded["revision"] != config["revision"] or downloaded["model_id"] != config["model_id"]:
            raise ValueError("Download receipt identity mismatch")
        if downloaded["upstream_manifest"] != config["model_files"]:
            raise ValueError("Download manifest differs from preregistered file identities")
        directory = MODULE / "models/modernbert-base" / config["revision"]
        records = validate_files(directory, config["model_files"])
        weight = next(item for item in records if item["file"] == "model.safetensors")
        if weight["sha256"] != config["weights_sha256"]:
            raise ValueError("Frozen weight SHA mismatch")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is unavailable; no dtype fallback allowed")
        if torch.__version__ != "2.8.0+cu128" or importlib.metadata.version("transformers") != "4.56.2":
            raise RuntimeError("Unexpected preregistered runtime")
        torch.set_num_threads(config["budget"]["threads"])
        torch.manual_seed(config["feature"]["seed"])
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True, torch_dtype=torch.bfloat16,
                                         attn_implementation="sdpa", reference_compile=False)
        model.requires_grad_(False).eval().to("cuda:0")
        if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
            raise ValueError("Encoder was not frozen")
        feature = config["feature"]
        raw_lengths = [len(ids) for ids in tokenizer(list(TEXTS), truncation=False)["input_ids"]]
        outputs, timings = [], []
        for texts in (list(TEXTS), list(reversed(TEXTS))):
            if claim["attempted_forward_calls"] >= config["budget"]["synthetic_forward_calls"]:
                raise RuntimeError("Forward budget exhausted")
            batch = tokenizer(texts, padding="max_length", truncation=True,
                              max_length=feature["max_length"], return_tensors="pt").to("cuda:0")
            claim["attempted_forward_calls"] += 1
            write_json(CLAIM, claim, exclusive=False)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                result = model(**batch, output_hidden_states=True)
                embedding = masked_pool(result.hidden_states, batch["attention_mask"], feature["layers"])
                # Check masked pooling against explicit per-row non-padding selection.
                reference = torch.stack([torch.cat([result.hidden_states[layer][row][batch["attention_mask"][row].bool()].float().mean(0)
                                                   for layer in feature["layers"]]) for row in range(len(texts))])
                torch.testing.assert_close(embedding, reference, atol=2e-5, rtol=2e-5)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - tick)
            if list(embedding.shape) != [4, feature["dimensions"]] or not bool(torch.isfinite(embedding).all()):
                raise ValueError("Invalid synthetic embedding")
            outputs.append(embedding.cpu())
            claim["completed_forward_calls"] += 1
            write_json(CLAIM, claim, exclusive=False)
            del result, embedding, reference, batch
            if torch.cuda.max_memory_allocated() > config["budget"]["max_cuda_allocated_gib"] * 2**30:
                raise RuntimeError("CUDA allocated memory budget exceeded")
        difference = float((outputs[0] - outputs[1].flip(0)).abs().max())
        if difference > 0.02:
            raise ValueError("Batch order consistency failed")
        report = {**claim, "state": "completed", "scope": "four_synthetic_texts_only",
                  "config": config, "verified_files": records,
                  "package_versions": {name: importlib.metadata.version(name) for name in
                                       ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "numpy")},
                  "gpu": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
                  "embedding_shape": [4, feature["dimensions"]], "raw_token_lengths": raw_lengths,
                  "truncated_records": sum(n > feature["max_length"] for n in raw_lengths),
                  "input_sha256": hashlib.sha256(canonical(TEXTS).encode()).hexdigest(),
                  "embedding_sha256": hashlib.sha256(outputs[0].numpy().tobytes()).hexdigest(),
                  "batch_order_max_absolute_difference": difference, "forward_seconds": timings,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "wall_seconds": time.perf_counter() - started,
                  "limitations": "No competition data, classification fit, full-development benchmark, Kaggle run or score."}
        write_json(receipt_path, report)
        claim.update(state="completed", result_sha256=digest(receipt_path))
        write_json(CLAIM, claim, exclusive=False)
        print(json.dumps({key: report[key] for key in ("state", "embedding_shape", "forward_seconds",
                                                     "peak_cuda_allocated_bytes", "batch_order_max_absolute_difference")}))
    except BaseException as error:
        claim.update(state="failed", exception_type=type(error).__name__, wall_seconds=time.perf_counter() - started)
        write_json(CLAIM, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "smoke"))
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    {"download": download, "smoke": smoke}[args.action](config)


if __name__ == "__main__":
    main()
