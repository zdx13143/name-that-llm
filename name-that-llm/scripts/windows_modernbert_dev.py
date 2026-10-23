"""One bounded Windows development run, using the existing 120-prompt partition.

Default is read-only readiness. --run consumes one fixed claim, including failed
attempts. No legacy runner imports, cache resume, full-data fit or submission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np

from probe_modernbert import canonical, digest, masked_pool, validate_files, write_json
from restore_team_split import historical_manifest_bytes

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_modernbert_standalone_20261024.json"
BASE = MODULE / "config/windows_modernbert_20261023.json"
LABELS = MODULE / "config/labels.json"
OUTPUT = MODULE / "outputs/windows_modernbert_dev_20261023"
# The amendment shares the original development allowance; it adds no new fits.
CLAIM = MODULE / "outputs/continuation_claims/windows-modernbert-frozen-dev-20261023.json"
SOURCES = (Path(__file__), Path(__file__).with_name("probe_modernbert.py"),
           Path(__file__).with_name("restore_team_split.py"), CONFIG, BASE, LABELS,
           MODULE / "requirements-windows-dev.txt", MODULE / "requirements-windows-gpu.txt")


def object_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def configuration():
    config = json.loads(CONFIG.read_bytes())
    base = json.loads(BASE.read_bytes())
    if (config["experiment_id"] != base["experiment_id"]
            or config["base_config"] != BASE.name
            or config["evaluation"]["classifier"] != base["evaluation"]["classifier"]
            or config["evaluation"]["folds"] != 5):
        raise ValueError("Development amendment differs from its fixed parent plan")
    implemented = {"dtype": "bfloat16", "pooling_dtype": "float32", "attention": "sdpa",
                   "reference_compile": False, "trust_remote_code": False,
                   "pooling": "attention-mask mean including special tokens, concatenate layers"}
    if any(base["feature"][key] != value for key, value in implemented.items()):
        raise ValueError("Feature declaration differs from the implemented extraction")
    return config, base


def load_development(train, input_config, labels_path=LABELS):
    """Reconstruct only split metadata, then retain only permitted texts/truth."""
    if digest(train) != input_config["train_sha256"]:
        raise ValueError("Frozen train SHA256 mismatch")
    if digest(labels_path) != input_config["labels_sha256"]:
        raise ValueError("Frozen labels SHA256 mismatch")
    labels = json.loads(labels_path.read_bytes())["labels"]
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError("Expected 18 distinct labels")
    metadata = []
    with train.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if sorted(reader.fieldnames or []) != sorted(["id", "prompt_id", "text", "model"]):
            raise ValueError("Training schema changed")
        for row in reader:
            if None in row or not row.get("id") or not row.get("prompt_id"):
                raise ValueError("Invalid split metadata")
            metadata.append({"id": row["id"], "prompt_id": row["prompt_id"]})
    if len({row["id"] for row in metadata}) != len(metadata):
        raise ValueError("Duplicate input IDs")
    payload = historical_manifest_bytes(metadata)
    if hashlib.sha256(payload).hexdigest() != input_config["split_sha256"]:
        raise ValueError("Reconstructed historical split SHA256 mismatch")
    manifest = json.loads(payload)
    dev, hold = manifest["development_indices"], manifest["holdout_indices"]
    groups = np.asarray([row["prompt_id"] for row in metadata])
    if (len(dev) != input_config["development_rows"] or len(hold) != input_config["protected_rows"]
            or len(set(groups[dev])) != input_config["development_groups"]
            or len(set(groups[hold])) != input_config["protected_groups"]
            or set(dev) & set(hold) or set(dev) | set(hold) != set(range(len(metadata)))
            or set(groups[dev]) & set(groups[hold])):
        raise ValueError("Historical development boundary changed")
    selected, wanted = {}, set(dev)
    with train.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index not in wanted:
                continue
            if (None in row or any(not row.get(key, "").strip() for key in ("id", "prompt_id", "text", "model"))
                    or row["model"] not in labels
                    or any(row[key] != metadata[index][key] for key in ("id", "prompt_id"))):
                raise ValueError("Invalid development record")
            selected[index] = row
    if digest(train) != input_config["train_sha256"] or len(selected) != len(dev):
        raise ValueError("Input changed while reading")
    rows = [selected[i] for i in dev]
    return {"row_index": np.asarray(dev, dtype=np.int64),
            "ids": np.asarray([row["id"] for row in rows]), "groups": groups[dev],
            "texts": np.asarray([row["text"] for row in rows]), "labels": np.asarray(labels),
            "truth": np.asarray([labels.index(row["model"]) for row in rows], dtype=np.int64)}


def grouped_folds(data, n_splits=5):
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=n_splits, shuffle=False).split(data["texts"], data["truth"], data["groups"]))
    fold_id = np.full(len(data["truth"]), -1, dtype=np.int64)
    for number, (train, valid) in enumerate(folds):
        if (set(data["groups"][train]) & set(data["groups"][valid])
                or np.any(fold_id[valid] != -1)
                or set(data["truth"][train]) != set(range(len(data["labels"])) )):
            raise ValueError("Invalid development fold")
        fold_id[valid] = number
    if np.any(fold_id < 0):
        raise ValueError("Incomplete fold coverage")
    return folds, fold_id


def contract(data, config, base, verified_files):
    _, fold_id = grouped_folds(data, config["evaluation"]["folds"])
    return {"config": config, "base_config": base, "model_files": verified_files,
            "sources": {str(p.relative_to(MODULE)): digest(p) for p in SOURCES},
            "text_sha256": object_hash(data["texts"].tolist()),
            "row_identity_sha256": object_hash({key: data[key].tolist() for key in
                                                 ("row_index", "ids", "groups", "truth", "labels")}),
            "fold_sha256": object_hash(fold_id.tolist()), "rows": len(fold_id),
            "package_versions": {name: importlib.metadata.version(name) for name in config["runtime"]},
            "python": sys.version, "platform": sys.platform}


def readiness(train, config, base):
    blockers = []
    if sys.platform != "win32":
        blockers.append("This registered run requires native Windows")
    for name, expected in config["runtime"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            blockers.append(f"Runtime {name}: {actual}; expected {expected}")
    if not train.is_file():
        blockers.append(f"Missing official training input: {train}")
    if CLAIM.exists() or OUTPUT.exists():
        blockers.append("Development claim or output already exists; no retry/resume is allowed")
    if not LABELS.is_file() or digest(LABELS) != base["input"]["labels_sha256"]:
        blockers.append("Frozen labels missing or SHA256 mismatch")
    directory = MODULE / "models/modernbert-base" / base["revision"]
    files = []
    try:
        files = validate_files(directory, base["model_files"])
    except (OSError, ValueError) as error:
        blockers.append(f"Model verification failed: {type(error).__name__}: {error}")
    if blockers:
        return {"state": "blocked", "blockers": blockers, "verified_model_files": len(files)}, None
    try:
        data = load_development(train, base["input"])
        value = contract(data, config, base, files)
    except (OSError, ValueError) as error:
        return {"state": "blocked", "blockers": [f"Input verification failed: {type(error).__name__}: {error}"]}, None
    return {"state": "ready", "contract_sha256": object_hash(value), "development_rows": len(data["truth"])}, value


def save_array(path, name, array, data, fold_id, contract_hash):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **{name: array}, **{key: data[key] for key in
                            ("row_index", "ids", "groups", "truth", "labels")},
                            fold_id=fold_id, contract_sha256=np.asarray(contract_hash))


def load_features(path, data, fold_id, contract_hash, dimensions):
    expected = {key: data[key] for key in ("row_index", "ids", "groups", "truth", "labels")}
    expected.update(fold_id=fold_id, contract_sha256=np.asarray(contract_hash))
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != set(expected) | {"features"}:
            raise ValueError("Feature cache schema differs")
        for key, value in expected.items():
            if not np.array_equal(artifact[key], value):
                raise ValueError(f"Feature cache identity differs: {key}")
        features = artifact["features"].copy()
    if features.dtype != np.float32 or features.shape != (len(fold_id), dimensions) or not np.isfinite(features).all():
        raise ValueError("Invalid feature matrix")
    return features


def consume(claim, key, limit, path=CLAIM):
    if claim[key] >= limit:
        raise RuntimeError(f"Budget exhausted: {key}")
    claim[key] += 1
    write_json(path, claim, exclusive=False)  # Persistent before the operation.


def extract_features(data, base, claim):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable; no fallback")
    torch.set_num_threads(base["budget"]["threads"])
    torch.set_num_interop_threads(base["budget"]["threads"])
    torch.manual_seed(base["feature"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    directory = MODULE / "models/modernbert-base" / base["revision"]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, torch_dtype=torch.bfloat16,
                                     attn_implementation="sdpa", reference_compile=False)
    model.requires_grad_(False).eval().to("cuda:0")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Encoder must be frozen")
    feature, budget = base["feature"], base["budget"]
    loaded = {"hidden_size": model.config.hidden_size, "num_hidden_layers": model.config.num_hidden_layers,
              "max_position_embeddings": model.config.max_position_embeddings,
              "attention": model.config._attn_implementation, "reference_compile": model.config.reference_compile,
              "dtype": str(next(model.parameters()).dtype), "padding_side": tokenizer.padding_side,
              "truncation_side": tokenizer.truncation_side,
              "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}
    if (loaded["hidden_size"] * len(feature["layers"]) != feature["dimensions"]
            or any(layer >= 0 or -layer > loaded["num_hidden_layers"] for layer in feature["layers"])
            or feature["max_length"] > loaded["max_position_embeddings"]
            or loaded["attention"] != feature["attention"] or loaded["reference_compile"] is not False
            or loaded["dtype"] != "torch.bfloat16" or tokenizer.pad_token_id != model.config.pad_token_id
            or tokenizer.cls_token_id != model.config.cls_token_id or tokenizer.sep_token_id != model.config.sep_token_id):
        raise ValueError("Loaded model/tokenizer settings differ from the feature declaration")
    result = np.empty((len(data["texts"]), feature["dimensions"]), dtype=np.float32)
    for start in range(0, len(result), feature["batch_size"]):
        texts = data["texts"][start:start + feature["batch_size"]].tolist()
        batch = tokenizer(texts, padding=feature["padding"], truncation=True,
                          max_length=feature["max_length"], return_tensors="pt").to("cuda:0")
        consume(claim, "attempted_forward_calls", budget["max_extraction_batches"])
        with torch.inference_mode():
            output = model(**batch, output_hidden_states=True)
            embedded = masked_pool(output.hidden_states, batch["attention_mask"], feature["layers"])
        if list(embedded.shape) != [len(texts), feature["dimensions"]] or not bool(torch.isfinite(embedded).all()):
            raise ValueError("Invalid embedding")
        result[start:start + len(texts)] = embedded.cpu().numpy()
        claim["completed_forward_calls"] += 1
        write_json(CLAIM, claim, exclusive=False)
        del output, embedded, batch
        if torch.cuda.max_memory_allocated() > budget["max_cuda_allocated_gib"] * 2**30:
            raise RuntimeError("CUDA allocated memory budget exceeded")
    gpu = {"device": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda, "loaded_settings": loaded,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    del model
    torch.cuda.empty_cache()
    return result, gpu


def probabilities(value, rows, classes):
    if value.shape != (rows, classes) or value.dtype.kind != "f":
        raise ValueError("Invalid probability shape/dtype")
    tolerance = max(1e-8, min(1e-6, np.finfo(value.dtype).eps * classes))
    if (not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)
            or not np.allclose(value.sum(1), 1, atol=tolerance, rtol=0)):
        raise ValueError("Invalid probability values")
    value = value.astype(np.float64, copy=True)
    return value / value.sum(1, keepdims=True)


def fit_head(features, data, folds, evaluation, claim, max_attempts, claim_path=CLAIM):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof = np.full((len(features), len(data["labels"])), np.nan)
    timings = []
    for train, valid in folds:
        model = make_pipeline(StandardScaler(), LogisticRegression(**evaluation["classifier"]))
        consume(claim, "attempted_fits", max_attempts, claim_path)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(features[train], data["truth"][train])
        claim["completed_fits"] += 1
        write_json(claim_path, claim, exclusive=False)
        if not np.array_equal(model.classes_, np.arange(len(data["labels"]))):
            raise ValueError("Classifier label order changed")
        oof[valid] = probabilities(model.predict_proba(features[valid]), len(valid), len(data["labels"]))
        timings.append(time.perf_counter() - started)
    return probabilities(oof, len(features), len(data["labels"])), timings


def metrics(oof, data, fold_id, evaluation):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
    prediction, truth = oof.argmax(1), data["truth"]
    groups, gi = np.unique(data["groups"], return_inverse=True)
    counts = np.zeros((len(groups), len(data["labels"])), dtype=np.int64)
    hits = np.zeros_like(counts)
    np.add.at(counts, (gi, truth), 1)
    np.add.at(hits, (gi, truth), prediction == truth)
    rng = np.random.default_rng(evaluation["bootstrap_seed"])
    draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)),
                           size=evaluation["bootstrap_resamples"])
    denominators = draws @ counts
    if np.any(denominators == 0):
        raise ValueError("Bootstrap sample lacks a class")
    return {"balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "fold_balanced_accuracy": [float(balanced_accuracy_score(truth[fold_id == i], prediction[fold_id == i]))
                                       for i in range(evaluation["folds"])],
            "per_class_recall": dict(zip(data["labels"].tolist(), recall_score(truth, prediction,
                                      labels=np.arange(len(data["labels"])), average=None, zero_division=0).tolist())),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(len(data["labels"]))).tolist(),
            "prompt_bootstrap_ba_95ci": np.quantile(((draws @ hits) / denominators).mean(1), [.025, .975]).tolist()}


def execute(train, config, base):
    claim = json.loads(CLAIM.read_bytes())
    if claim["state"] != "started" or claim["supervisor_pid"] != os.getppid():
        raise RuntimeError("Execution requires the live supervisor and an unused claim")
    data = load_development(train, base["input"])
    verified = validate_files(MODULE / "models/modernbert-base" / base["revision"], base["model_files"])
    current = contract(data, config, base, verified)
    if object_hash(current) != claim["contract_sha256"]:
        raise ValueError("Run contract changed after readiness")
    folds, fold_id = grouped_folds(data, config["evaluation"]["folds"])
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    with threadpool_limits(limits=base["budget"]["threads"]):
        features, gpu = extract_features(data, base, claim)
    extraction_seconds = time.perf_counter() - started
    cache = OUTPUT / "features.npz"
    save_array(cache, "features", features, data, fold_id, claim["contract_sha256"])
    features = load_features(cache, data, fold_id, claim["contract_sha256"], base["feature"]["dimensions"])
    with threadpool_limits(limits=base["budget"]["threads"]):
        oof, timings = fit_head(features, data, folds, config["evaluation"], claim,
                               base["budget"]["max_classifier_fit_attempts"])
    save_array(OUTPUT / "oof.npz", "probabilities", oof, data, fold_id, claim["contract_sha256"])
    report = {"scope": config["scope"], "contract_sha256": claim["contract_sha256"],
              "metrics": metrics(oof, data, fold_id, config["evaluation"]), "gpu": gpu,
              "extraction_seconds": extraction_seconds, "fold_fit_and_predict_seconds": timings,
              "features_sha256": digest(cache), "oof_sha256": digest(OUTPUT / "oof.npz"),
              "attempted_fits": claim["attempted_fits"], "completed_fits": claim["completed_fits"],
              "final_holdout_evaluations": 0, "full_data_final_fits": 0,
              "limitations": "Descriptive development evidence only; no paired baseline, model promotion or Kaggle score."}
    write_json(OUTPUT / "summary.json", report)
    claim.update(state="worker_completed", summary_sha256=digest(OUTPUT / "summary.json"))
    write_json(CLAIM, claim, exclusive=False)


def terminate_worker(process):
    """Record termination trouble instead of losing the original failed claim."""
    import psutil
    result = {"pid": process.pid, "errors": []}
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            result["errors"].append(f"child {child.pid}: {type(error).__name__}")
    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            result["errors"].append(f"worker: {type(error).__name__}")
    result["worker_still_running"] = process.poll() is None
    result["returncode"] = process.returncode
    return result


def supervise(train, value, base):
    """External wall-clock/RSS guard works even while a native fit blocks Python."""
    import psutil
    claim = {"state": "started", "experiment_id": value["config"]["experiment_id"],
             "supervisor_pid": os.getpid(), "contract_sha256": object_hash(value),
             "attempted_forward_calls": 0, "completed_forward_calls": 0,
             "attempted_fits": 0, "completed_fits": 0, "final_holdout_evaluations": 0,
             "full_data_final_fits": 0, "started_unix_seconds": time.time()}
    write_json(CLAIM, claim)  # Fixed exclusive path, independent of train/output arguments.
    process, phase = None, "pre_spawn"
    started, peak_rss = time.perf_counter(), 0
    try:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        write_json(OUTPUT / "contract.json", value)
        for source in SOURCES:
            target = OUTPUT / "source_snapshot" / source.relative_to(MODULE)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "TOKENIZERS_PARALLELISM": "false"}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--train", str(train), "--execute"],
                                       env=env, stdout=stdout, stderr=stderr)
            phase = "worker"
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter() - started > base["budget"]["development_timeout_seconds"]:
                    raise TimeoutError("Development wall-clock budget exceeded")
                try:
                    rss = (psutil.Process().memory_info().rss + monitored.memory_info().rss
                           + sum(p.memory_info().rss for p in monitored.children(recursive=True)))
                    peak_rss = max(peak_rss, rss)
                    if rss > base["budget"]["max_host_rss_gib"] * 2**30:
                        raise MemoryError("Development process tree RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed"
                or claim["attempted_fits"] != 5 or claim["completed_fits"] != 5
                or digest(OUTPUT / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Worker failed or returned an incomplete result; inspect retained logs")
        receipt = {"state": "completed", "wall_seconds": time.perf_counter() - started,
                   "sampled_peak_process_tree_rss_bytes": peak_rss, "exit_code": process.returncode,
                   "summary_sha256": claim["summary_sha256"]}
        write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(OUTPUT / "supervisor.json"))
        write_json(CLAIM, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = terminate_worker(process)
        except BaseException as cleanup_error:
            termination = {"cleanup_exception": type(cleanup_error).__name__, "pid": process.pid}
        finally:
            claim = json.loads(CLAIM.read_bytes())
            claim.update(state="failed", exception_type=type(error).__name__, exception_message=str(error),
                         phase=phase, termination=termination, wall_seconds=time.perf_counter() - started,
                         sampled_peak_process_tree_rss_bytes=peak_rss)
            write_json(CLAIM, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=MODULE / "data/raw/train.csv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config, base = configuration()
    train = args.train.resolve()
    if args.execute:
        execute(train, config, base)
        return 0
    state, value = readiness(train, config, base)
    if state["state"] != "ready" or not args.run:
        print(json.dumps(state, ensure_ascii=False))
        return 0 if state["state"] == "ready" else 2
    print(json.dumps(supervise(train, value, base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
