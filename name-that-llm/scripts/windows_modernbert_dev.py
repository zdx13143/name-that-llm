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
ORIGINAL_CLAIM, ORIGINAL_OUTPUT = CLAIM, OUTPUT
RECOVERY_CONFIG = MODULE / "config/windows_modernbert_startup_recovery_20261024.json"
RECOVERY_CONFIG_SHA256 = "f30618fa4b2c268e70c3e13466af1504e10e16b4963dd8f52a8eedcdde86295f"
RECOVERY = None
ROUND2 = False
ROUND2_CONFIG = MODULE / "config/windows_modernbert_round2_20261024.json"
ROUND2_CONFIG_SHA256 = "32bb45624a6969be47db8802be9c84a47d5cda78a9ad07e6453fcf6c89c8c5b5"
SOURCES = (Path(__file__), Path(__file__).with_name("probe_modernbert.py"),
           Path(__file__).with_name("restore_team_split.py"), CONFIG, BASE, LABELS,
           MODULE / "requirements-windows-dev.txt", MODULE / "requirements-windows-gpu.txt")


def object_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def verify_preserved_claims(config):
    for relative, expected in config["preserved_claims"].items():
        path = (MODULE / relative).resolve()
        if not path.is_relative_to((MODULE / "outputs/continuation_claims").resolve()) or digest(path) != expected:
            raise ValueError("Preserved prior-round claim identity differs")


def activate_round2():
    global CONFIG, CLAIM, OUTPUT, SOURCES, ROUND2
    if RECOVERY is not None or digest(ROUND2_CONFIG) != ROUND2_CONFIG_SHA256:
        raise ValueError("Round 2 protocol identity differs or recovery is active")
    config = json.loads(ROUND2_CONFIG.read_bytes())
    if config["experiment_id"] != "windows-modernbert-frozen-dev-20261024-r2":
        raise ValueError("Unexpected round 2 experiment")
    verify_preserved_claims(config)
    previous_config = CONFIG
    CONFIG = ROUND2_CONFIG
    CLAIM = MODULE / "outputs/continuation_claims/windows-modernbert-frozen-dev-20261024-r2.json"
    OUTPUT = MODULE / "outputs/windows_modernbert_dev_20261024_r2"
    SOURCES = tuple(CONFIG if path == previous_config else path for path in SOURCES)
    ROUND2 = True


def verify_zero_work_startup(amendment, claim_path, output):
    paths = {"claim": claim_path, "contract": output / "contract.json",
             "stderr": output / "stderr.log", "stdout": output / "stdout.log"}
    if any(digest(path) != amendment["original_artifact_sha256"][key] for key, path in paths.items()):
        raise ValueError("Startup recovery does not match the preserved failed attempt")
    prior = json.loads(claim_path.read_bytes())
    counters = ("attempted_forward_calls", "completed_forward_calls", "attempted_fits",
                "completed_fits", "final_holdout_evaluations", "full_data_final_fits")
    if (prior.get("state") != "failed" or prior.get("phase") != "reconcile"
            or prior.get("experiment_id") != amendment["experiment_id"]
            or any(prior.get(key) != 0 for key in counters)
            or prior.get("termination", {}).get("worker_still_running") is not False
            or prior.get("termination", {}).get("returncode") != 1):
        raise ValueError("Only this zero-work startup failure can use the explicit recovery")
    if {path.name for path in output.iterdir()} != {"contract.json", "source_snapshot", "stderr.log", "stdout.log"}:
        raise ValueError("Unexpected artifacts in the failed startup")
    previous = json.loads((output / "contract.json").read_bytes())
    if object_hash(previous) != prior["contract_sha256"]:
        raise ValueError("Original failed contract identity differs")
    for relative, expected in previous["sources"].items():
        target = (output / "source_snapshot" / relative).resolve()
        if not target.is_relative_to((output / "source_snapshot").resolve()) or digest(target) != expected:
            raise ValueError("Original failed source snapshot differs")
    return previous


def activate_startup_recovery():
    """One named correction; the original failed claim is never overwritten."""
    global CLAIM, OUTPUT, RECOVERY
    if digest(RECOVERY_CONFIG) != RECOVERY_CONFIG_SHA256:
        raise ValueError("Startup recovery amendment identity differs")
    amendment = json.loads(RECOVERY_CONFIG.read_bytes())
    if amendment["experiment_id"] != "windows-modernbert-frozen-dev-20261023" or amendment["recovery_number"] != 1:
        raise ValueError("Unexpected startup recovery amendment")
    previous = verify_zero_work_startup(amendment, ORIGINAL_CLAIM, ORIGINAL_OUTPUT)
    RECOVERY = {"amendment": amendment, "amendment_sha256": digest(RECOVERY_CONFIG),
                "original_contract": previous}
    CLAIM = MODULE / "outputs/continuation_claims/windows-modernbert-frozen-dev-20261023-startup-recovery-1.json"
    OUTPUT = MODULE / "outputs/windows_modernbert_dev_20261023_startup_recovery_1"


def configuration():
    config = json.loads(CONFIG.read_bytes())
    base = json.loads(BASE.read_bytes())
    parent_id = config["parent_experiment_id"] if ROUND2 else config["experiment_id"]
    if (parent_id != base["experiment_id"]
            or config["base_config"] != BASE.name
            or config["evaluation"]["classifier"] != base["evaluation"]["classifier"]
            or config["evaluation"]["folds"] != 5):
        raise ValueError("Development amendment differs from its fixed parent plan")
    if ROUND2 and any(base["budget"][key] != value for key, value in config["budget"].items()):
        raise ValueError("Round 2 execution budget differs from its registration")
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
    if ROUND2:
        verify_preserved_claims(config)
    _, fold_id = grouped_folds(data, config["evaluation"]["folds"])
    value = {"config": config, "base_config": base, "model_files": verified_files,
            "sources": {str(p.relative_to(MODULE)): digest(p) for p in SOURCES},
            "text_sha256": object_hash(data["texts"].tolist()),
            "row_identity_sha256": object_hash({key: data[key].tolist() for key in
                                                 ("row_index", "ids", "groups", "truth", "labels")}),
            "fold_sha256": object_hash(fold_id.tolist()), "rows": len(fold_id),
            "package_versions": {name: importlib.metadata.version(name) for name in config["runtime"]},
            "python": sys.version, "platform": sys.platform}
    if RECOVERY is not None:
        previous = RECOVERY["original_contract"]
        if {key: item for key, item in value.items() if key != "sources"} != {
                key: item for key, item in previous.items() if key != "sources"}:
            raise ValueError("Startup recovery changed the registered experiment")
        before, after = previous["sources"], value["sources"]
        allowed = str(Path(__file__).relative_to(MODULE))
        if set(before) != set(after) or any(before[key] != after[key] for key in before if key != allowed):
            raise ValueError("Startup recovery changed sources outside the runner")
        value["startup_recovery"] = {key: RECOVERY[key] for key in ("amendment", "amendment_sha256")}
    return value


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


def consume(claim, key, limit, path=None):
    if path is None:
        path = CLAIM
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
        consume(claim, "attempted_forward_calls", budget["max_extraction_batches"], CLAIM)
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


def fit_head(features, data, folds, evaluation, claim, max_attempts, claim_path=None):
    if claim_path is None:
        claim_path = CLAIM
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


def candidate_gate(result, config):
    gate = config["candidate_gate"]
    folds = result["fold_balanced_accuracy"]
    values = np.asarray([result["balanced_accuracy"], *folds], dtype=np.float64)
    passed = (len(folds) == config["evaluation"]["folds"] and np.isfinite(values).all()
              and np.all((values >= 0) & (values <= 1))
              and result["balanced_accuracy"] >= gate["min_pooled_balanced_accuracy"]
              and min(folds, default=-1) >= gate["min_each_fold_balanced_accuracy"])
    return {"metric_gate_passed": bool(passed), "thresholds": gate,
            "requires_completed_supervisor": True,
            "final_fit_attempts": 0, "kaggle_submissions": 0}


def require_live_supervisor(claim):
    """Accept a direct parent or the single Windows venv redirector we launched."""
    import psutil
    failure = "Execution requires the live supervisor and an unused claim"
    if (claim.get("state") != "started" or any(claim.get(key) != 0 for key in
            ("attempted_forward_calls", "completed_forward_calls", "attempted_fits", "completed_fits"))):
        raise RuntimeError(failure)
    try:
        parent = psutil.Process().parent()
        if parent is None:
            raise RuntimeError(failure)
        if parent.pid != claim.get("supervisor_pid"):
            if (sys.platform != "win32" or sys.prefix == sys.base_prefix
                    or Path(sys.executable).resolve() == Path(sys._base_executable).resolve()
                    or os.path.normcase(str(Path(psutil.Process().exe()).resolve()))
                    != os.path.normcase(str(Path(sys._base_executable).resolve()))
                    or os.path.normcase(str(Path(parent.exe()).resolve()))
                    != os.path.normcase(str(Path(sys.executable).resolve()))):
                raise RuntimeError(failure)
            parent = parent.parent()
        if (parent is None or parent.pid != claim.get("supervisor_pid")
                or parent.create_time() != claim.get("supervisor_create_time")):
            raise RuntimeError(failure)
    except psutil.Error as error:
        raise RuntimeError(failure) from error


def execute(train, config, base):
    claim = json.loads(CLAIM.read_bytes())
    require_live_supervisor(claim)
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
                               base["budget"]["max_classifier_fit_attempts"], CLAIM)
    save_array(OUTPUT / "oof.npz", "probabilities", oof, data, fold_id, claim["contract_sha256"])
    report = {"scope": config["scope"], "contract_sha256": claim["contract_sha256"],
              "metrics": metrics(oof, data, fold_id, config["evaluation"]), "gpu": gpu,
              "extraction_seconds": extraction_seconds, "fold_fit_and_predict_seconds": timings,
              "features_sha256": digest(cache), "oof_sha256": digest(OUTPUT / "oof.npz"),
              "attempted_fits": claim["attempted_fits"], "completed_fits": claim["completed_fits"],
              "final_holdout_evaluations": 0, "full_data_final_fits": 0,
              "limitations": "Descriptive development evidence only; no paired baseline, model promotion or Kaggle score."}
    if ROUND2:
        verify_preserved_claims(config)
        report["candidate_gate"] = candidate_gate(report["metrics"], config)
        report["limitations"] = "Repeatedly used development partition; no paired baseline or independent holdout evidence. Metric gate alone does not establish submission readiness or a Kaggle score."
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


def worker_command(train):
    command = [sys.executable, str(Path(__file__).resolve()), "--train", str(train), "--execute"]
    if RECOVERY is not None:
        command.append("--recover-zero-work-startup")
    if ROUND2:
        command.append("--round2")
    return command


def supervise(train, value, base):
    """External wall-clock/RSS guard works even while a native fit blocks Python."""
    import psutil
    claim = {"state": "started", "experiment_id": value["config"]["experiment_id"],
             "supervisor_pid": os.getpid(), "supervisor_create_time": psutil.Process().create_time(),
             "contract_sha256": object_hash(value),
             "attempted_forward_calls": 0, "completed_forward_calls": 0,
             "attempted_fits": 0, "completed_fits": 0, "final_holdout_evaluations": 0,
             "full_data_final_fits": 0, "started_unix_seconds": time.time()}
    if RECOVERY is not None:
        claim["startup_recovery"] = value["startup_recovery"]
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
        if RECOVERY is not None:
            target = OUTPUT / "source_snapshot" / RECOVERY_CONFIG.relative_to(MODULE)
            with target.open("xb") as handle:
                handle.write(RECOVERY_CONFIG.read_bytes())
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "TOKENIZERS_PARALLELISM": "false"}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(worker_command(train),
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
                or claim["attempted_forward_calls"] != base["budget"]["max_extraction_batches"]
                or claim["completed_forward_calls"] != claim["attempted_forward_calls"]
                or digest(OUTPUT / "summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Worker failed or returned an incomplete result; inspect retained logs")
        if ROUND2:
            verify_preserved_claims(value["config"])
        receipt = {"state": "completed", "wall_seconds": time.perf_counter() - started,
                   "sampled_peak_process_tree_rss_bytes": peak_rss, "exit_code": process.returncode,
                   "summary_sha256": claim["summary_sha256"]}
        if ROUND2:
            summary = json.loads((OUTPUT / "summary.json").read_bytes())
            receipt["candidate_development_gate_passed"] = candidate_gate(
                summary["metrics"], value["config"])["metric_gate_passed"]
            receipt["candidate_deployment_validation_complete"] = False
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
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument("--round2", action="store_true", help="Use the separately registered independent round 2")
    protocol.add_argument("--recover-zero-work-startup", action="store_true",
                        help="Use the documented one-time recovery of the exact zero-work launcher failure")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.recover_zero_work_startup:
        activate_startup_recovery()
    if args.round2:
        activate_round2()
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
