"""One immutable five-fold supervised ModernBERT experiment. Default is readiness."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil

import windows_modernbert_dev as dev
from probe_modernbert import digest, validate_files, write_json
from windows_prompt_gaussian_round3 import paired_comparison

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_modernbert_finetune_round4_20261025.json"
CONFIG_SHA256 = "123abe15d965bcf8b17cadae6d25340b237d9cd85992121ce2d65aabbc9f2eaa"
BASE = MODULE / "config/windows_modernbert_20261023.json"
CLAIM = MODULE / "outputs/continuation_claims/windows-modernbert-finetune-dev-20261025-r4.json"
OUTPUT = MODULE / "outputs/windows_modernbert_finetune_20261025_r4"
SOURCES = (Path(__file__), Path(__file__).with_name("modernbert_finetune_training.py"),
           Path(__file__).with_name("windows_modernbert_dev.py"), Path(__file__).with_name("probe_modernbert.py"),
           Path(__file__).with_name("restore_team_split.py"), Path(__file__).with_name("windows_prompt_gaussian_round3.py"),
           Path(__file__).with_name("prompt_gaussian_head.py"), CONFIG, BASE, MODULE / "config/labels.json",
           MODULE / "requirements-windows-dev.txt", MODULE / "requirements-windows-gpu.txt")


def configuration():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("Registered Round4 configuration identity differs")
    config, base = json.loads(CONFIG.read_bytes()), json.loads(BASE.read_bytes())
    if config["experiment_id"] != "windows-modernbert-finetune-dev-20261025-r4":
        raise ValueError("Unexpected experiment")
    return config, base


def load_inputs(config, base):
    dev.verify_preserved_claims(config)
    data = dev.load_development(MODULE / "data/raw/train.csv", base["input"], MODULE / "config/labels.json")
    folds, fold_id = dev.grouped_folds(data, config["evaluation"]["folds"])
    reference = json.loads((MODULE / config["fold_reference_output"] / "contract.json").read_bytes())
    if (dev.object_hash(reference) != config["fold_reference_contract_sha256"]
            or dev.object_hash(fold_id.tolist()) != reference["fold_sha256"]
            or dev.object_hash(data["texts"].tolist()) != reference["text_sha256"]):
        raise ValueError("Original development texts/fold identity differs")
    previous = MODULE / config["baseline_output"]
    previous_contract = json.loads((previous / "contract.json").read_bytes())
    previous_claim = json.loads((MODULE / "outputs/continuation_claims/windows-prompt-gaussian-dev-20261024-r3.json").read_bytes())
    if (dev.object_hash(previous_contract) != config["baseline_contract_sha256"]
            or previous_claim["state"] != "completed"
            or previous_claim["contract_sha256"] != config["baseline_contract_sha256"]
            or digest(previous / "summary.json") != previous_claim["summary_sha256"]
            or digest(previous / "supervisor.json") != previous_claim["supervisor_sha256"]
            or digest(previous / "oof.npz") != config["baseline_oof_sha256"]):
        raise ValueError("Completed Round3 baseline identity differs")
    with np.load(previous / "oof.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"probabilities", "row_index", "ids", "groups", "truth", "labels", "fold_id", "contract_sha256"}:
            raise ValueError("Unexpected baseline OOF schema")
        for key in ("row_index", "ids", "groups", "truth", "labels"):
            if not np.array_equal(saved[key], data[key]):
                raise ValueError("Baseline row/label identity differs: " + key)
        if not np.array_equal(saved["fold_id"], fold_id) or str(saved["contract_sha256"]) != config["baseline_contract_sha256"]:
            raise ValueError("Baseline fold/contract identity differs")
        baseline = dev.probabilities(saved["probabilities"], len(fold_id), 18)
    return data, folds, fold_id, baseline


def expected_counts(folds, config):
    method, budget = config["training"], config["budget"]
    batch, accumulation, epochs = method["micro_batch_size"], method["gradient_accumulation"], method["epochs"]
    training = sum(math.ceil(len(train) / batch) * epochs for train, _ in folds)
    validation = sum(math.ceil(len(valid) / config["evaluation"]["evaluation_batch_size"]) for _, valid in folds)
    steps = sum(math.ceil(len(train) / (batch * accumulation)) * epochs for train, _ in folds)
    counts = {"training_microbatches": training, "validation_forward_calls": validation,
              "training_optimizer_steps": steps, "logical_forward_calls": training + validation + 3,
              "backward_calls": training + 1, "optimizer_steps": steps + 1}
    if any(budget["max_" + key] != number for key, number in counts.items()):
        raise ValueError("Computed operation counts differ from the preregistered budget")
    return counts


def contract(config, base, data, fold_id, files, counts):
    return {"config": config, "base_config": base, "model_files": files, "expected_counts": counts,
            "sources": {str(path.relative_to(MODULE)): digest(path) for path in SOURCES},
            "row_identity_sha256": dev.object_hash({key: data[key].tolist() for key in
                                                   ("row_index", "ids", "groups", "truth", "labels")}),
            "text_sha256": dev.object_hash(data["texts"].tolist()), "fold_sha256": dev.object_hash(fold_id.tolist()),
            "package_versions": {name: importlib.metadata.version(name) for name in config["runtime"]},
            "python": sys.version, "platform": sys.platform}


def readiness(config, base):
    blockers = []
    if sys.platform != "win32":
        blockers.append("Registered runtime is native Windows")
    if CLAIM.exists() or OUTPUT.exists():
        blockers.append("Fixed claim/output already exists; no retry or resume")
    if shutil.disk_usage(MODULE).free < config["budget"]["min_disk_free_gib"] * 2**30:
        blockers.append("Insufficient free disk for retained fold checkpoints")
    for name, expected in config["runtime"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            blockers.append(f"Runtime {name}: {actual}; expected {expected}")
    if blockers:
        return {"state": "blocked", "blockers": blockers}, None
    try:
        files = validate_files(MODULE / "models/modernbert-base" / base["revision"], base["model_files"])
        data, folds, fold_id, _ = load_inputs(config, base)
        counts = expected_counts(folds, config)
        value = contract(config, base, data, fold_id, files, counts)
    except (OSError, ValueError) as error:
        return {"state": "blocked", "blockers": [f"{type(error).__name__}: {error}"]}, None
    return {"state": "ready", "contract_sha256": dev.object_hash(value), "rows": len(fold_id), "expected_counts": counts}, value


class BudgetHooks:
    """Persist attempts before each logical operation; explicit new-round path only."""
    def __init__(self, claim, config, claim_path, resource_check):
        self.claim, self.config, self.path = claim, config, Path(claim_path)
        self.resource_check = resource_check
        b = config["budget"]
        forward = ("forward_calls", b["max_logical_forward_calls"])
        backward = ("backward_calls", b["max_backward_calls"])
        step = ("optimizer_steps", b["max_optimizer_steps"])
        self.operations = {
            "train_forward": [forward, ("training_microbatches", b["max_training_microbatches"])],
            "train_backward": [backward],
            "train_step": [step, ("training_optimizer_steps", b["max_training_optimizer_steps"])],
            "valid_forward": [forward, ("validation_forward_calls", b["max_validation_forward_calls"])],
            "probe_forward": [forward, ("preflight_train_forward_calls", b["synthetic_preflight_train_forward_calls"])],
            "probe_eval_forward": [forward, ("preflight_eval_forward_calls", b["synthetic_preflight_eval_forward_calls"])],
            "probe_backward": [backward, ("preflight_backward_calls", b["synthetic_preflight_backward_calls"])],
            "probe_step": [step, ("preflight_optimizer_steps", b["synthetic_preflight_optimizer_steps"])],
        }

    def before(self, operation):
        counters = self.operations[operation]
        if any(self.claim["attempted_" + key] >= limit for key, limit in counters):
            raise RuntimeError("Operation budget exhausted: " + operation)
        self.claim["active_operation"] = operation
        for key, _ in counters:
            self.claim["attempted_" + key] += 1
        write_json(self.path, self.claim, exclusive=False)

    def after(self, operation):
        # Synchronize CUDA before declaring a successfully completed operation.
        # A resource violation leaves the attempted count consumed and stops.
        self.resource_check()
        for key, _ in self.operations[operation]:
            if self.claim["completed_" + key] >= self.claim["attempted_" + key]:
                raise RuntimeError("Completion without a matching attempt")
            self.claim["completed_" + key] += 1
        write_json(self.path, self.claim, exclusive=False)


def fresh_claim(config, value):
    claim = {"state": "started", "experiment_id": config["experiment_id"],
             "supervisor_pid": os.getpid(), "supervisor_create_time": psutil.Process().create_time(),
             "contract_sha256": dev.object_hash(value), "started_unix_seconds": time.time(),
             "final_holdout_evaluations": 0, "full_data_final_fits": 0}
    for counter in ("fits", "forward_calls", "backward_calls", "optimizer_steps", "training_microbatches",
                    "training_optimizer_steps", "validation_forward_calls", "preflight_train_forward_calls",
                    "preflight_eval_forward_calls", "preflight_backward_calls", "preflight_optimizer_steps"):
        claim["attempted_" + counter] = 0
        claim["completed_" + counter] = 0
    return claim


def initial_model(directory, labels, method, seed, device):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from modernbert_finetune_training import seed_everything
    seed_everything(seed)
    model_config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model_config.num_labels = len(labels)
    model_config.id2label = dict(enumerate(labels))
    model_config.label2id = {label: i for i, label in enumerate(labels)}
    model_config.classifier_pooling = method["classifier_pooling"]
    model_config.classifier_dropout = method["classifier_dropout"]
    model_config.reference_compile = False
    model_config.problem_type = "single_label_classification"
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        directory, config=model_config, local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa", dtype=torch.float32, output_loading_info=True)
    if (set(loading["missing_keys"]) != {"classifier.weight", "classifier.bias"}
            or set(loading["unexpected_keys"]) - {"decoder.weight", "decoder.bias"}
            or loading.get("mismatched_keys") or loading.get("error_msgs")):
        raise ValueError("Unexpected pretrained-to-classification loading identity: " + str(loading))
    if (model.config.reference_compile is not False or model.config._attn_implementation != "sdpa"
            or any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("Model configuration or parameter training policy differs")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device), loading


def save_checkpoint(model, folder):
    """Retain and compare every exported tensor; no additional model forward."""
    import torch
    from safetensors import safe_open
    folder.mkdir(exist_ok=False)
    model.save_pretrained(folder, safe_serialization=True, max_shard_size="2GB")
    state = model.state_dict()
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(state):
            raise ValueError("Checkpoint state keys differ")
        for key, parameter in state.items():
            if not torch.equal(saved.get_tensor(key), parameter.detach().cpu()):
                raise ValueError("Checkpoint tensor changed: " + key)
    return {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(folder.iterdir())}


def metric_gate(metrics, comparison, config):
    g = config["candidate_gate"]
    folds = metrics["fold_balanced_accuracy"]
    values = [metrics["balanced_accuracy"], *folds, comparison["pooled_ba_gain"],
              *comparison["prompt_paired_gain_95ci"], *comparison["fold_ba_gains"]]
    checks = {"finite_complete": bool(len(folds) == 5 and len(comparison["fold_ba_gains"]) == 5
                                      and np.isfinite(values).all() and all(0 <= v <= 1 for v in [metrics["balanced_accuracy"], *folds])),
              "pooled": metrics["balanced_accuracy"] >= g["min_pooled_balanced_accuracy"],
              "each_fold": min(folds, default=-1) >= g["min_each_fold_balanced_accuracy"],
              "paired_gain": comparison["pooled_ba_gain"] >= g["min_paired_gain_over_round3"],
              "paired_interval": comparison["prompt_paired_gain_95ci"][0] > g["paired_prompt_95ci_lower_exceeds"],
              "improved_folds": comparison["improved_folds"] >= g["min_improved_folds_over_round3"]}
    return {"metric_gate_passed": all(checks.values()), "checks": checks,
            "requires_completed_supervisor": True, "deployment_validation_complete": False}


def execute(config, base):
    claim = json.loads(CLAIM.read_bytes())
    dev.require_live_supervisor(claim)
    data, folds, fold_id, baseline = load_inputs(config, base)
    directory = MODULE / "models/modernbert-base" / base["revision"]
    files = validate_files(directory, base["model_files"])
    value = contract(config, base, data, fold_id, files, expected_counts(folds, config))
    if dev.object_hash(value) != claim["contract_sha256"]:
        raise ValueError("Execution contract changed after readiness")
    import torch
    from transformers import AutoTokenizer
    from modernbert_finetune_training import optimizer_for, predict, train_epoch
    torch.set_num_threads(config["budget"]["threads"])
    torch.set_num_interop_threads(config["budget"]["threads"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The registered CUDA BF16 device is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < config["budget"]["min_start_free_cuda_gib"] * 2**30:
        raise MemoryError("Insufficient free GPU memory before the bounded training preflight")
    torch.cuda.reset_peak_memory_stats(device)

    def resources():
        torch.cuda.synchronize(device)
        if (torch.cuda.max_memory_allocated(device) > config["budget"]["max_cuda_allocated_gib"] * 2**30
                or torch.cuda.max_memory_reserved(device) > config["budget"]["max_cuda_reserved_gib"] * 2**30):
            raise MemoryError("Registered CUDA memory limit exceeded")

    hooks = BudgetHooks(claim, config, CLAIM, resources)
    method = config["training"]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    labels = data["labels"].tolist()

    def encode(texts):
        result = tokenizer(list(texts), max_length=method["max_length"], padding="max_length", truncation=True, return_tensors="pt")
        if set(result) != {"input_ids", "attention_mask"}:
            raise ValueError("Unexpected tokenizer tensor schema")
        return dict(result)

    # This new synthetic backward probe is budgeted independently of real folds.
    model, loading = initial_model(directory, labels, method, method["base_seed"] - 1, device)
    optimizer = optimizer_for(model, method)
    synthetic = encode([("A small synthetic passage about a quiet shore and a careful writer. " * 40),
                        ("Numbers and punctuation: one, two, three; does this artificial sentence behave? " * 40),
                        ("This is fabricated training input for a memory test, not competition data. " * 40),
                        ("The model reads four made-up texts while its optimizer allocates gradient states. " * 40)])
    probe_info, probe_steps = train_epoch(model, synthetic, torch.tensor([0, 1, 2, 3]), optimizer, method,
                                         np.arange(4), 0, 1, hooks, device, phase="probe")
    normal = predict(model, synthetic, 4, hooks, device, phase="probe_eval")
    reversed_input = {key: torch.flip(value, dims=[0]) for key, value in synthetic.items()}
    reverse = predict(model, reversed_input, 4, hooks, device, phase="probe_eval")[::-1]
    error = float(np.max(np.abs(normal.astype(float) - reverse.astype(float))))
    if probe_steps != 1 or error > 0.005 or not np.array_equal(normal.argmax(1), reverse.argmax(1)):
        raise ValueError("Synthetic fine-tuning or order-invariance preflight failed")
    write_json(OUTPUT / "synthetic_preflight.json", {"state": "completed", "new_purpose": "full-parameter backward and optimizer memory",
               "training": probe_info, "loading_info": loading, "order_probability_max_abs_error": error,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device), "competition_rows": 0})
    del model, optimizer, synthetic, reversed_input, normal, reverse
    gc.collect(); torch.cuda.empty_cache()
    print("Synthetic backward/memory preflight completed; starting registered five folds.", flush=True)

    encoded = encode(data["texts"])
    truth = torch.as_tensor(data["truth"], dtype=torch.long)
    oof = np.full((len(truth), len(labels)), np.nan, dtype=np.float64)
    records = []
    for number, (train, valid) in enumerate(folds):
        dev.consume(claim, "attempted_fits", config["budget"]["max_classifier_fit_attempts"], CLAIM)
        claim.update(active_fold=number)
        write_json(CLAIM, claim, exclusive=False)
        started = time.perf_counter()
        seed = method["base_seed"] + number
        model, loading = initial_model(directory, labels, method, seed, device)
        optimizer = optimizer_for(model, method)
        training = {key: tensor[train] for key, tensor in encoded.items()}
        training_truth = truth[train]
        total_steps = math.ceil(len(train) / (method["micro_batch_size"] * method["gradient_accumulation"])) * method["epochs"]
        generator, step, epochs = np.random.default_rng(seed), 0, []
        for epoch in range(method["epochs"]):
            claim.update(active_epoch=epoch + 1)
            write_json(CLAIM, claim, exclusive=False)
            information, step = train_epoch(model, training, training_truth, optimizer, method,
                                             generator.permutation(len(train)), step, total_steps, hooks, device)
            information["epoch"] = epoch + 1
            epochs.append(information)
            print(json.dumps({"fold": number, **information, "elapsed_seconds": time.perf_counter() - started}), flush=True)
        if step != total_steps:
            raise ValueError("Optimizer step coverage differs")
        checkpoint = save_checkpoint(model, OUTPUT / f"fold{number}")
        claim["completed_fits"] += 1
        write_json(CLAIM, claim, exclusive=False)
        validation = {key: tensor[valid] for key, tensor in encoded.items()}
        oof[valid] = dev.probabilities(predict(model, validation, config["evaluation"]["evaluation_batch_size"], hooks, device), len(valid), 18)
        records.append({"fold": number, "seed": seed, "train_rows": len(train), "validation_rows": len(valid),
                        "train_groups": len(np.unique(data["groups"][train])), "validation_groups": len(np.unique(data["groups"][valid])),
                        "trainable_parameters": sum(p.numel() for p in model.parameters()), "loading_info": loading,
                        "epochs": epochs, "optimizer_steps": step, "checkpoint": checkpoint,
                        "seconds": time.perf_counter() - started})
        write_json(OUTPUT / f"fold{number}_receipt.json", records[-1])
        print(f"Fold {number} completed at the fixed final epoch; metrics remain deferred until all folds complete.", flush=True)
        del model, optimizer, training, training_truth, validation
        gc.collect(); torch.cuda.empty_cache()

    oof = dev.probabilities(oof, len(truth), 18)
    dev.save_array(OUTPUT / "oof.npz", "probabilities", oof, data, fold_id, claim["contract_sha256"])
    result = dev.metrics(oof, data, fold_id, config["evaluation"])
    comparison = paired_comparison(oof, baseline, data, fold_id, config["evaluation"])
    summary = {"scope": config["scope"], "contract_sha256": claim["contract_sha256"], "metrics": result,
               "paired_round3_comparison": comparison, "candidate_gate": metric_gate(result, comparison, config),
               "folds": records, "oof_sha256": digest(OUTPUT / "oof.npz"),
               "gpu": {"name": torch.cuda.get_device_name(device), "total_bytes": total, "initial_free_bytes": free,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
               "operation_counts": {key: value for key, value in claim.items() if key.startswith(("attempted_", "completed_"))},
               "old_holdout_evaluations": 0, "full_data_final_fits": 0, "conditional_final_fits": 0,
               "new_kaggle_uploads": 0, "new_kaggle_saved_runs": 0, "new_kaggle_formal_submissions": 0,
               "limitations": "Repeatedly used development partition; fixed-epoch OOF is not new independent holdout evidence. GPU seeded computation is not promised bitwise reproducible across platforms."}
    dev.verify_preserved_claims(config)
    write_json(OUTPUT / "summary.json", summary)
    claim.update(state="worker_completed", summary_sha256=digest(OUTPUT / "summary.json"))
    write_json(CLAIM, claim, exclusive=False)


def require_complete(claim, config):
    b = config["budget"]
    expected = {"fits": 5, "forward_calls": b["max_logical_forward_calls"], "backward_calls": b["max_backward_calls"],
                "optimizer_steps": b["max_optimizer_steps"], "training_microbatches": b["max_training_microbatches"],
                "training_optimizer_steps": b["max_training_optimizer_steps"], "validation_forward_calls": b["max_validation_forward_calls"],
                "preflight_train_forward_calls": 1, "preflight_eval_forward_calls": 2,
                "preflight_backward_calls": 1, "preflight_optimizer_steps": 1}
    if (claim["state"] != "worker_completed" or claim["final_holdout_evaluations"] != 0
            or claim["full_data_final_fits"] != 0
            or any(claim["attempted_" + key] != number or claim["completed_" + key] != number for key, number in expected.items())):
        raise RuntimeError("Worker operation counts or state incomplete")


def supervise(config, value):
    claim = fresh_claim(config, value)
    write_json(CLAIM, claim)
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
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
               "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--execute"], env=env, stdout=stdout, stderr=stderr)
            phase = "worker"
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter() - started > config["budget"]["development_timeout_seconds"]:
                    raise TimeoutError("Registered wall-clock limit exceeded")
                try:
                    rss = (psutil.Process().memory_info().rss + monitored.memory_info().rss
                           + sum(p.memory_info().rss for p in monitored.children(recursive=True)))
                    peak_rss = max(peak_rss, rss)
                    if rss > config["budget"]["max_host_rss_gib"] * 2**30:
                        raise MemoryError("Registered process-tree RSS limit exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if process.returncode != 0:
            raise RuntimeError("Worker failed; inspect retained logs, no retry")
        require_complete(claim, config)
        if digest(OUTPUT / "summary.json") != claim["summary_sha256"]:
            raise ValueError("Summary identity differs")
        if any(digest(MODULE / relative) != expected for relative, expected in value["sources"].items()):
            raise ValueError("Execution sources changed during the run")
        dev.verify_preserved_claims(config)
        summary = json.loads((OUTPUT / "summary.json").read_bytes())
        for fold in summary["folds"]:
            for filename, artifact in fold["checkpoint"].items():
                if digest(OUTPUT / f"fold{fold['fold']}" / filename) != artifact["sha256"]:
                    raise ValueError("Retained checkpoint identity differs")
        receipt = {"state": "completed", "exit_code": 0, "wall_seconds": time.perf_counter() - started,
                   "sampled_peak_process_tree_rss_bytes": peak_rss, "summary_sha256": claim["summary_sha256"],
                   "candidate_development_gate_passed": metric_gate(summary["metrics"], summary["paired_round3_comparison"], config)["metric_gate_passed"],
                   "candidate_deployment_validation_complete": False}
        write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(OUTPUT / "supervisor.json"))
        write_json(CLAIM, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = dev.terminate_worker(process)
        except BaseException as cleanup_error:
            termination = {"cleanup_exception": type(cleanup_error).__name__}
        finally:
            claim = json.loads(CLAIM.read_bytes())
            claim.update(state="failed", phase=phase, exception_type=type(error).__name__, exception_message=str(error),
                         termination=termination, wall_seconds=time.perf_counter() - started,
                         sampled_peak_process_tree_rss_bytes=peak_rss)
            write_json(CLAIM, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config, base = configuration()
    if args.execute:
        execute(config, base)
        return 0
    state, value = readiness(config, base)
    if not args.run or state["state"] != "ready":
        print(json.dumps(state))
        return 0 if state["state"] == "ready" else 2
    print(json.dumps(supervise(config, value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
