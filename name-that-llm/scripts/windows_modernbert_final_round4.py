"""Conditional, single final development-only fit; requires the completed Round4 gate."""
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

import windows_modernbert_finetune_round4 as cv
from probe_modernbert import digest, write_json

MODULE = cv.MODULE
CLAIM = MODULE / "outputs/continuation_claims/windows-modernbert-final-dev-20261025-r4.json"
OUTPUT = MODULE / "outputs/windows_modernbert_final_20261025_r4"
RECONCILIATION = MODULE / "outputs/windows_4060/round4_final_reconciliation_20261025.json"
SOURCES = (*cv.SOURCES, Path(__file__), Path(__file__).with_name("verify_round4_results.py"))
COUNTS = {"fits": 1, "forward_calls": 2164, "backward_calls": 2160, "optimizer_steps": 272,
          "training_microbatches": 2160, "training_optimizer_steps": 272, "validation_forward_calls": 4,
          "preflight_train_forward_calls": 0, "preflight_eval_forward_calls": 0,
          "preflight_backward_calls": 0, "preflight_optimizer_steps": 0}


def inputs_and_contract():
    config, base = cv.configuration()
    claim = json.loads(cv.CLAIM.read_bytes())
    supervisor = json.loads((cv.OUTPUT / "supervisor.json").read_bytes())
    summary = json.loads((cv.OUTPUT / "summary.json").read_bytes())
    verified = json.loads(RECONCILIATION.read_bytes())
    if (claim["state"] != "completed" or supervisor["state"] != "completed" or supervisor["exit_code"] != 0
            or not supervisor["candidate_development_gate_passed"]
            or not cv.metric_gate(summary["metrics"], summary["paired_round3_comparison"], config)["metric_gate_passed"]
            or verified["state"] != "verified" or not verified["development_gate_passed"]
            or claim["contract_sha256"] != verified["contract_sha256"]
            or digest(cv.OUTPUT / "supervisor.json") != claim["supervisor_sha256"]
            or digest(cv.OUTPUT / "summary.json") != claim["summary_sha256"]
            or digest(cv.OUTPUT / "oof.npz") != verified["oof_sha256"]):
        raise ValueError("Completed, independently reconciled Round4 development gate required")
    previous_contract = json.loads((cv.OUTPUT / "contract.json").read_bytes())
    if cv.dev.object_hash(previous_contract) != claim["contract_sha256"]:
        raise ValueError("Round4 contract identity differs")
    for relative, expected in previous_contract["sources"].items():
        if digest(MODULE / relative) != expected:
            raise ValueError("Round4 training recipe source changed: " + relative)
    for package, expected in config["runtime"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("Registered runtime changed: " + package)
    data, _, _, _ = cv.load_inputs(config, base)
    conditional = config["conditional_deployment_budget"]
    if (len(data["truth"]) != conditional["final_training_rows"]
            or len(np.unique(data["groups"])) != conditional["final_training_groups"]
            or math.ceil(len(data["truth"]) / 4) * 4 != conditional["max_final_training_microbatches"]
            or math.ceil(len(data["truth"]) / 32) * 4 != conditional["max_final_optimizer_steps"]
            or conditional["max_final_development_finetune_attempts"] != 1
            or conditional["max_deployment_synthetic_forward_calls"] != 4):
        raise ValueError("Final training counts differ from the preregistered conditional budget")
    directory = MODULE / "models/modernbert-base" / base["revision"]
    files = cv.validate_files(directory, base["model_files"])
    config["experiment_id"] = "windows-modernbert-final-dev-20261025-r4"
    config["budget"].update(max_classifier_fit_attempts=1, max_training_microbatches=2160,
                            max_training_optimizer_steps=272, max_validation_forward_calls=4,
                            max_logical_forward_calls=2164, max_backward_calls=2160, max_optimizer_steps=272,
                            synthetic_preflight_train_forward_calls=0, synthetic_preflight_eval_forward_calls=0,
                            synthetic_preflight_backward_calls=0, synthetic_preflight_optimizer_steps=0)
    value = {"config": config, "base_config": base, "expected_counts": COUNTS,
             "final_seed": config["training"]["base_seed"] + 5,
             "previous_contract_sha256": claim["contract_sha256"], "previous_claim_sha256": digest(cv.CLAIM),
             "reconciliation_sha256": digest(RECONCILIATION), "model_files": files,
             "deployment_synthetic_probability_tolerance": 0.005,
             "deployment_synthetic_argmax_must_match": True,
             "sources": {str(p.relative_to(MODULE)): digest(p) for p in SOURCES},
             "row_identity_sha256": cv.dev.object_hash({k: data[k].tolist() for k in
                                                       ("row_index", "ids", "groups", "truth", "labels")}),
             "text_sha256": cv.dev.object_hash(data["texts"].tolist())}
    return config, base, data, value


def execute():
    claim = json.loads(CLAIM.read_bytes())
    cv.dev.require_live_supervisor(claim)
    config, base, data, value = inputs_and_contract()
    if cv.dev.object_hash(value) != claim["contract_sha256"]:
        raise ValueError("Final execution contract changed")
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from modernbert_finetune_training import optimizer_for, predict, train_epoch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Registered CUDA BF16 device unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < config["budget"]["min_start_free_cuda_gib"] * 2**30:
        raise MemoryError("Insufficient free GPU memory")
    torch.cuda.reset_peak_memory_stats(device)

    def resources():
        torch.cuda.synchronize(device)
        if (torch.cuda.max_memory_allocated(device) > config["budget"]["max_cuda_allocated_gib"] * 2**30
                or torch.cuda.max_memory_reserved(device) > config["budget"]["max_cuda_reserved_gib"] * 2**30):
            raise MemoryError("Registered CUDA memory budget exceeded")

    hooks = cv.BudgetHooks(claim, config, CLAIM, resources)
    method, labels = config["training"], data["labels"].tolist()
    directory = MODULE / "models/modernbert-base" / base["revision"]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"

    def encode(texts):
        result = dict(tokenizer(list(texts), max_length=256, padding="max_length", truncation=True, return_tensors="pt"))
        if set(result) != {"input_ids", "attention_mask"}:
            raise ValueError("Tokenizer tensor schema differs")
        return result

    encoded, truth = encode(data["texts"]), torch.as_tensor(data["truth"], dtype=torch.long)
    cv.dev.consume(claim, "attempted_fits", 1, CLAIM)
    model, loading = cv.initial_model(directory, labels, method, value["final_seed"], device)
    optimizer = optimizer_for(model, method)
    generator, step, epochs = np.random.default_rng(value["final_seed"]), 0, []
    for epoch in range(4):
        claim.update(active_epoch=epoch + 1)
        write_json(CLAIM, claim, exclusive=False)
        information, step = train_epoch(model, encoded, truth, optimizer, method,
                                         generator.permutation(len(truth)), step, 272, hooks, device)
        information["epoch"] = epoch + 1
        epochs.append(information)
        print(json.dumps(information), flush=True)
    if step != 272:
        raise ValueError("Final training step coverage differs")
    checkpoint = cv.save_checkpoint(model, OUTPUT / "model")
    tokenizer.save_pretrained(OUTPUT / "model")
    claim["completed_fits"] += 1
    write_json(CLAIM, claim, exclusive=False)
    del optimizer, encoded, truth
    gc.collect()
    synthetic = encode([
        "A fabricated letter describes a calm morning beside the sea. " * 40,
        "First, count the imaginary objects. Second, explain the invented pattern. " * 40,
        "Could a hypothetical narrator prefer short sentences? This example supplies no real answer. " * 40,
        "This artificial paragraph tests exported inference: commas, brackets (like these), and numbers 123. " * 40])
    reversed_input = {key: torch.flip(tensor, [0]) for key, tensor in synthetic.items()}
    gpu = predict(model, synthetic, 4, hooks, device)
    gpu_reverse = predict(model, reversed_input, 4, hooks, device)[::-1]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    exported_config = AutoConfig.from_pretrained(OUTPUT / "model", local_files_only=True, trust_remote_code=False)
    exported_config.reference_compile = False
    cpu_model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT / "model", config=exported_config, local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa", dtype=torch.float32)
    cpu = predict(cpu_model, synthetic, 4, hooks, torch.device("cpu"))
    cpu_reverse = predict(cpu_model, reversed_input, 4, hooks, torch.device("cpu"))[::-1]
    errors = {"gpu_order": float(np.max(np.abs(gpu - gpu_reverse))),
              "cpu_order": float(np.max(np.abs(cpu - cpu_reverse))),
              "cpu_vs_gpu": float(np.max(np.abs(cpu - gpu)))}
    if (max(errors.values()) > value["deployment_synthetic_probability_tolerance"]
            or any(not np.array_equal(gpu.argmax(1), p.argmax(1)) for p in (gpu_reverse, cpu, cpu_reverse))):
        raise ValueError("Exported synthetic CPU/GPU equivalence failed: " + str(errors))
    artifacts = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted((OUTPUT / "model").iterdir())}
    for filename, expected in checkpoint.items():
        if artifacts[filename] != expected:
            raise ValueError("Checkpoint changed during tokenizer export")
    for counter, expected in COUNTS.items():
        if claim["attempted_" + counter] != expected or claim["completed_" + counter] != expected:
            raise ValueError("Incomplete final operation count: " + counter)
    cv.dev.verify_preserved_claims(config)
    report = {"state": "worker_completed", "contract_sha256": claim["contract_sha256"],
              "training_rows": 2159, "training_prompt_groups": 120, "seed": value["final_seed"],
              "epochs": epochs, "loading_info": loading, "artifacts": artifacts, "labels": labels,
              "synthetic_equivalence_errors": errors, "synthetic_competition_rows": 0,
              "operation_counts": {k: v for k, v in claim.items() if k.startswith(("attempted_", "completed_"))},
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "old_holdout_evaluations": 0, "full_data_final_fits": 0,
              "kaggle_uploads": 0, "kaggle_saved_runs": 0, "kaggle_formal_submissions": 0,
              "linux_offline_runtime_validation_complete": False}
    write_json(OUTPUT / "summary.json", report)
    claim.update(state="worker_completed", summary_sha256=digest(OUTPUT / "summary.json"))
    write_json(CLAIM, claim, exclusive=False)


def supervise(config, value):
    claim = cv.fresh_claim(config, value)
    write_json(CLAIM, claim)
    process, started, peak_rss = None, time.perf_counter(), 0
    try:
        OUTPUT.mkdir(exist_ok=False)
        write_json(OUTPUT / "contract.json", value)
        for source in SOURCES:
            target = OUTPUT / "source_snapshot" / source.relative_to(MODULE)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
               "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--execute"],
                                       env=env, stdout=stdout, stderr=stderr)
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter() - started > 3600:
                    raise TimeoutError("Final stage exceeded 3600 seconds")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss
                    rss += sum(p.memory_info().rss for p in monitored.children(recursive=True))
                    peak_rss = max(peak_rss, rss)
                    if rss > 12 * 2**30:
                        raise MemoryError("Final stage exceeded 12 GiB process-tree RSS")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.2)
        claim = json.loads(CLAIM.read_bytes())
        if process.returncode != 0 or claim["state"] != "worker_completed":
            raise RuntimeError("Conditional final worker failed; no retry")
        if (claim["final_holdout_evaluations"] != 0 or claim["full_data_final_fits"] != 0
                or any(claim[prefix + name] != expected for name, expected in COUNTS.items()
                       for prefix in ("attempted_", "completed_"))):
            raise ValueError("Final claim count reconciliation failed")
        if digest(OUTPUT / "summary.json") != claim["summary_sha256"]:
            raise ValueError("Final summary changed")
        report = json.loads((OUTPUT / "summary.json").read_bytes())
        for filename, artifact in report["artifacts"].items():
            if digest(OUTPUT / "model" / filename) != artifact["sha256"]:
                raise ValueError("Final artifact changed")
        for relative, expected in value["sources"].items():
            if digest(MODULE / relative) != expected:
                raise ValueError("Final stage source changed")
        if digest(cv.CLAIM) != value["previous_claim_sha256"]:
            raise ValueError("Completed Round4 claim changed")
        cv.dev.verify_preserved_claims(config)
        receipt = {"state": "completed", "exit_code": 0, "wall_seconds": time.perf_counter() - started,
                   "sampled_peak_process_tree_rss_bytes": peak_rss, "summary_sha256": claim["summary_sha256"],
                   "windows_synthetic_cpu_export_validation_complete": True,
                   "linux_offline_runtime_validation_complete": False, "kaggle_formal_submissions": 0}
        write_json(OUTPUT / "supervisor.json", receipt)
        claim.update(state="completed", supervisor_sha256=digest(OUTPUT / "supervisor.json"))
        write_json(CLAIM, claim, exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = cv.dev.terminate_worker(process)
        except BaseException as cleanup_error:
            termination = {"cleanup_exception": type(cleanup_error).__name__}
        finally:
            claim = json.loads(CLAIM.read_bytes())
            claim.update(state="failed", exception_type=type(error).__name__, exception_message=str(error),
                         termination=termination, wall_seconds=time.perf_counter() - started)
            write_json(CLAIM, claim, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        execute()
        return 0
    if CLAIM.exists() or OUTPUT.exists():
        print(json.dumps({"state": "blocked", "reason": "Final claim/output exists; no retry or resume"}))
        return 2
    try:
        if sys.platform != "win32" or shutil.disk_usage(MODULE).free < 10 * 2**30:
            raise RuntimeError("Native Windows and at least 10 GiB free disk required")
        config, _, _, value = inputs_and_contract()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"state": "blocked", "reason": str(error)}))
        return 2
    if not args.run:
        print(json.dumps({"state": "ready", "contract_sha256": cv.dev.object_hash(value), "counts": COUNTS}))
        return 0
    print(json.dumps(supervise(config, value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
