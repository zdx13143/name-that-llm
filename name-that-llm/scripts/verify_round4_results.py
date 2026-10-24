"""Verify completed Round4 artifacts without raw data, model loading or new inference."""
from pathlib import Path
import hashlib
import json
import numpy as np

MODULE = Path(__file__).resolve().parents[1]
OUTPUT = MODULE / "outputs/windows_modernbert_finetune_20261025_r4"
EXPECTED_CONTRACT = "7bc6528641140e557b6c36400d9ef1be6323a2452d4fd804246162199145c774"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal_numbers(actual, expected, message):
    check(np.allclose(actual, expected, rtol=0, atol=1e-12), message)


def main():
    contract = read(OUTPUT / "contract.json")
    fingerprint = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    check(fingerprint == EXPECTED_CONTRACT, "Preregistered execution contract differs")
    config, budget = contract["config"], contract["config"]["budget"]
    claim = read(MODULE / "outputs/continuation_claims/windows-modernbert-finetune-dev-20261025-r4.json")
    summary, supervisor = read(OUTPUT / "summary.json"), read(OUTPUT / "supervisor.json")
    check(claim["state"] == supervisor["state"] == "completed" and supervisor["exit_code"] == 0,
          "Run must be complete before verification")
    check(claim["contract_sha256"] == summary["contract_sha256"] == fingerprint, "Contract chain differs")
    check(sha(OUTPUT / "summary.json") == claim["summary_sha256"] == supervisor["summary_sha256"], "Summary chain differs")
    check(sha(OUTPUT / "supervisor.json") == claim["supervisor_sha256"], "Supervisor chain differs")
    for relative, expected in config["preserved_claims"].items():
        check(sha(MODULE / relative) == expected, "Historical claim changed: " + relative)
    for relative, expected in contract["sources"].items():
        check(sha(MODULE / relative) == sha(OUTPUT / "source_snapshot" / relative) == expected,
              "Source or snapshot changed: " + relative)
    counts = {"fits": 5, "forward_calls": 9183, "backward_calls": 8641, "optimizer_steps": 1081,
              "training_microbatches": 8640, "training_optimizer_steps": 1080, "validation_forward_calls": 540,
              "preflight_train_forward_calls": 1, "preflight_eval_forward_calls": 2,
              "preflight_backward_calls": 1, "preflight_optimizer_steps": 1}
    for name, expected in counts.items():
        for prefix in ("attempted_", "completed_"):
            check(claim[prefix + name] == summary["operation_counts"][prefix + name] == expected,
                  "Incomplete count: " + prefix + name)
    check(claim["final_holdout_evaluations"] == claim["full_data_final_fits"] == 0, "Protected operation occurred")
    check(supervisor["wall_seconds"] <= budget["development_timeout_seconds"], "Wall budget exceeded")
    check(supervisor["sampled_peak_process_tree_rss_bytes"] <= budget["max_host_rss_gib"] * 2**30, "RSS exceeded")
    for kind in ("allocated", "reserved"):
        check(summary["gpu"]["peak_" + kind + "_bytes"] <= budget["max_cuda_" + kind + "_gib"] * 2**30,
              "CUDA memory exceeded")
    baseline_path = MODULE / config["baseline_output"] / "oof.npz"
    check(sha(baseline_path) == config["baseline_oof_sha256"], "Baseline OOF changed")
    check(sha(OUTPUT / "oof.npz") == summary["oof_sha256"], "Round4 OOF changed")
    with np.load(OUTPUT / "oof.npz", allow_pickle=False) as current, np.load(baseline_path, allow_pickle=False) as baseline:
        check(set(current.files) == set(baseline.files), "OOF schema differs")
        for key in ("row_index", "ids", "groups", "truth", "labels", "fold_id"):
            check(np.array_equal(current[key], baseline[key]), "OOF identity differs: " + key)
        check(str(current["contract_sha256"]) == fingerprint, "OOF contract differs")
        p, truth, fold = current["probabilities"], current["truth"], current["fold_id"]
        check(p.shape == (2159, 18) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid probabilities")
        equal_numbers(p.sum(1), np.ones(2159), "Probabilities do not sum to one")
        check(len(np.unique(current["ids"])) == 2159 and len(np.unique(current["groups"])) == 120, "Coverage differs")
        correct = (p.argmax(1) == truth).astype(int)
        delta = correct - (baseline["probabilities"].argmax(1) == truth).astype(int)
        recalls = [correct[truth == i].mean() for i in range(18)]
        ba = float(np.mean(recalls))
        fold_ba = [float(np.mean([correct[(fold == f) & (truth == c)].mean() for c in range(18)])) for f in range(5)]
        gains = [float(np.mean([delta[(fold == f) & (truth == c)].mean() for c in range(18)])) for f in range(5)]
        gain = float(np.mean([delta[truth == i].mean() for i in range(18)]))
        equal_numbers(ba, summary["metrics"]["balanced_accuracy"], "Pooled BA differs")
        equal_numbers(fold_ba, summary["metrics"]["fold_balanced_accuracy"], "Fold BA differs")
        equal_numbers(recalls, [summary["metrics"]["per_class_recall"][label] for label in current["labels"]], "Recalls differ")
        matrix = np.zeros((18, 18), dtype=int)
        np.add.at(matrix, (truth, p.argmax(1)), 1)
        check(np.array_equal(matrix, summary["metrics"]["confusion_matrix"]), "Confusion matrix differs")
        groups = np.unique(current["groups"])
        totals = np.array([[np.sum((current["groups"] == g) & (truth == c)) for c in range(18)] for g in groups])
        differences = np.array([[delta[(current["groups"] == g) & (truth == c)].sum() for c in range(18)] for g in groups])
        hits = np.array([[correct[(current["groups"] == g) & (truth == c)].sum() for c in range(18)] for g in groups])
        evaluation = config["evaluation"]
        samples = np.random.default_rng(evaluation["bootstrap_seed"]).multinomial(
            120, np.full(120, 1 / 120), size=evaluation["bootstrap_resamples"])
        denominator = samples @ totals
        check(np.all(denominator > 0), "Bootstrap missing classes")
        interval = np.quantile(((samples @ differences) / denominator).mean(1), [.025, .975])
        equal_numbers(np.quantile(((samples @ hits) / denominator).mean(1), [.025, .975]),
                      summary["metrics"]["prompt_bootstrap_ba_95ci"], "BA interval differs")
        paired = summary["paired_round3_comparison"]
        equal_numbers(gain, paired["pooled_ba_gain"], "Paired gain differs")
        equal_numbers(gains, paired["fold_ba_gains"], "Fold gains differ")
        equal_numbers(interval, paired["prompt_paired_gain_95ci"], "Paired interval differs")
        improved = sum(v > 1e-12 for v in gains)
        check(improved == paired["improved_folds"], "Improved fold count differs")
        fold_rows = [int(np.sum(fold == f)) for f in range(5)]
        check(fold_rows == [432, 432, 432, 432, 431], "Fold coverage differs")
        for g in groups:
            check(len(np.unique(fold[current["groups"] == g])) == 1, "Prompt crosses folds")
    check(len(summary["folds"]) == 5, "Five complete fold receipts required")
    for f, record in enumerate(summary["folds"]):
        check(record["fold"] == f and record["validation_rows"] == fold_rows[f], "Fold receipt identity differs")
        check(record == read(OUTPUT / f"fold{f}_receipt.json"), "Fold receipt contents differ")
        check(len(record["epochs"]) == 4 and record["optimizer_steps"] == 216, "Epoch count differs")
        for epoch in record["epochs"]:
            check(epoch["examples_seen"] == 2159 - fold_rows[f] and epoch["microbatches"] == 432
                  and epoch["optimizer_steps"] == 54 and np.isfinite(epoch["mean_training_loss"]), "Epoch coverage differs")
        check(set(record["checkpoint"]) == {"config.json", "model.safetensors"}, "Checkpoint artifacts incomplete")
        for filename, artifact in record["checkpoint"].items():
            path = OUTPUT / f"fold{f}" / filename
            check(sha(path) == artifact["sha256"] and path.stat().st_size == artifact["bytes"], "Checkpoint identity differs")
    gate = config["candidate_gate"]
    passed = bool(ba >= gate["min_pooled_balanced_accuracy"] and min(fold_ba) >= gate["min_each_fold_balanced_accuracy"]
                  and gain >= gate["min_paired_gain_over_round3"] and interval[0] > gate["paired_prompt_95ci_lower_exceeds"]
                  and improved >= gate["min_improved_folds_over_round3"])
    check(passed == summary["candidate_gate"]["metric_gate_passed"] == supervisor["candidate_development_gate_passed"], "Gate differs")
    result = {"state": "verified", "contract_sha256": fingerprint, "oof_sha256": summary["oof_sha256"],
              "pooled_ba": ba, "fold_ba": fold_ba, "paired_gain": gain, "paired_95ci": interval.tolist(),
              "improved_folds": improved, "development_gate_passed": passed, "verified_operation_counts": counts,
              "new_model_forward_calls": 0, "new_fits": 0, "raw_data_reads": 0,
              "limitations": "Recomputed saved predictions and hashes; no new model inference or independent held-out evaluation."}
    destination = MODULE / "outputs/windows_4060/round4_final_reconciliation_20261025.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
