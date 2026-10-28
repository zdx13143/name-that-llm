"""Reconcile saved R9 arrays and hashes; never load texts, models or the runner."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def diagnostics(groups, true_groups):
    _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    _, true_inverse, true_sizes = np.unique(true_groups, return_inverse=True, return_counts=True)
    table = np.zeros((len(true_sizes), len(sizes)), dtype=np.int64)
    np.add.at(table, (true_inverse, inverse), 1)
    tp = int((table * (table - 1) // 2).sum())
    pp = int((sizes * (sizes - 1) // 2).sum())
    ap = int((true_sizes * (true_sizes - 1) // 2).sum())
    return {"rows": len(groups), "clusters": len(sizes), "singleton_rows": int(sizes[sizes == 1].sum()),
            "eligible_rows": int(sizes[(sizes >= 2) & (sizes <= 18)].sum()), "oversize_rows": int(sizes[sizes > 18].sum()),
            "size_histogram": {str(n): int((sizes == n).sum()) for n in sorted(set(sizes.tolist()))},
            "adjusted_rand_index": float(adjusted_rand_score(true_groups, groups)), "pair_true_positives": tp,
            "inferred_pairs": pp, "true_pairs": ap, "pair_precision": tp / pp if pp else None, "pair_recall": tp / ap if ap else None}


def main():
    config = read(ROOT / "config/windows_topic_word_round9_20261028.json")
    folder = ROOT / "outputs/windows_topic_word_20261028_r9"
    summary, supervisor, contract = [read(folder / name) for name in ("summary.json", "supervisor.json", "contract.json")]
    claim = read(ROOT / "outputs/continuation_claims/windows-topic-word-dev-20261028-r9.json")
    assert claim["state"] == supervisor["state"] == "completed" and supervisor["exit_code"] == 0
    assert claim["summary_sha256"] == supervisor["summary_sha256"] == sha(folder / "summary.json")
    assert claim["supervisor_sha256"] == sha(folder / "supervisor.json")
    assert claim["contract_sha256"] == summary["contract_sha256"] == canonical_hash(contract)
    assert contract["config"] == config
    for name, expected in {**config["input_sha256"], **config["preserved_claims"]}.items():
        assert sha(ROOT / name) == expected, name
    for name, expected in contract["sources"].items():
        assert sha(ROOT / name) == sha(folder / "source_snapshot" / name) == expected, name
    baseline = arrays(ROOT / config["baseline_output"] / "candidate_oof.npz")
    truth, fold_id = baseline["truth"], baseline["fold_id"]
    assert len(truth) == len(set(baseline["ids"].tolist())) == 2159
    groups = np.unique(baseline["groups"])
    assert len(groups) == 120 and set(fold_id) == set(range(5))
    assert all(set(truth[fold_id == i]) == set(range(18)) for i in range(5))
    assert contract["fold_sha256"] == canonical_hash(fold_id.tolist())
    assert contract["row_identity_sha256"] == canonical_hash({k: baseline[k].tolist() for k in ("row_index", "ids", "groups", "truth", "labels")})
    blocks = [np.flatnonzero(baseline["groups"] == group) for group in groups]
    assert all(len(np.unique(fold_id[rows])) == 1 for rows in blocks)
    counts = np.array([np.bincount(truth[rows], minlength=18) for rows in blocks])
    draws = np.random.default_rng(config["evaluation"]["bootstrap_seed"]).multinomial(
        len(groups), np.ones(len(groups)) / len(groups), size=config["evaluation"]["bootstrap_resamples"])
    denominators = draws @ counts
    old = baseline["probabilities"].argmax(1)
    full = np.ones(len(truth), dtype=bool)
    ba = lambda pred, selected: float(np.mean([np.mean(pred[selected & (truth == label)] == label) for label in range(18)]))
    assert abs(ba(old, full) - summary["baseline_metrics"]["balanced_accuracy"]) < 1e-12
    outputs, checked, aggregate = {}, {}, {key: 0 for key in ("rows", "clusters", "singleton_rows", "eligible_rows", "oversize_rows", "pair_true_positives", "inferred_pairs", "true_pairs")}
    for name in ("candidate", "half", "singleton"):
        saved = arrays(folder / f"{name}_oof.npz")
        outputs[name] = saved
        for key in ("row_index", "ids", "groups", "truth", "labels", "fold_id"):
            np.testing.assert_array_equal(saved[key], baseline[key])
        assert str(saved["contract_sha256"]) == claim["contract_sha256"]
        assert sha(folder / f"{name}_oof.npz") == summary["oof_sha256"][name]
        p = saved["probabilities"]
        assert p.shape == (2159, 18) and p.dtype == np.float64 and np.isfinite(p).all() and (p > 0).all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-12, rtol=0)
        pred, metrics, comparison = p.argmax(1), summary["metrics"][name], summary["comparisons"][name]
        pooled = ba(pred, full)
        assert abs(pooled - metrics["balanced_accuracy"]) < 1e-12
        folds = [ba(pred, fold_id == i) for i in range(5)]
        np.testing.assert_allclose(folds, metrics["fold_balanced_accuracy"], atol=1e-12, rtol=0)
        confusion = np.zeros((18, 18), dtype=int)
        np.add.at(confusion, (truth, pred), 1)
        np.testing.assert_array_equal(confusion, metrics["confusion_matrix"])
        for label, text in enumerate(baseline["labels"]):
            assert abs(confusion[label, label] / confusion[label].sum() - metrics["per_class_recall"][str(text)]) < 1e-12
        hits = np.array([np.bincount(truth[rows], weights=(pred[rows] == truth[rows]), minlength=18) for rows in blocks])
        np.testing.assert_allclose(np.quantile(((draws @ hits) / denominators).mean(1), [.025, .975]), metrics["prompt_bootstrap_ba_95ci"], atol=1e-12, rtol=0)
        diff = np.array([np.bincount(truth[rows], weights=(pred[rows] == truth[rows]).astype(int) - (old[rows] == truth[rows]).astype(int), minlength=18) for rows in blocks])
        ci = np.quantile(((draws @ diff) / denominators).mean(1), [.025, .975])
        np.testing.assert_allclose(ci, comparison["prompt_paired_gain_95ci"], atol=1e-12, rtol=0)
        gain = pooled - ba(old, full)
        fold_gains = [folds[i] - ba(old, fold_id == i) for i in range(5)]
        np.testing.assert_allclose(fold_gains, comparison["fold_ba_gains"], atol=1e-12, rtol=0)
        assert abs(gain - comparison["pooled_ba_gain"]) < 1e-12
        assert sum(g > 1e-12 for g in fold_gains) == comparison["improved_folds"]
        checked[name] = {"ba": pooled, "gain": gain, "ci": ci.tolist(), "fold_ba": folds,
                         "improved_folds": comparison["improved_folds"], "correct": int((pred == truth).sum()),
                         "changed": int((pred != old).sum()), "corrected": int(((pred == truth) & (old != truth)).sum()),
                         "harmed": int(((pred != truth) & (old == truth)).sum())}
    assert len(summary["folds"]) == 5 and [r["fold"] for r in summary["folds"]] == list(range(5))
    for i, record in enumerate(summary["folds"]):
        path = folder / f"fold{i}"
        assert record == read(path / "receipt.json") and record["fold"] == i
        assert record["scores_sha256"] == sha(path / "scores.npz")
        assert record["model_sha256"] == sha(path / "model.joblib")
        assert record["permutation_max_abs_error"] <= 1e-7
        saved = arrays(path / "scores.npz")
        valid = np.flatnonzero(fold_id == i)
        np.testing.assert_array_equal(saved["valid_positions"], valid)
        np.testing.assert_array_equal(saved["labels"], baseline["labels"])
        assert str(saved["contract_sha256"]) == claim["contract_sha256"]
        for name in outputs:
            scores = saved[name]
            assert scores.shape == (len(valid), 18) and np.isfinite(scores).all()
            ex = np.exp(scores - scores.max(1, keepdims=True))
            np.testing.assert_allclose(ex / ex.sum(1, keepdims=True), outputs[name]["probabilities"][valid], atol=1e-12, rtol=0)
        for key, diagnostic in (("groups", "validation_group_diagnostics"), ("half_groups", "half_group_diagnostics")):
            found = diagnostics(saved[key], baseline["groups"][valid])
            assert found == record[diagnostic]
            if key == "groups":
                for field in aggregate:
                    aggregate[field] += found[field]
    c, settings = checked["candidate"], config["candidate_gate"]
    checks = {"finite_complete": True, "pooled": c["ba"] >= settings["min_pooled_balanced_accuracy"],
              "each_fold": min(c["fold_ba"]) >= settings["min_each_fold_balanced_accuracy"],
              "paired_gain": c["gain"] >= settings["min_gain"], "paired_interval": c["ci"][0] > 0,
              "improved_folds": c["improved_folds"] >= settings["min_improved_folds"],
              "half_batch_robustness": checked["half"]["gain"] >= settings["min_half_batch_gain"],
              "singleton_robustness": checked["singleton"]["gain"] >= settings["min_singleton_gain"]}
    assert checks == summary["candidate_gate"]["checks"]
    assert all(checks.values()) == summary["candidate_gate"]["metric_gate_passed"] == supervisor["candidate_development_gate_passed"]
    for key, expected in (("attempted_fits", 5), ("completed_fits", 5), ("attempted_groupings", 20), ("completed_groupings", 20)):
        assert claim[key] == summary[key] == expected
    for key in ("attempted_vectorizer_fits", "completed_vectorizer_fits", "attempted_forward_calls", "completed_forward_calls"):
        assert claim[key] == 0
    for key in ("new_supervised_vectorizer_fits", "new_encoder_forward_calls", "old_holdout_evaluations", "full_data_final_fits", "new_kaggle_submissions"):
        assert summary[key] == 0
    assert supervisor["wall_seconds"] < config["budget"]["development_timeout_seconds"]
    assert supervisor["sampled_peak_process_tree_rss_bytes"] < config["budget"]["max_host_rss_gib"] * 2**30
    aggregate["pair_precision"] = aggregate["pair_true_positives"] / aggregate["inferred_pairs"] if aggregate["inferred_pairs"] else None
    aggregate["pair_recall"] = aggregate["pair_true_positives"] / aggregate["true_pairs"]
    output = {"state": "verified", "summary_sha256": sha(folder / "summary.json"), "verifier_sha256": sha(Path(__file__)),
              "rows": len(truth), "groups": len(groups), "baseline_ba": ba(old, full), "baseline_correct": int((old == truth).sum()),
              "results": checked, "group_diagnostics": aggregate, "gate_passed": all(checks.values()),
              "preserved_claims": len(config["preserved_claims"]), "new_groupings": 0, "new_fits": 0, "new_holdout_evaluations": 0}
    with (folder / "verification.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
