"""Independently reconcile the completed paired experiment; never fit/read raw texts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

MODULE = Path(__file__).resolve().parents[1]
FOLDER = MODULE / "outputs/windows_external_weighting_20261026_r7_startup1"
CLAIM = MODULE / "outputs/continuation_claims/windows-external-weighting-dev-20261026-r7-startup-1.json"
IDENTITY = ("row_index", "ids", "groups", "truth", "labels", "fold_id")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def archive(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in handle.files}


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def check():
    contract, claim = read(FOLDER / "contract.json"), read(CLAIM)
    summary, receipt = read(FOLDER / "summary.json"), read(FOLDER / "supervisor.json")
    config = contract["config"]
    assert config == read(MODULE / "config/windows_external_round7_startup_20261026.json")
    assert object_hash(contract) == claim["contract_sha256"] == summary["contract_sha256"]
    assert claim["state"] == receipt["state"] == "completed" and receipt["exit_code"] == 0
    assert digest(FOLDER / "summary.json") == claim["summary_sha256"] == receipt["summary_sha256"]
    assert digest(FOLDER / "supervisor.json") == claim["supervisor_sha256"]
    for name, count in (("fits", 10), ("vectorizer_fits", 20)):
        assert claim["attempted_" + name] == claim["completed_" + name] == count
        assert summary["attempted_" + name] == summary["completed_" + name] == count
    for name in ("attempted_forward_calls", "completed_forward_calls", "old_holdout_evaluations", "full_data_final_fits"):
        assert claim[name] == 0
    for name in ("new_encoder_forward_calls", "conditional_final_fits", "old_holdout_evaluations", "new_kaggle_submissions"):
        assert summary[name] == 0
    assert receipt["wall_seconds"] <= config["budget"]["development_timeout_seconds"]
    assert receipt["sampled_peak_process_tree_rss_bytes"] <= config["budget"]["max_host_rss_gib"] * 2**30
    for path, expected in config["preserved_claims"].items():
        assert digest(MODULE / path) == expected
    for path, expected in contract["sources"].items():
        assert digest(MODULE / path) == digest(FOLDER / "source_snapshot" / path) == expected
    assert digest(MODULE / config["pool_path"]) == config["pool_sha256"]
    assert digest(MODULE / config["pool_receipt_path"]) == contract["pool_receipt_sha256"]
    all_oof = {name: archive(FOLDER / f"{name}_oof.npz") for name in ("candidate", "control")}
    for name in all_oof:
        assert digest(FOLDER / f"{name}_oof.npz") == summary[f"{name}_oof_sha256"]
        assert str(all_oof[name]["contract_sha256"]) == claim["contract_sha256"]
    prior = MODULE / config["round6_output"] / "oof.npz"
    assert digest(prior) == config["round6_oof_sha256"]
    all_oof["round6"] = archive(prior)
    candidate = all_oof["candidate"]
    for item in all_oof.values():
        for key in IDENTITY:
            np.testing.assert_array_equal(item[key], candidate[key])
        p = item["probabilities"]
        assert p.shape == (2159, 18) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        close(p.sum(1), np.ones(2159))
    assert len(np.unique(candidate["ids"])) == len(np.unique(candidate["row_index"])) == 2159
    assert len(np.unique(candidate["groups"])) == 120 and len(candidate["labels"]) == 18
    assert object_hash({k: candidate[k].tolist() for k in IDENTITY if k != "fold_id"}) == contract["row_identity_sha256"]
    assert object_hash(candidate["fold_id"].tolist()) == contract["fold_sha256"]
    truth, groups, folds = (candidate[k] for k in ("truth", "groups", "fold_id"))
    assert len(summary["folds"]) == 5 and set(folds) == set(range(5))
    for number, record in enumerate(summary["folds"]):
        train, valid = np.flatnonzero(folds != number), np.flatnonzero(folds == number)
        assert record["fold"] == number and record["official_train_rows"] == len(train) and record["valid_rows"] == len(valid)
        assert not set(groups[train]) & set(groups[valid])
        selection = contract["fold_external_selections"][number]
        assert record["external_rows"] == selection["rows"]
        assert record["external_ids_sha256"] == selection["ids_sha256"]
        assert record["punctuation_present"] == selection["punctuation_present"]
        folder = FOLDER / f"fold{number}"
        assert read(folder / "receipt.json") == record
        assert digest(folder / "views.joblib") == record["views_sha256"]
        assert sum(record["vocabulary_sizes"].values()) == record["features"]
        for name in ("control", "candidate"):
            model = record["models"][name]
            assert model["weight"] == config["method"][name + "_official_weight"]
            assert 0 <= model["inference_invariance_max_abs_error"] <= 1e-7
            assert 0 < model["iterations"] < config["method"]["classifier"]["max_iter"]
            assert digest(folder / f"{name}.joblib") == model["model_sha256"]
            assert digest(folder / f"{name}_scores.npz") == model["scores_sha256"]
            saved = archive(folder / f"{name}_scores.npz")
            np.testing.assert_array_equal(saved["valid_positions"], valid)
            np.testing.assert_array_equal(saved["labels"], candidate["labels"])
            assert str(saved["contract_sha256"]) == claim["contract_sha256"]
            close(softmax(saved["scores"], axis=1), all_oof[name]["probabilities"][valid])
    _, gi = np.unique(groups, return_inverse=True)
    counts = np.zeros((120, 18), dtype=np.int64)
    np.add.at(counts, (gi, truth), 1)
    draws = np.random.default_rng(config["evaluation"]["bootstrap_seed"]).multinomial(
        120, np.full(120, 1 / 120), size=config["evaluation"]["bootstrap_resamples"])
    den = draws @ counts
    assert (den > 0).all()
    computed = {}
    for name in ("candidate", "control"):
        prediction = all_oof[name]["probabilities"].argmax(1)
        m = summary["metrics" if name == "candidate" else "pool_control_metrics"]
        computed[name] = balanced_accuracy_score(truth, prediction)
        close(computed[name], m["balanced_accuracy"])
        close([balanced_accuracy_score(truth[folds == n], prediction[folds == n]) for n in range(5)], m["fold_balanced_accuracy"])
        close(recall_score(truth, prediction, labels=np.arange(18), average=None), [m["per_class_recall"][label] for label in candidate["labels"]])
        np.testing.assert_array_equal(confusion_matrix(truth, prediction, labels=np.arange(18)), m["confusion_matrix"])
        hits = np.zeros_like(counts)
        np.add.at(hits, (gi, truth), prediction == truth)
        close(np.quantile(((draws @ hits) / den).mean(1), [.025, .975]), m["prompt_bootstrap_ba_95ci"])
    gains = {}
    for name, key in (("control", "paired_pool_control_comparison"), ("round6", "paired_round6_comparison")):
        difference = (candidate["probabilities"].argmax(1) == truth).astype(int) - (all_oof[name]["probabilities"].argmax(1) == truth)
        reported = summary[key]
        gains[name] = np.mean([difference[truth == label].mean() for label in range(18)])
        close(gains[name], reported["pooled_ba_gain"])
        per_fold = [np.mean([difference[(folds == f) & (truth == c)].mean() for c in range(18)]) for f in range(5)]
        close(per_fold, reported["fold_ba_gains"])
        assert sum(v > 1e-12 for v in per_fold) == reported["improved_folds"]
        hits = np.zeros_like(counts)
        np.add.at(hits, (gi, truth), difference)
        close(np.quantile(((draws @ hits) / den).mean(1), [.025, .975]), reported["prompt_paired_gain_95ci"])
    metric, comparison, threshold = summary["metrics"], summary["paired_pool_control_comparison"], config["candidate_gate"]
    checks = {"finite_complete": True,
              "pooled": computed["candidate"] >= threshold["min_pooled_balanced_accuracy"],
              "each_fold": min(metric["fold_balanced_accuracy"]) >= threshold["min_each_fold_balanced_accuracy"],
              "gain_over_pool_control": gains["control"] >= threshold["min_gain_over_pool_control"],
              "paired_interval": comparison["prompt_paired_gain_95ci"][0] > threshold["paired_prompt_95ci_lower_exceeds"],
              "improved_folds": comparison["improved_folds"] >= threshold["min_improved_folds"],
              "gain_over_round6": gains["round6"] >= threshold["min_gain_over_round6"]}
    assert checks == summary["candidate_gate"]["checks"]
    passed = bool(all(checks.values()))
    assert passed == summary["candidate_gate"]["metric_gate_passed"] == receipt["candidate_development_gate_passed"]
    return {"state": "verified", "rows": 2159, "groups": 120, "folds": 5,
            "classifier_fits": 10, "vectorizer_fits": 20, "new_encoder_forward_calls": 0,
            "old_holdout_evaluations": 0, "balanced_accuracy": float(computed["candidate"]),
            "control_balanced_accuracy": float(computed["control"]), "development_gate_passed": passed,
            "checks": {k: bool(v) for k, v in checks.items()},
            "summary_sha256": digest(FOLDER / "summary.json"), "supervisor_sha256": digest(FOLDER / "supervisor.json"),
            "verifier_sha256": digest(Path(__file__)), "scope": "Saved OOF/scores and artifact identity only; no raw texts, new fit or independent holdout."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = check()
    if args.write:
        with (FOLDER / "verification.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result))
