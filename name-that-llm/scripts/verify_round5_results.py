"""Read-only reconciliation of completed Round5 artifacts; never fits or reads raw text."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix

import windows_stylegram_round5 as run


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def main():
    config, _ = run.configuration()
    run.dev.verify_preserved_claims(config)
    output = run.OUTPUT
    claim = json.loads(run.CLAIM.read_bytes())
    contract = json.loads((output / "contract.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    receipt = json.loads((output / "supervisor.json").read_bytes())
    assert claim["state"] == receipt["state"] == "completed" and receipt["exit_code"] == 0
    assert claim["contract_sha256"] == summary["contract_sha256"] == run.dev.object_hash(contract)
    assert config == contract["config"]
    assert run.digest(output / "summary.json") == claim["summary_sha256"] == receipt["summary_sha256"]
    assert run.digest(output / "supervisor.json") == claim["supervisor_sha256"]
    assert claim["attempted_fits"] == claim["completed_fits"] == summary["attempted_fits"] == summary["completed_fits"] == 5
    assert claim["attempted_forward_calls"] == claim["completed_forward_calls"] == summary["new_encoder_forward_calls"] == 0
    assert receipt["wall_seconds"] <= config["budget"]["development_timeout_seconds"]
    assert receipt["sampled_peak_process_tree_rss_bytes"] <= config["budget"]["max_host_rss_gib"] * 2**30
    for relative, expected in contract["sources"].items():
        assert run.digest(run.MODULE / relative) == run.digest(output / "source_snapshot" / relative) == expected
    for key, filename in (("oof_sha256", "oof.npz"), ("style_oof_sha256", "style_oof.npz")):
        assert run.digest(output / filename) == summary[key]
    baseline_path = run.MODULE / config["baseline_output"] / "oof.npz"
    assert run.digest(baseline_path) == config["baseline_oof_sha256"]
    arrays = []
    for path in (output / "oof.npz", output / "style_oof.npz", baseline_path):
        with np.load(path, allow_pickle=False) as artifact:
            arrays.append({key: artifact[key].copy() for key in artifact.files})
    candidate, style, baseline = arrays
    identity = ("row_index", "ids", "groups", "truth", "labels", "fold_id")
    for other in arrays[1:]:
        for key in identity:
            np.testing.assert_array_equal(candidate[key], other[key])
    assert str(candidate["contract_sha256"]) == str(style["contract_sha256"]) == claim["contract_sha256"]
    assert str(baseline["contract_sha256"]) == config["baseline_contract_sha256"]
    assert len(candidate["truth"]) == 2159 and len(np.unique(candidate["groups"])) == 120
    assert len(np.unique(candidate["ids"])) == len(np.unique(candidate["row_index"])) == 2159
    assert len(candidate["labels"]) == 18
    assert run.dev.object_hash({key:candidate[key].tolist() for key in identity if key != "fold_id"}) == contract["row_identity_sha256"]
    assert run.dev.object_hash(candidate["fold_id"].tolist()) == contract["fold_sha256"]
    truth, fold = candidate["truth"], candidate["fold_id"]
    assert set(fold) == set(range(5))
    for group in np.unique(candidate["groups"]):
        assert len(set(fold[candidate["groups"] == group])) == 1
    for item in arrays:
        p = item["probabilities"]
        assert p.shape == (2159,18) and np.isfinite(p).all() and np.all((0 <= p) & (p <= 1))
        close(p.sum(1), np.ones(2159))
    weight = config["method"]["style_weight"]
    close(candidate["probabilities"], weight*style["probabilities"]+(1-weight)*baseline["probabilities"])
    predictions = [item["probabilities"].argmax(1) for item in arrays]
    for prediction, report in zip(predictions[:2], (summary["metrics"], summary["style_control_metrics"])):
        close(balanced_accuracy_score(truth, prediction), report["balanced_accuracy"])
        close([balanced_accuracy_score(truth[fold == i], prediction[fold == i]) for i in range(5)], report["fold_balanced_accuracy"])
        close(recall_score(truth, prediction, labels=np.arange(18), average=None), [report["per_class_recall"][name] for name in candidate["labels"]])
        np.testing.assert_array_equal(confusion_matrix(truth,prediction,labels=np.arange(18)), report["confusion_matrix"])
    _, group_inverse = np.unique(candidate["groups"], return_inverse=True)
    counts = np.zeros((120,18), dtype=np.int64)
    np.add.at(counts, (group_inverse, truth), 1)
    draws = np.random.default_rng(config["evaluation"]["bootstrap_seed"]).multinomial(120,np.full(120,1/120),size=2000)
    denominators = draws @ counts
    for prediction, report in ((predictions[0],summary["metrics"]),(predictions[1],summary["style_control_metrics"])):
        hits = np.zeros_like(counts)
        np.add.at(hits,(group_inverse,truth),prediction == truth)
        close(np.quantile(((draws@hits)/denominators).mean(1),[.025,.975]), report["prompt_bootstrap_ba_95ci"])
    for reference, report in ((predictions[2],summary["paired_round3_comparison"]),(predictions[1],summary["paired_style_control_comparison"])):
        diff = (predictions[0] == truth).astype(int) - (reference == truth)
        close(np.mean([diff[truth==c].mean() for c in range(18)]), report["pooled_ba_gain"])
        gains = [np.mean([diff[(fold==i)&(truth==c)].mean() for c in range(18)]) for i in range(5)]
        close(gains, report["fold_ba_gains"])
        assert sum(v > 1e-12 for v in gains) == report["improved_folds"]
        hits = np.zeros_like(counts)
        np.add.at(hits,(group_inverse,truth),diff)
        close(np.quantile(((draws@hits)/denominators).mean(1),[.025,.975]),report["prompt_paired_gain_95ci"])
    assert len(summary["folds"]) == 5
    for i, record in enumerate(summary["folds"]):
        assert record["fold"] == i
        assert run.digest(output / f"style_fold{i}.joblib") == record["model_sha256"]
        assert max(record["inference_invariance_max_abs_error"],record["export_max_abs_error"]) <= 1e-10
    gate = run.metric_gate(summary["metrics"],summary["paired_round3_comparison"],summary["paired_style_control_comparison"],config)
    assert gate == summary["candidate_gate"] and receipt["candidate_development_gate_passed"] == gate["metric_gate_passed"]
    result = {"state":"verified", "rows":2159, "groups":120, "folds":5,
              "balanced_accuracy":summary["metrics"]["balanced_accuracy"],
              "style_control_balanced_accuracy":summary["style_control_metrics"]["balanced_accuracy"],
              "development_gate_passed":gate["metric_gate_passed"],
              "summary_sha256":run.digest(output / "summary.json"), "supervisor_sha256":run.digest(output / "supervisor.json")}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
