"""Read-only reconciliation of saved R8 outputs; no transform/model execution."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def main():
    config = read(ROOT / "config/windows_sinkhorn_round8_startup_20261026.json")
    folder = ROOT / "outputs/windows_sinkhorn_20261026_r8_startup1"
    summary, receipt, contract = [read(folder / name) for name in ("summary.json", "supervisor.json", "contract.json")]
    claim = read(ROOT / "outputs/continuation_claims/windows-frozen-svc-sinkhorn-dev-20261026-r8-startup-1.json")
    assert claim["state"] == receipt["state"] == "completed" and receipt["exit_code"] == 0
    assert claim["summary_sha256"] == receipt["summary_sha256"] == sha(folder / "summary.json")
    assert claim["supervisor_sha256"] == sha(folder / "supervisor.json")
    assert claim["contract_sha256"] == summary["contract_sha256"] == canonical_hash(contract)
    assert contract["config"] == config
    for name, expected in {**config["input_sha256"], **config["preserved_claims"]}.items():
        assert sha(ROOT / name) == expected, name
    for name, expected in contract["sources"].items():
        assert sha(folder / "source_snapshot" / name) == expected, name
        assert sha(ROOT / name) == expected, name
    with np.load(ROOT / config["baseline_output"] / "candidate_oof.npz", allow_pickle=False) as z:
        baseline = {key: z[key].copy() for key in z.files}
    with np.load(folder / "candidate_oof.npz", allow_pickle=False) as z:
        candidate = {key: z[key].copy() for key in z.files}
    for key in ("row_index", "ids", "groups", "truth", "labels", "fold_id"):
        np.testing.assert_array_equal(candidate[key], baseline[key])
    assert str(candidate["contract_sha256"]) == claim["contract_sha256"]
    assert sha(folder / "candidate_oof.npz") == summary["candidate_oof_sha256"]
    old_p, new_p, truth = baseline["probabilities"], candidate["probabilities"], candidate["truth"]
    assert new_p.shape == (2159, 18) and np.isfinite(new_p).all() and (new_p > 0).all()
    np.testing.assert_allclose(new_p.sum(1), 1, atol=1e-12, rtol=0)
    old, new = old_p.argmax(1), new_p.argmax(1)
    def ba(prediction, selected):
        return float(np.mean([np.mean(prediction[selected & (truth == label)] == label) for label in range(18)]))
    full = np.ones(len(truth), dtype=bool)
    for prediction, key in ((old, "baseline_metrics"), (new, "candidate_metrics")):
        assert abs(ba(prediction, full) - summary[key]["balanced_accuracy"]) < 1e-12
        confusion = np.zeros((18, 18), dtype=int)
        np.add.at(confusion, (truth, prediction), 1)
        np.testing.assert_array_equal(confusion, summary[key]["confusion_matrix"])
        for label, row in enumerate(summary[key]["per_class"]):
            assert row["label"] == candidate["labels"][label]
            assert row["support"] == int((truth == label).sum())
            assert row["correct"] == int(((truth == label) & (prediction == label)).sum())
            assert abs(row["recall"] - row["correct"] / row["support"]) < 1e-12
    gains, mass_errors = [], []
    for fold in summary["folds"]:
        selected = candidate["fold_id"] == fold["fold"]
        old_ba, new_ba = ba(old, selected), ba(new, selected)
        assert abs(old_ba - fold["baseline_ba"]) < 1e-12
        assert abs(new_ba - fold["candidate_ba"]) < 1e-12
        mass_errors.append(float(np.max(np.abs(new_p[selected].sum(0) - selected.sum() / 18))))
        assert mass_errors[-1] < 1e-10
        gains.append(new_ba - old_ba)
    compare = summary["comparison"]
    assert compare["changed_predictions"] == int((new != old).sum())
    assert compare["corrected"] == int(((new == truth) & (old != truth)).sum())
    assert compare["harmed"] == int(((new != truth) & (old == truth)).sum())
    assert abs(compare["gain"] - (ba(new, full) - ba(old, full))) < 1e-12
    groups = np.unique(candidate["groups"])
    blocks = [np.flatnonzero(candidate["groups"] == group) for group in groups]
    counts = np.array([np.bincount(truth[rows], minlength=18) for rows in blocks])
    difference = np.array([np.bincount(truth[rows], weights=(new[rows] == truth[rows]).astype(int)
                          - (old[rows] == truth[rows]).astype(int), minlength=18) for rows in blocks])
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    draws = rng.integers(len(groups), size=(config["evaluation"]["bootstrap_resamples"], len(groups)))
    samples = np.mean(difference[draws].sum(1) / counts[draws].sum(1), axis=1)
    np.testing.assert_allclose(np.quantile(samples, [.025, .975]), compare["descriptive_prompt_95ci"], atol=1e-12, rtol=0)
    checks = {"pooled": ba(new, full) >= .42, "each_fold": min(f["candidate_ba"] for f in summary["folds"]) >= .35,
              "gain": compare["gain"] >= .01, "paired_lower_bound": compare["descriptive_prompt_95ci"][0] > 0,
              "improved_folds": sum(g > 1e-12 for g in gains) >= 4}
    assert checks == summary["candidate_gate"]["checks"]
    assert all(checks.values()) == summary["candidate_gate"]["passed"] == receipt["candidate_development_gate_passed"]
    assert claim["attempted_transforms"] == claim["completed_transforms"] == summary["completed_transforms"] == 10
    for key in ("model_fit_calls", "model_prediction_calls", "new_encoder_forward_calls", "old_holdout_evaluations"):
        assert claim[key] == summary[key] == 0
    assert summary["new_kaggle_submissions"] == 0
    assert receipt["wall_seconds"] + .75 < 120 and receipt["sampled_peak_process_tree_rss_bytes"] < 2 * 2**30
    output = {"state": "verified", "summary_sha256": sha(folder / "summary.json"),
              "verifier_sha256": sha(Path(__file__)), "rows": len(truth), "groups": len(groups),
              "baseline_ba": ba(old, full), "candidate_ba": ba(new, full),
              "improved_folds": sum(g > 1e-12 for g in gains), "tied_folds": sum(abs(g) <= 1e-12 for g in gains),
              "max_class_mass_error": max(mass_errors), "gate_passed": all(checks.values()),
              "new_transforms": 0, "new_fits": 0, "new_holdout_evaluations": 0}
    with (folder / "verification.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
