"""Independently reconcile saved R10 arrays; no model/text loading or runner calls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/windows_margin_20261028_r10"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def check():
    contract, summary, supervisor = [read(OUT / n) for n in ("contract.json", "summary.json", "supervisor.json")]
    c = contract["config"]
    claim = read(ROOT / "outputs/continuation_claims/windows-margin-center-dev-20261028-r10.json")
    assert claim["state"] == supervisor["state"] == "completed" and supervisor["exit_code"] == 0
    assert sha(OUT / "summary.json") == claim["summary_sha256"] == supervisor["summary_sha256"]
    assert sha(OUT / "supervisor.json") == claim["supervisor_sha256"]
    for key, value in {"attempted_fits":0,"completed_fits":0,"attempted_transforms":20,"completed_transforms":20,
                       "attempted_vectorizer_fits":0,"completed_vectorizer_fits":0,"attempted_forward_calls":0,
                       "completed_forward_calls":0,"old_holdout_evaluations":0}.items():
        assert claim[key] == supervisor[key] == value
    for path, expected in {**c["input_sha256"], **c["preserved_claims"], **contract["sources"]}.items():
        assert sha(ROOT / path) == expected, path
    for path, expected in contract["sources"].items():
        assert sha(OUT / "source_snapshot" / path) == expected
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == claim["contract_sha256"] == summary["contract_sha256"]
    def load(p):
        with np.load(p, allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}
    data = load(OUT / "candidate_oof.npz")
    identity = ("row_index", "ids", "groups", "truth", "labels", "fold_id")
    assert len(data["ids"]) == 2159 and len(np.unique(data["groups"])) == 120
    for i in range(5):
        assert set(data["truth"][data["fold_id"] == i]) == set(range(18))
    assert len(summary["folds"]) == 5 and [f["fold"] for f in summary["folds"]] == list(range(5))
    for record in summary["folds"]:
        i = record["fold"]
        rows = np.flatnonzero(data["fold_id"] == i)
        assert sha(OUT / f"fold{i}_scores.npz") == record["scores_sha256"]
        saved, original = load(OUT / f"fold{i}_scores.npz"), load(ROOT / c["baseline_output"] / f"fold{i}/scores.npz")
        np.testing.assert_array_equal(rows, saved["valid_positions"])
        for n, grouping in [("candidate", original["groups"]), ("half", original["half_groups"]), ("singleton", np.arange(len(rows)))]:
            expected = original[n].copy()
            # Independent arithmetic check of already-saved outputs, never a new candidate.
            for g in set(grouping.tolist()):
                members = np.flatnonzero(grouping == g)
                if 2 <= len(members) <= 18:
                    expected[members] = original[n][members] - .5 * original[n][members].sum(0) / len(members)
            np.testing.assert_allclose(saved[n], expected, atol=1e-12, rtol=0)
            p = load(OUT / f"{n}_oof.npz")["probabilities"]
            np.testing.assert_allclose(softmax(saved[n], axis=1), p[rows], atol=1e-12, rtol=0)
    groups, gi = np.unique(data["groups"], return_inverse=True)
    truth, labels, fid = data["truth"], data["labels"], data["fold_id"]
    counts = np.zeros((120,18), np.int64)
    np.add.at(counts, (gi,truth), 1)
    draws = np.random.default_rng(20261028).multinomial(120, np.full(120, 1/120), size=2000)
    results = {}
    for name in ("candidate", "half", "singleton"):
        saved, baseline = load(OUT / f"{name}_oof.npz"), load(ROOT / c["baseline_output"] / f"{name}_oof.npz")
        assert sha(OUT / f"{name}_oof.npz") == summary["oof_sha256"][name]
        for k in identity:
            np.testing.assert_array_equal(saved[k], data[k]); np.testing.assert_array_equal(baseline[k], data[k])
        p = saved["probabilities"]
        assert p.shape == (2159,18) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
        pred, before = p.argmax(1), baseline["probabilities"].argmax(1)
        ba = float(balanced_accuracy_score(truth,pred))
        fold_ba = [float(balanced_accuracy_score(truth[fid==i],pred[fid==i])) for i in range(5)]
        m = summary["metrics"][name]
        np.testing.assert_allclose([ba,*fold_ba],[m["balanced_accuracy"],*m["fold_balanced_accuracy"]],atol=1e-12)
        np.testing.assert_array_equal(confusion_matrix(truth,pred),m["confusion_matrix"])
        recalls = recall_score(truth,pred,labels=np.arange(18),average=None)
        np.testing.assert_allclose(recalls,[m["per_class_recall"][str(s)] for s in labels],atol=1e-12)
        diff = (pred==truth).astype(int)-(before==truth)
        hits = np.zeros_like(counts); np.add.at(hits,(gi,truth),diff)
        ci = np.quantile(((draws@hits)/(draws@counts)).mean(1),[.025,.975]).tolist()
        delta = float(np.mean([diff[truth==k].mean() for k in range(18)]))
        improved = sum(float(np.mean([diff[(fid==i)&(truth==k)].mean() for k in range(18)]))>1e-12 for i in range(5))
        comp=summary["comparisons_over_r9"][name]
        np.testing.assert_allclose([delta,*ci],[comp["pooled_ba_gain"],*comp["prompt_paired_gain_95ci"]],atol=1e-12)
        assert improved==comp["improved_folds"]
        changes = {"correct":int((pred==truth).sum()),"changed":int((pred!=before).sum()),"corrected":int(((before!=truth)&(pred==truth)).sum()),"harmed":int(((before==truth)&(pred!=truth)).sum())}
        assert changes==summary["changes"][name]
        results[name]={"balanced_accuracy":ba,"gain_over_r9":delta,"ci":ci,"improved_folds":improved,"fold_ba":fold_ba,**changes}
    r7=load(ROOT/c["r7_output"]/"candidate_oof.npz")
    gain7=results["candidate"]["balanced_accuracy"]-balanced_accuracy_score(truth,r7["probabilities"].argmax(1))
    np.testing.assert_allclose(gain7,summary["comparison_over_r7"]["pooled_ba_gain"],atol=1e-12)
    m=results["candidate"]
    checks={"pooled":m["balanced_accuracy"]>=.45,"each_fold":min(m["fold_ba"])>=.35,"gain_over_r9":m["gain_over_r9"]>=.005,
            "positive_interval":m["ci"][0]>0,"improved_folds":m["improved_folds"]>=4,"total_gain_over_r7":bool(gain7>=.01),
            "half_batch":results["half"]["gain_over_r9"]>=-.01,"singleton":results["singleton"]["gain_over_r9"]>=-.02}
    assert checks==summary["gate"]["checks"] and all(checks.values())==summary["gate"]["passed"]==summary["margin_correction"]
    return {"state":"verified","summary_sha256":sha(OUT/"summary.json"),"margin_correction":summary["margin_correction"],
            "metrics":results,"gain_over_r7":float(gain7),"gate_passed":all(checks.values()),"new_model_fits":0,"new_groupings":0,"new_holdout_evaluations":0}


if __name__ == "__main__":
    value=check()
    with (OUT/"verification.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(value))
