"""Reconcile nested calibration artifacts without fitting or reading raw texts."""
import json
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold
import windows_calibrated_round6 as run


def close(a,b,tol=1e-12):
    np.testing.assert_allclose(a,b,atol=tol,rtol=0)


def archive(path):
    with np.load(path,allow_pickle=False) as handle:
        return {key:handle[key].copy() for key in handle.files}


def check():
    config,_ = run.configuration()
    run.dev.verify_preserved_claims(config)
    folder = run.OUTPUT
    claim = json.loads(run.CLAIM.read_bytes())
    summary = json.loads((folder/"summary.json").read_bytes())
    receipt = json.loads((folder/"supervisor.json").read_bytes())
    contract = json.loads((folder/"contract.json").read_bytes())
    assert claim["state"] == receipt["state"] == "completed" and receipt["exit_code"] == 0
    assert run.dev.object_hash(contract) == claim["contract_sha256"] == summary["contract_sha256"]
    assert contract["config"] == config
    assert run.digest(folder/"summary.json") == claim["summary_sha256"] == receipt["summary_sha256"]
    assert run.digest(folder/"supervisor.json") == claim["supervisor_sha256"]
    assert claim["attempted_fits"] == claim["completed_fits"] == summary["attempted_fits"] == summary["completed_fits"] == 30
    assert claim["attempted_temperature_fits"] == claim["completed_temperature_fits"] == summary["attempted_temperature_fits"] == summary["completed_temperature_fits"] == 10
    assert claim["attempted_forward_calls"] == claim["completed_forward_calls"] == 0
    assert receipt["wall_seconds"] <= config["budget"]["development_timeout_seconds"]
    assert receipt["sampled_peak_process_tree_rss_bytes"] <= config["budget"]["max_host_rss_gib"]*2**30
    for path,expected in contract["sources"].items():
        assert run.digest(run.MODULE/path) == run.digest(folder/"source_snapshot"/path) == expected
    for path,expected in contract["outer_models"].items():
        assert run.digest(run.MODULE/path) == expected
    assert run.digest(folder/"oof.npz") == summary["oof_sha256"]
    base = archive(run.MODULE/config["baseline_output"]/"oof.npz")
    style = archive(run.MODULE/config["style_output"]/"style_oof.npz")
    candidate = archive(folder/"oof.npz")
    ca = archive(folder/"calibrated_style_oof.npz")
    cb = archive(folder/"calibrated_gaussian_oof.npz")
    identity = ("row_index","ids","groups","truth","labels","fold_id")
    for other in (base,style,ca,cb):
        for key in identity:
            np.testing.assert_array_equal(other[key],candidate[key])
    assert len(candidate["truth"]) == 2159 and len(np.unique(candidate["groups"])) == 120
    assert len(np.unique(candidate["row_index"])) == len(np.unique(candidate["ids"])) == 2159
    assert str(candidate["contract_sha256"]) == str(ca["contract_sha256"]) == str(cb["contract_sha256"]) == claim["contract_sha256"]
    assert run.dev.object_hash({k:candidate[k].tolist() for k in identity if k != "fold_id"}) == contract["row_identity_sha256"]
    assert run.dev.object_hash(candidate["fold_id"].tolist()) == contract["fold_sha256"]
    for item in (candidate,ca,cb,base,style):
        p = item["probabilities"]
        assert p.shape == (2159,18) and np.isfinite(p).all() and np.all((0<=p)&(p<=1))
        close(p.sum(1),np.ones(2159))
    truth,groups,fold = candidate["truth"],candidate["groups"],candidate["fold_id"]
    nested_ids = []
    objective_calls = 0
    assert len(summary["folds"]) == 5
    for number,record in enumerate(summary["folds"]):
        assert record["fold"] == number
        tr,va = np.where(fold!=number)[0],np.where(fold==number)[0]
        path = folder/f"fold{number}"
        for file,key in (("inner_scores.npz","inner_scores_sha256"),("outer_scores.npz","outer_scores_sha256"),("calibration.json","calibration_sha256")):
            assert run.digest(path/file) == record[key]
        inner,outer = archive(path/"inner_scores.npz"),archive(path/"outer_scores.npz")
        np.testing.assert_array_equal(inner["outer_train_positions"],tr)
        np.testing.assert_array_equal(inner["outer_valid_positions"],va)
        for key in ("row_index","ids","groups","truth"):
            np.testing.assert_array_equal(inner[key],candidate[key][tr])
            np.testing.assert_array_equal(outer[key],candidate[key][va])
        np.testing.assert_array_equal(inner["labels"],candidate["labels"])
        np.testing.assert_array_equal(outer["labels"],candidate["labels"])
        np.testing.assert_array_equal(outer["fold_id"],fold[va])
        assert str(inner["contract_sha256"]) == str(outer["contract_sha256"]) == claim["contract_sha256"]
        assert not set(inner["groups"]) & set(outer["groups"])
        inner_id = np.full(len(tr),-1)
        assert len(record["inner_records"]) == 3
        for i,(itr,iva) in enumerate(GroupKFold(n_splits=3).split(tr,truth[tr],groups[tr])):
            r = record["inner_records"][i]
            inner_id[iva] = i
            assert r["inner_fold"] == i and r["train_rows"] == len(itr) and r["valid_rows"] == len(iva)
            assert not set(groups[tr[itr]]) & set(groups[tr[iva]])
            assert r["train_row_identity_sha256"] == run.dev.object_hash(candidate["row_index"][tr[itr]].tolist())
            assert r["valid_row_identity_sha256"] == run.dev.object_hash(candidate["row_index"][tr[iva]].tolist())
            assert r["gaussian"]["dimensions"] == 128
        np.testing.assert_array_equal(inner_id,inner["inner_fold_id"])
        nested_ids.append(inner_id.tolist())
        calibration = json.loads((path/"calibration.json").read_bytes())
        assert calibration == record["calibration"]
        for family,source,reference,target in (("style",style,style,ca),("gaussian",base,base,cb)):
            scores = inner[f"{family}_scores"]
            val_scores = outer[f"{family}_scores"]
            assert scores.shape == (len(tr),18) and val_scores.shape == (len(va),18)
            assert np.isfinite(scores).all() and np.isfinite(val_scores).all()
            c = calibration[family]
            beta = c["inverse_temperature"]
            assert .05 <= beta <= 50 and c["training_rows"] == len(tr)
            assert 0 < c["objective_evaluations"] <= 100
            objective_calls += c["objective_evaluations"]
            close(beta*c["temperature"],1)
            before = np.mean(logsumexp(scores,axis=1)-scores[np.arange(len(tr)),truth[tr]])
            logits = scores*beta
            after = np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(tr)),truth[tr]])
            close(before,c["training_nll_before"])
            close(after,c["training_nll_after"])
            assert after <= before + 1e-8
            derivative = np.mean((softmax(logits,axis=1)*scores).sum(1)-scores[np.arange(len(tr)),truth[tr]])
            if .050001 < beta < 49.999999:
                assert abs(derivative) < 1e-5
            close(softmax(val_scores,axis=1),reference["probabilities"][va])
            close(softmax(val_scores*beta,axis=1),target["probabilities"][va])
            np.testing.assert_array_equal(target["probabilities"][va].argmax(1),reference["probabilities"][va].argmax(1))
        close(candidate["probabilities"][va],.8*ca["probabilities"][va]+.2*cb["probabilities"][va])
    assert run.dev.object_hash(nested_ids) == contract["inner_fold_sha256"]
    pred = candidate["probabilities"].argmax(1)
    metrics = summary["metrics"]
    close(balanced_accuracy_score(truth,pred),metrics["balanced_accuracy"])
    close([balanced_accuracy_score(truth[fold==i],pred[fold==i]) for i in range(5)],metrics["fold_balanced_accuracy"])
    close(recall_score(truth,pred,average=None,labels=np.arange(18)),[metrics["per_class_recall"][label] for label in candidate["labels"]])
    np.testing.assert_array_equal(confusion_matrix(truth,pred,labels=np.arange(18)),metrics["confusion_matrix"])
    _,gi = np.unique(groups,return_inverse=True)
    counts = np.zeros((120,18),dtype=int)
    np.add.at(counts,(gi,truth),1)
    draws = np.random.default_rng(config["evaluation"]["bootstrap_seed"]).multinomial(120,np.full(120,1/120),size=2000)
    den = draws@counts
    for reference,key in ((base,"paired_round3_comparison"),(style,"paired_style_control_comparison")):
        diff = (pred==truth).astype(int)-(reference["probabilities"].argmax(1)==truth)
        report = summary[key]
        close(np.mean([diff[truth==i].mean() for i in range(18)]),report["pooled_ba_gain"])
        gains = [np.mean([diff[(fold==f)&(truth==i)].mean() for i in range(18)]) for f in range(5)]
        close(gains,report["fold_ba_gains"])
        assert sum(v>1e-12 for v in gains) == report["improved_folds"]
        hits = np.zeros_like(counts)
        np.add.at(hits,(gi,truth),diff)
        close(np.quantile(((draws@hits)/den).mean(1),[.025,.975]),report["prompt_paired_gain_95ci"])
    hits = np.zeros_like(counts)
    np.add.at(hits,(gi,truth),pred==truth)
    close(np.quantile(((draws@hits)/den).mean(1),[.025,.975]),metrics["prompt_bootstrap_ba_95ci"])
    gate = run.previous.metric_gate(metrics,summary["paired_round3_comparison"],summary["paired_style_control_comparison"],config)
    assert gate == summary["candidate_gate"] and gate["metric_gate_passed"] == receipt["candidate_development_gate_passed"]
    return {"state":"verified","rows":2159,"groups":120,"outer_folds":5,"inner_folds_each":3,
            "classifier_fits":30,"temperature_fits":10,"objective_evaluations":objective_calls,
            "balanced_accuracy":metrics["balanced_accuracy"],"development_gate_passed":gate["metric_gate_passed"],
            "summary_sha256":run.digest(folder/"summary.json"),"supervisor_sha256":run.digest(folder/"supervisor.json")}


if __name__ == "__main__":
    print(json.dumps(check()))
