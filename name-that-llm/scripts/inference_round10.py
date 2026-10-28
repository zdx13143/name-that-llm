"""Frozen topic-word SVC with whole-batch text-only grouping and margin shrinkage."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from inference_round7 import VERSIONS, digest, write_json, block_network, verify_runtime, read_test, write_submission


def verify_assets(root, expected_manifest):
    if digest(root / "manifest.json") != expected_manifest:
        raise ValueError("Manifest identity changed")
    m = json.loads((root / "manifest.json").read_bytes())
    if (m["candidate"] != "windows-topic-word-final-20261028-r10" or m["official_weight"] != 4
            or m["word_alpha"] != .5 or m["grouping"] != dict(min_df=2, max_df=.05, cap=18, threshold=.15)
            or m["margin_correction"] != m["development_gate_passed"]):
        raise ValueError("Unexpected candidate metadata")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if actual != set(m["files"]) | {"manifest.json"}:
        raise ValueError("Unexpected asset files")
    for name, record in m["files"].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or p.stat().st_size != record["bytes"] or digest(p) != record["sha256"]:
            raise ValueError("Asset bytes changed: " + name)
    return m


def run(args):
    block_network()
    assets, output = args.assets.resolve(), args.output.resolve()
    m, versions = verify_assets(assets, args.manifest_sha256), verify_runtime()
    output.mkdir(parents=True, exist_ok=False)
    claim = {"state": "started", "action": args.action, "manifest_sha256": args.manifest_sha256, "versions": versions,
             "python": sys.version, "model_fits": 0, "supervised_vectorizer_fits": 0, "encoder_forward_calls": 0,
             "old_holdout_evaluations": 0, "attempted_prediction_calls": 0, "completed_prediction_calls": 0,
             "text_only_grouping_calls": 0, "grouping_vectorizer_fit_attempts": 0,
             "development_gate_passed": m["development_gate_passed"], "margin_correction": m["margin_correction"]}
    write_json(output / "claim.json", claim, True)
    start, finished = time.monotonic(), threading.Event()
    def watchdog():
        if not finished.wait(900 if args.action == "synthetic" else 32400):
            (output / "timeout.txt").write_text("Inference wall-clock exceeded", encoding="utf-8")
            os._exit(124)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        import joblib
        import numpy as np
        from threadpoolctl import threadpool_limits
        from margin_center_round10 import predict
        artifact = joblib.load(assets / "topic_svc.joblib")
        labels = json.loads((assets / "labels.json").read_bytes())["labels"]
        if (artifact["contract_sha256"] != m["final_contract_sha256"] or artifact["labels"].tolist() != labels
                or artifact["word_alpha"] != m["word_alpha"] or artifact["grouping"] != m["grouping"]
                or artifact["margin_correction"] != m["margin_correction"]
                or [(n,w) for n,_,w in artifact["views"]] != [("char",1.0),("word",.7),("shape",.5),("function",.5)]
                or any(np.dtype(v.dtype) != np.dtype("float32") for _,v,_ in artifact["views"])):
            raise ValueError("Model metadata differs")
        def infer(texts, ids):
            claim["attempted_prediction_calls"] += 1
            claim["text_only_grouping_calls"] += 1
            claim["grouping_vectorizer_fit_attempts"] += int(len(texts) >= 40)
            write_json(output / "claim.json", claim)
            result, groups = predict(artifact, texts, ids)
            claim["completed_prediction_calls"] += 1
            write_json(output / "claim.json", claim)
            return result, groups
        with threadpool_limits(limits=2):
            if args.action == "synthetic":
                with np.load(assets / "synthetic_scores.npz", allow_pickle=False) as z:
                    texts, ids, reference, reference_groups = z["texts"].tolist(), z["ids"].tolist(), z["scores"].copy(), z["groups"].copy()
                    if z["labels"].tolist() != labels or str(z["contract_sha256"]) != m["final_contract_sha256"]:
                        raise ValueError("Synthetic identity differs")
                scores, groups = infer(texts, ids)
                reverse, reverse_groups = infer(texts[::-1], ids[::-1])
                single, single_group = infer(texts[:1], ids[:1])
                np.testing.assert_array_equal(groups, reference_groups)
                np.testing.assert_array_equal(groups, reverse_groups[::-1])
                if len(texts) != 80 or sorted(np.unique(groups, return_counts=True)[1]) != [2] * 40 or single.shape != (1,18):
                    raise ValueError("Synthetic group paths not exercised")
                error = max(float(np.abs(scores-reference).max()), float(np.abs(scores-reverse[::-1]).max()))
                if error > 1e-7 or not np.array_equal(scores.argmax(1), reference.argmax(1)):
                    raise ValueError("Synthetic cross-platform/order mismatch")
                claim.update(synthetic_max_abs_error=error, scope="80 artificial texts, complete/reverse/singleton; no real data")
            else:
                ids, texts = read_test(args.test)
                scores, groups = infer(texts, ids)
            unique, sizes = np.unique(groups, return_counts=True)
        write_submission(output / "submission.parquet", ids, scores, labels)
        claim.update(state="completed", rows=len(ids), clusters=len(unique), singleton_rows=int(sizes[sizes==1].sum()),
                     eligible_rows=int(sizes[(sizes>=2)&(sizes<=18)].sum()), output_sha256=digest(output / "submission.parquet"),
                     wall_seconds=time.monotonic()-start)
        write_json(output / "claim.json", claim)
        print(json.dumps(claim), flush=True)
    except BaseException as error:
        claim.update(state="failed", error_type=type(error).__name__, error=str(error), wall_seconds=time.monotonic()-start)
        write_json(output / "claim.json", claim)
        raise
    finally:
        finished.set()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["synthetic", "test"])
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--test", type=Path)
    args = p.parse_args()
    if args.action == "test" and args.test is None:
        p.error("test action requires --test")
    run(args)
