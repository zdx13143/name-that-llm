"""Frozen R7 SVC inference: no fitting, no encoder, no network."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

VERSIONS = {"numpy": "2.2.6", "scipy": "1.16.3", "scikit-learn": "1.7.2",
            "joblib": "1.5.2", "threadpoolctl": "3.6.0", "pyarrow": "21.0.0"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def block_network():
    def denied(*args, **kwargs):
        raise RuntimeError("Network disabled for frozen inference")
    socket.create_connection = socket.socket.connect = socket.socket.connect_ex = denied
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"


def verify_assets(root, expected_manifest):
    if digest(root / "manifest.json") != expected_manifest:
        raise ValueError("Manifest identity changed")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (manifest["candidate"] != "windows-external-weighting-requested-final-20261026-r7"
            or manifest["official_weight"] != 4 or manifest["development_gate_passed"] is not False):
        raise ValueError("Wrong fixed candidate or altered gate statement")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Unexpected asset file set")
    for name, record in manifest["files"].items():
        path = root / name
        if (not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size != record["bytes"]
                or digest(path) != record["sha256"]):
            raise ValueError(f"Asset bytes changed: {name}")
    return manifest


def verify_runtime():
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if sys.platform != "linux" or sys.version_info[:2] != (3, 11) or versions != VERSIONS:
        raise ValueError(f"Expected isolated Linux Python 3.11 and pinned dependencies: {versions}")
    return versions


def read_test(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "text"]:
            raise ValueError("Expected test id,text columns only")
        rows = list(reader)
    if (not 1 <= len(rows) <= 4096 or any(not row["id"] or not row["text"] or None in row for row in rows)
            or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("Invalid or duplicate test rows, or more than 4096 rows")
    return [row["id"] for row in rows], [row["text"] for row in rows]


def write_submission(path, ids, scores, labels):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    if (len(labels) != 18 or len(set(labels)) != 18 or scores.shape != (len(ids), 18)
            or not np.isfinite(scores).all() or len(set(ids)) != len(ids)):
        raise ValueError("Invalid submission mapping")
    models = [str(labels[int(i)]) for i in scores.argmax(axis=1)]
    if path.exists():
        raise FileExistsError("Preserve existing submission")
    pq.write_table(pa.table({"id": pa.array(ids, type=pa.string()), "model": pa.array(models, type=pa.string())}), path)
    restored = pq.read_table(path)
    if (restored.column_names != ["id", "model"] or restored["id"].to_pylist() != ids
            or restored["model"].to_pylist() != models):
        raise ValueError("Parquet round-trip mismatch")


def run(args):
    block_network()
    assets, output = args.assets.resolve(), args.output.resolve()
    manifest = verify_assets(assets, args.manifest_sha256)
    versions = verify_runtime()
    output.mkdir(parents=True, exist_ok=False)
    claim = {"state": "started", "action": args.action, "manifest_sha256": args.manifest_sha256,
             "versions": versions, "python": sys.version, "model_fits": 0, "encoder_forward_calls": 0,
             "old_holdout_evaluations": 0, "attempted_prediction_calls": 0, "completed_prediction_calls": 0,
             "development_gate_passed": False}
    write_json(output / "claim.json", claim, True)
    start, expired = time.monotonic(), threading.Event()
    def timeout():
        if not expired.wait(900 if args.action == "synthetic" else 32400):
            (output / "timeout.txt").write_text("Inference wall-time exceeded", encoding="utf-8")
            os._exit(124)
    threading.Thread(target=timeout, daemon=True).start()
    try:
        import joblib
        import numpy as np
        from threadpoolctl import threadpool_limits
        from weighted_style_round7 import decision
        artifact = joblib.load(assets / "weighted_svc.joblib")
        labels = json.loads((assets / "labels.json").read_bytes())["labels"]
        if (artifact["contract_sha256"] != manifest["final_contract_sha256"]
                or artifact["labels"].tolist() != labels):
            raise ValueError("Final model contract or label order changed")
        if [(name, weight) for name, _, weight in artifact["views"]] != [("char", 1.0), ("word", .7), ("shape", .5), ("function", .5)]:
            raise ValueError("Frozen views changed")
        if any(np.dtype(view.dtype) != np.dtype("float32") for _, view, _ in artifact["views"]):
            raise ValueError("Frozen R7 TF-IDF dtype changed")
        def predict(texts):
            claim["attempted_prediction_calls"] += 1
            write_json(output / "claim.json", claim)
            result = decision(artifact["views"], artifact["classifier"], texts)
            claim["completed_prediction_calls"] += 1
            write_json(output / "claim.json", claim)
            return result
        with threadpool_limits(limits=2):
            if args.action == "synthetic":
                with np.load(assets / "synthetic_scores.npz", allow_pickle=False) as saved:
                    texts, reference = saved["texts"].tolist(), saved["scores"].copy()
                    if saved["labels"].tolist() != labels or saved["contract_sha256"].item() != manifest["final_contract_sha256"]:
                        raise ValueError("Synthetic reference identity changed")
                ids = [f"artificial-{i}" for i in range(len(texts))]
                scores = predict(texts)
                reverse = predict(texts[::-1])[::-1]
                singleton = np.concatenate([predict([text]) for text in texts])
                error = max(float(np.abs(scores - reference).max()), float(np.abs(reverse - scores).max()), float(np.abs(singleton - scores).max()))
                if error > 1e-7 or not np.array_equal(scores.argmax(1), reference.argmax(1)):
                    raise ValueError(f"Cross-platform synthetic mismatch: {error}")
                claim.update(synthetic_max_abs_error=error, scope="four artificial texts only; no competition data evaluated")
            else:
                ids, texts = read_test(args.test)
                scores = np.concatenate([predict(texts[at:at + 128]) for at in range(0, len(texts), 128)])
        write_submission(output / "submission.parquet", ids, scores, labels)
        claim.update(state="completed", rows=len(ids), output_sha256=digest(output / "submission.parquet"), wall_seconds=time.monotonic() - start)
        write_json(output / "claim.json", claim)
        print(json.dumps(claim), flush=True)
    except BaseException as error:
        claim.update(state="failed", error_type=type(error).__name__, error=str(error), wall_seconds=time.monotonic() - start)
        write_json(output / "claim.json", claim)
        raise
    finally:
        expired.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["synthetic", "test"])
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--test", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "test" and arguments.test is None:
        parser.error("test action requires --test")
    run(arguments)
