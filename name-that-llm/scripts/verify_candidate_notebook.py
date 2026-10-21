#!/usr/bin/env python3
"""Execute the fixed candidate locally once, then audit its interface artifacts.

This executes code cells sequentially in Python, not a remote Jupyter kernel.
It performs two full-data fits, no CV/holdout evaluation, and no Kaggle request.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import nbformat
import numpy as np
import pyarrow.parquet as pq

from model_cpu import file_hash, read_csv

MODULE = Path(__file__).resolve().parents[1]
HOLDOUT_CLAIM_SHA256 = "5a93d4efc1590e6acc63bdd3d40b3f3ebf860aa85e04d6af5bda582638c68db0"


def verify(notebook_path, data_dir, output_dir, protected_claim):
    if file_hash(protected_claim) != HOLDOUT_CLAIM_SHA256:
        raise ValueError("Historical holdout claim changed; stop for review")
    notebook_digest = file_hash(notebook_path)
    document = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(document)
    for cell in document.cells:
        if cell.cell_type == "code" and (cell.outputs or cell.execution_count is not None):
            raise ValueError("The upload artifact must be unexecuted")
    # A distinct, exclusive local verification directory prevents implicit reruns.
    output_dir.mkdir(parents=True, exist_ok=False)
    env_keys = ("NTL_INPUT_DIR", "NTL_OUTPUT_DIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous_env = {key: os.environ.get(key) for key in env_keys}
    os.environ.update(NTL_INPUT_DIR=str(data_dir.resolve()), NTL_OUTPUT_DIR=str(output_dir.resolve()))
    namespace = {"__name__": "__main__"}
    count = 0
    started = time.monotonic()
    receipt = {"notebook_sha256": notebook_digest, "status": "running",
               "execution_method": "sequential Python exec of validated notebook code cells",
               "kaggle_executed": False, "formal_submission": False,
               "final_holdout_evaluations": 0}
    def timeout(signum, frame):
        raise TimeoutError("Local notebook verification exceeded 1800 seconds")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1800)
    try:
        for cell in document.cells:
            if cell.cell_type != "code":
                continue
            capture = io.StringIO()
            count += 1
            cell.execution_count = count
            try:
                with redirect_stdout(capture), redirect_stderr(capture):
                    exec(compile(cell.source, f"{notebook_path.name}:cell-{count}", "exec"), namespace)
            finally:
                cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue())]
            print(json.dumps({"executed_code_cell": count}), flush=True)
        runtime = json.loads((output_dir / "runtime.json").read_text())
        claim = json.loads((output_dir / "candidate_attempt.json").read_text())
        if (runtime["attempted_fits"] != 2 or runtime["full_training_classifier_fits"] != 2
                or claim["state"] != "completed" or claim["attempted_fits"] != 2
                or claim["completed_fits"] != 2):
            raise ValueError("Unexpected final-fit accounting")
        if runtime["version_mismatches"]:
            raise ValueError("Local reproduction has package drift")
        if runtime["final_holdout_evaluations"] != 0 or runtime["formal_submission"]:
            raise ValueError("Notebook claimed an unauthorized operation")
        if any(record["convergence_warnings"] for record in runtime["fit_records"]):
            raise ValueError("Convergence warning in final fitting")
        test = read_csv(data_dir / "test.csv")
        submission = pq.read_table(output_dir / "submission.parquet")
        if submission.column_names != ["id", "model"]:
            raise ValueError("Submission schema mismatch")
        result = submission.to_pydict()
        with np.load(output_dir / "probabilities.npz", allow_pickle=False) as values:
            if set(values.files) != {"ids", "labels", "original_style", "partial", "blend"}:
                raise ValueError("Probability artifact schema mismatch")
            labels = values["labels"].tolist()
            expected_labels = json.loads((MODULE / "config/labels.json").read_text())["labels"]
            if labels != expected_labels or values["ids"].tolist() != [r["id"] for r in test]:
                raise ValueError("Probability row/label identity mismatch")
            for key in ("original_style", "partial", "blend"):
                p = values[key]
                if (p.shape != (len(test), 18) or not np.isfinite(p).all()
                        or np.any(p < 0) or np.any(p > 1)
                        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)):
                    raise ValueError("Invalid saved probabilities")
            expected = .75 * values["original_style"] + .25 * values["partial"]
            expected /= expected.sum(axis=1, keepdims=True)
            max_difference = float(np.max(np.abs(expected - values["blend"])))
            np.testing.assert_allclose(expected, values["blend"], rtol=0, atol=1e-15)
            if (result["id"] != values["ids"].tolist()
                    or result["model"] != [labels[i] for i in expected.argmax(axis=1)]):
                raise ValueError("Submission does not match frozen weighted probabilities")
        if file_hash(notebook_path) != notebook_digest:
            raise ValueError("Notebook changed during execution")
        if file_hash(protected_claim) != HOLDOUT_CLAIM_SHA256:
            raise ValueError("Protected claim changed during execution")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt.update(status="passed", executed_code_cells=count,
            full_training_classifier_fits=2, train_rows=runtime["train_rows"], test_rows=len(test),
            train_sha256=runtime["input_hashes"]["train_sha256"],
            submission_sha256=file_hash(output_dir / "submission.parquet"),
            probabilities_sha256=file_hash(output_dir / "probabilities.npz"),
            runtime_sha256=file_hash(output_dir / "runtime.json"),
            maximum_blend_probability_difference=max_difference,
            historical_holdout_claim_sha256=HOLDOUT_CLAIM_SHA256,
            peak_rss_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
            elapsed_seconds=time.monotonic() - started,
            scope="Full-data final fitting and public-example interface check only; no leaderboard evidence")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}",
                       executed_code_cells=count, elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        nbformat.write(document, output_dir / "executed_candidate_blend.ipynb")
        (output_dir / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=MODULE / "notebooks/candidate_blend_offline.ipynb")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--protected-claim", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=MODULE / "outputs/single_submission_20261021/local_reproduction")
    arguments = parser.parse_args()
    verify(arguments.notebook, arguments.data_dir, arguments.output, arguments.protected_claim)
