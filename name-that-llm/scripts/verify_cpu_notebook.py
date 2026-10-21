#!/usr/bin/env python3
"""Execute the exported Notebook locally after finalization, then compare output.

This performs an additional full-training fit, never another holdout evaluation.
It is not a Kaggle execution or a formal competition submission.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from model_cpu import file_hash, read_csv

ROOT = Path(__file__).resolve().parents[1]


def verify_notebook(notebook_path, run_dir, data_dir, output_dir):
    for required in ("frozen_selection.json", "holdout_metrics.json", "model.joblib"):
        if not (run_dir / required).is_file():
            raise ValueError("Finalize the selected model once before running the real Notebook reproduction")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Notebook reproduction output already exists; do not silently repeat full-training fits")
    output_dir.mkdir(parents=True, exist_ok=True)
    document = json.loads(notebook_path.read_text())
    frozen = json.loads((run_dir / "frozen_selection.json").read_text())
    provenance = document["metadata"]["ntl_provenance"]
    if provenance["selected_config_id"] != frozen["selected_id"]:
        raise ValueError("Notebook configuration differs from frozen final model")
    if provenance["search_summary_sha256"] != frozen["search_summary_sha256"]:
        raise ValueError("Notebook source experiment differs from frozen final model")
    previous = {key: os.environ.get(key) for key in ("NTL_INPUT_DIR", "NTL_OUTPUT_DIR")}
    os.environ["NTL_INPUT_DIR"] = str(data_dir.resolve())
    os.environ["NTL_OUTPUT_DIR"] = str(output_dir.resolve())
    namespace = {"__name__": "__main__"}
    count = 0
    started = time.monotonic()
    try:
        with threadpool_limits(limits=2):
            for cell in document["cells"]:
                if cell["cell_type"] != "code":
                    continue
                if cell["outputs"] or cell["execution_count"] is not None:
                    raise ValueError("Deliverable Notebook must start with empty outputs and execution counts")
                capture = io.StringIO()
                count += 1
                with redirect_stdout(capture), redirect_stderr(capture):
                    exec(compile("".join(cell["source"]), f"{notebook_path.name}:cell-{count}", "exec"), namespace)
                cell["execution_count"] = count
                cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": capture.getvalue()}]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    runtime = json.loads((output_dir / "runtime.json").read_text())
    if runtime["version_mismatches"]:
        raise ValueError("Notebook runtime differs from frozen local runtime; cannot claim exact reproduction")
    saved = joblib.load(run_dir / "model.joblib")
    test = read_csv(data_dir / "test.csv")
    texts = [row["text"] for row in test]
    predictions = [saved["labels"][index] for index in saved["pipeline"].predict(texts)]
    actual = pq.read_table(output_dir / "submission.parquet").to_pydict()
    if actual["id"] != [row["id"] for row in test] or actual["model"] != predictions:
        raise ValueError("Notebook predictions differ from the frozen local model")
    old_scores = saved["pipeline"].decision_function(texts)
    new_scores = namespace["pipeline"].decision_function(texts)
    np.testing.assert_allclose(old_scores, new_scores, rtol=1e-5, atol=1e-6)
    receipt = {"scope": "Real local exported-Notebook execution; public examples verify interface only, not hidden-test quality.",
               "notebook_sha256": file_hash(notebook_path), "model_sha256": file_hash(run_dir / "model.joblib"),
               "executed_code_cells": count, "test_rows": len(test), "ids_match": True,
               "predictions_match": True, "decision_scores_match": True,
               "maximum_absolute_decision_score_difference": float(np.max(np.abs(old_scores - new_scores))),
               "runtime_json_parsed": True, "runtime": runtime,
               "additional_full_training_classifier_fits": runtime["full_training_classifier_fits"],
               "elapsed_seconds": time.monotonic() - started,
               "kaggle_executed": False, "formal_submission": False}
    (output_dir / "reproduction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output_dir / "executed_cpu_offline.ipynb").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("test_rows", "predictions_match", "decision_scores_match",
                       "additional_full_training_classifier_fits", "elapsed_seconds")}))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks" / "cpu_offline.ipynb")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "notebook_reproduction")
    args = parser.parse_args()
    verify_notebook(args.notebook, args.run_dir, args.data_dir, args.output)
