"""Validate immutable inputs and saved OOF arrays before reusing experiments."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from model_cpu import config_id, file_hash, read_csv

ROOT = Path(__file__).resolve().parents[1]


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_holdout_available(train_hash, project_root=None):
    project_root = Path(project_root) if project_root else ROOT
    outputs = project_root / "outputs"
    if (outputs / "holdout_claims" / f"{train_hash}.json").exists():
        raise ValueError("Shared final holdout was already claimed or opened")
    # Also honor legacy markers and legacy train+split claim names.
    if any((outputs / "holdout_claims").glob(f"{train_hash}-*.json")):
        raise ValueError("Shared final holdout was already claimed or opened")
    for marker in outputs.glob("*/holdout_opened.json"):
        summary_path = marker.parent / "search_summary.json"
        if not summary_path.exists() or json.loads(summary_path.read_text())["train_sha256"] == train_hash:
            raise ValueError("Shared final holdout was already claimed or opened")


def inspect_training(data_dir, labels_path, expected_train_hash, expected_label_hash):
    train = Path(data_dir) / "train.csv"
    if file_hash(train) != expected_train_hash:
        raise ValueError("Prior training data hash differs from current input")
    if file_hash(labels_path) != expected_label_hash:
        raise ValueError("Prior label reference hash differs from current input")
    with train.open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    required = ["id", "prompt_id", "text", "model"]
    if len(columns) != len(set(columns)) or set(columns) != set(required):
        raise ValueError("Training CSV schema differs from the audited schema")
    schema = {"format": "CSV UTF-8", "columns": columns,
              "field_representation": "text", "required_columns": required}
    labels = json.loads(Path(labels_path).read_text())["labels"]
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError("Expected exactly 18 distinct frozen labels")
    rows = read_csv(train)
    if any(None in row or any(not isinstance(row.get(key), str) or not row[key].strip()
                             for key in required) for row in rows):
        raise ValueError("Training CSV has malformed or blank required fields")
    if any(row["model"] not in labels for row in rows):
        raise ValueError("Training label is outside the frozen reference")
    y = np.array([labels.index(row["model"]) for row in rows], dtype=int)
    return rows, labels, y, schema


def validate_oof(path, dev, y, n_labels, expected_score=None):
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"row_index", "truth", "prediction"}:
            raise ValueError("OOF schema requires row_index, truth, prediction only")
        arrays = {key: artifact[key].copy() for key in artifact.files}
    for name, value in arrays.items():
        if value.ndim != 1 or value.dtype.kind not in "iu" or len(value) != len(dev):
            raise ValueError(f"OOF schema has invalid shape or integer dtype: {name}")
    if not np.array_equal(arrays["row_index"], dev):
        raise ValueError("Prior OOF row order differs from frozen development rows")
    if len(set(arrays["row_index"])) != len(dev):
        raise ValueError("Prior OOF row indices are not unique")
    if not np.array_equal(arrays["truth"], y[dev]):
        raise ValueError("Prior OOF truth differs from current frozen label mapping")
    if np.any(arrays["prediction"] < 0) or np.any(arrays["prediction"] >= n_labels):
        raise ValueError("Prior OOF contains an invalid predicted class")
    score = float(balanced_accuracy_score(y[dev], arrays["prediction"]))
    if expected_score is not None and not np.isclose(score, expected_score, atol=1e-12, rtol=0):
        raise ValueError("Prior OOF predictions do not reproduce the recorded score")
    schema = {key: {"dtype": str(value.dtype), "shape": list(value.shape)}
              for key, value in sorted(arrays.items())}
    return arrays, schema, score


def validate_prior(run_dir, data_dir, labels_path, persist_contract=True):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "search_summary.json").read_text())
    rows, labels, y, train_schema = inspect_training(data_dir, labels_path,
        summary["train_sha256"], summary["label_reference_sha256"])
    manifest_path = run_dir / "split_manifest.json"
    if file_hash(manifest_path) != summary["split_sha256"]:
        raise ValueError("Prior frozen split hash changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["train_sha256"] != summary["train_sha256"]:
        raise ValueError("Prior manifest references a different training input")
    dev = np.asarray(manifest["development_indices"])
    heldout = np.asarray(manifest["holdout_indices"])
    if any(a.ndim != 1 or a.dtype.kind not in "iu" or len(set(a)) != len(a) for a in (dev, heldout)):
        raise ValueError("Prior split indices must be unique integer vectors")
    if set(dev) & set(heldout) or set(dev) | set(heldout) != set(range(len(rows))):
        raise ValueError("Prior split is not a complete disjoint partition")
    groups = np.array([row["prompt_id"] for row in rows])
    if set(groups[dev]) & set(groups[heldout]):
        raise ValueError("Prior split shares prompt groups with final holdout")
    selected = summary["selected"]
    if selected["id"] != config_id(selected["config"]):
        raise ValueError("Prior selected config hash changed")
    oof_path = run_dir / "trials" / f"{selected['id']}-{summary['seed']}-oof.npz"
    arrays, oof_schema, _ = validate_oof(oof_path, dev, y, len(labels), selected["balanced_accuracy"])
    contract = {"version": 1, "train_sha256": summary["train_sha256"],
        "label_reference_sha256": summary["label_reference_sha256"],
        "split_sha256": summary["split_sha256"],
        "training_schema": train_schema, "training_schema_sha256": object_hash(train_schema),
        "oof_schema": oof_schema, "oof_schema_sha256": object_hash(oof_schema),
        "selected_oof_sha256": file_hash(oof_path),
        "selected_id": selected["id"], "search_summary_sha256": file_hash(run_dir / "search_summary.json")}
    contract_path = run_dir / "artifact_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Frozen prior input/OOF schema or artifact hash changed")
    elif persist_contract:
        with contract_path.open("x") as handle:
            json.dump(contract, handle, indent=2)
            handle.write("\n")
    return {"summary": summary, "contract": contract, "contract_sha256": object_hash(contract),
            "rows": rows, "labels": labels, "y": y, "dev": dev,
            "heldout": heldout, "groups": groups, "oof": arrays}
