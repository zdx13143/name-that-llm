"""Restore byte-identical historical split metadata from the audited train.csv.

The 30-prompt final holdout was already used. Recreating its metadata does not
reopen it or authorize further evaluation. This tool keeps only CSV id/prompt_id
metadata, performs no model fitting or scoring, and never imports old search or
finalization code. Its only output is the fixed local team-input manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/team_dev_20261021.json"
DESTINATION = MODULE / "outputs/team_inputs/split_manifest.json"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SPLIT_SHA256 = "c8c24921f617ad483b9e14cb23d0d64a0fbbde44c0214868445e8d0e30d6e1db"
SEED = 20261018
STABILITY_SEEDS = [20261019, 20261020]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_manifest_bytes(metadata):
    """Reproduce only the original freeze_split metadata algorithm and encoding."""
    groups = np.array([row["prompt_id"] for row in metadata])
    unique = np.unique(groups)
    if len(metadata) != 2699 or len(unique) != 150:
        raise ValueError("Audited metadata must have 2,699 rows and 150 prompt groups")
    shuffled = np.random.default_rng(SEED).permutation(unique)
    holdout_groups = shuffled[:max(1, int(np.ceil(len(unique) * 0.2)))]
    heldout = np.flatnonzero(np.isin(groups, holdout_groups))
    dev = np.flatnonzero(~np.isin(groups, holdout_groups))

    # Historical seeded folds, not the new team's GroupKFold comparison folds.
    dev_groups = np.random.default_rng(SEED).permutation(np.unique(groups[dev]))
    fold_of = {int(index): fold
               for fold, selected in enumerate(np.array_split(dev_groups, 5))
               for index in dev[np.isin(groups[dev], selected)]}
    manifest = {
        "seed": SEED, "stability_seeds": STABILITY_SEEDS,
        "train_sha256": TRAIN_SHA256,
        "development_indices": dev.tolist(), "holdout_indices": heldout.tolist(),
        "rows": [{"row_index": index, "id": row["id"], "prompt_id": row["prompt_id"],
                  "role": "development" if index in fold_of else "holdout",
                  "development_fold": fold_of.get(index)}
                 for index, row in enumerate(metadata)],
    }
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _persist_verified_manifest(destination, payload, expected_sha256):
    """Create once, or verify identical existing bytes without rewriting them."""
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Reconstructed manifest SHA-256 differs from the frozen split")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise ValueError("Existing manifest has different bytes; refusing to overwrite")
        return "verified_existing"
    return "created"


def restore_split(train_path):
    config = json.loads(CONFIG.read_text())
    if config["train_sha256"] != TRAIN_SHA256 or config["split_sha256"] != SPLIT_SHA256:
        raise ValueError("Team configuration does not reference the fixed historical inputs")
    if file_sha256(train_path) != TRAIN_SHA256:
        raise ValueError("Official training file SHA-256 mismatch")
    metadata = []
    with Path(train_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"id", "prompt_id"}.issubset(reader.fieldnames):
            raise ValueError("Training CSV lacks split metadata columns")
        for row in reader:
            metadata.append({"id": row["id"], "prompt_id": row["prompt_id"]})
    payload = historical_manifest_bytes(metadata)
    state = _persist_verified_manifest(DESTINATION, payload, config["split_sha256"])
    return {"state": state, "path": str(DESTINATION), "split_sha256": SPLIT_SHA256,
            "development_rows": 2159, "development_groups": 120,
            "protected_rows": 540, "protected_groups": 30,
            "training_fits": 0, "holdout_evaluations": 0,
            "scope": "Restore existing split metadata only; the historical final holdout remains used."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Official audited train.csv")
    args = parser.parse_args()
    try:
        receipt = restore_split(args.train)
    except (OSError, ValueError, KeyError) as error:
        print(f"Split restoration refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
