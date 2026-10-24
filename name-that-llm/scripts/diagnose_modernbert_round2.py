"""Descriptive diagnostics from completed development artifacts; no model fitting."""
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

from windows_modernbert_dev import MODULE, load_development, digest, object_hash, write_json


def separation(features, groups):
    x = features.astype(np.float64)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    centre = x.mean(0)
    total = float(np.square(x - centre).sum())
    means = np.zeros((len(unique), x.shape[1]), dtype=np.float64)
    np.add.at(means, inverse, x)
    means /= counts[:, None]
    between = float((counts[:, None] * np.square(means - centre)).sum())
    eta = between / total
    # Under a uniformly permuted fixed-size grouping, E[SSB/SST]=(k-1)/(N-1).
    null = (len(unique) - 1) / (len(x) - 1)
    return {"groups": len(unique), "eta_squared": eta,
            "random_group_expected_eta_squared": null,
            "adjusted_explained_fraction": 1 - (1 - eta) / (1 - null)}


def main():
    prior = MODULE / "outputs/windows_modernbert_dev_20261024_r2"
    base = json.loads((MODULE / "config/windows_modernbert_20261023.json").read_bytes())
    summary = json.loads((prior / "summary.json").read_bytes())
    assert digest(prior / "features.npz") == summary["features_sha256"] == "f600b58712a6539ff84efe1630550de558dd76f1aedb41c84c13d768bdfcc7a0"
    assert digest(prior / "oof.npz") == summary["oof_sha256"] == "105ce885457e15f5e70844e9941e519edd4c85f07deab17803f3d9833496a17b"
    data = load_development(MODULE / "data/raw/train.csv", base["input"])
    with np.load(prior / "features.npz", allow_pickle=False) as artifact:
        x = artifact["features"].copy()
        for key in ("row_index", "ids", "groups", "truth", "labels"):
            np.testing.assert_array_equal(data[key], artifact[key])
    with np.load(prior / "oof.npz", allow_pickle=False) as artifact:
        for key in ("row_index", "ids", "groups", "truth", "labels"):
            np.testing.assert_array_equal(data[key], artifact[key])
        prediction = artifact["probabilities"].argmax(1)
    token_file = MODULE / "models/modernbert-base" / base["revision"] / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(token_file))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths = np.asarray([len(result.ids) for result in tokenizer.encode_batch(data["texts"].tolist())])
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    assert np.all(norms > 0)
    normalized = x.astype(np.float64) / norms[:, None]
    result = {
        "scope": "Original 120-prompt development only; descriptive post-hoc diagnosis, not causal attribution or a new candidate score.",
        "features_sha256": digest(prior / "features.npz"), "oof_sha256": digest(prior / "oof.npz"),
        "tokenizer_sha256": digest(token_file), "rows": len(x), "dimensions": x.shape[1],
        "raw_separation": {"prompt": separation(x, data["groups"]), "model": separation(x, data["truth"])},
        "l2_separation": {"prompt": separation(normalized, data["groups"]), "model": separation(normalized, data["truth"])},
        "token_lengths_including_special_tokens": {"quantiles_0_25_50_75_90_95_100": np.quantile(lengths, [0,.25,.5,.75,.9,.95,1]).tolist(),
                                                   "over_256": int((lengths > 256).sum()),
                                                   "over_256_fraction": float((lengths > 256).mean()),
                                                   "over_384": int((lengths > 384).sum())},
        "accuracy_by_truncation": {},
        "protected_evaluations": 0, "new_model_forwards": 0, "new_classifier_fits": 0,
        "limitations": ["Group-count differences require the displayed random-group expectation; eta-squared ratios are not causal effect ratios.",
                        "Grouping diagnostics use already-used development labels and may inform only a newly registered development experiment.",
                        "Truncated/non-truncated subsets differ in topics and label mix; their accuracy gap is not a truncation ablation.",
                        "Small class-mean separation does not imply absence of useful model information in other directions."]
    }
    for name, selected in (("over_256", lengths > 256), ("at_most_256", lengths <= 256)):
        result["accuracy_by_truncation"][name] = {"rows": int(selected.sum()),
                                               "ordinary_accuracy": float((prediction[selected] == data["truth"][selected]).mean())}
    destination = MODULE / "outputs/windows_4060/round2_diagnostics_20261024.json"
    write_json(destination, result)
    print(json.dumps({"receipt_sha256": digest(destination), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
