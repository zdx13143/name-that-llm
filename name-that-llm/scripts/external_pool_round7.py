"""Prepare attributed public passages; learn punctuation exclusions on training rows only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from probe_modernbert import digest, write_json

MODULE = Path(__file__).resolve().parents[1]
REFERENCE = MODULE / "outputs/round7_reference_20261026"
MARKS = ("\u2014", "\u2018", "\u2019", "\u201c", "\u201d")
EXCLUDED_OWNERS = {"haileycheng", "michaelsemenoff"}
MARKDOWN = re.compile(r"```|\*\*|(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)")


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def shingles(text, n=8):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return {tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))}


def canonical_group(owner, row):
    pid = row.get("prompt_id") or row.get("pid")
    if not pid:
        raise ValueError(f"Missing external prompt id: {owner}")
    if owner in {"seyitkaangunes", "willrice", "alperen5252525", "kaplinskiy", "abdullah1afify", "dophuchung"}:
        return f"seyit:{pid}"
    if owner == "lucasboesen":
        bank = row["prompt_source"]
        return f"{'seyit' if bank == 'seyitkaangunes' else bank}:{pid}"
    if owner == "prvsiyan":
        if row.get("split") == "proxy_v1":
            return f"prvsiyan_v1:{pid.removeprefix('v1_')}"
        if "split" not in row:
            return f"prvsiyan_v1:{pid}"
        return f"prvsiyan_new:{pid}"
    if owner == "lxdlxdlxd":
        return f"lx:{row['run']}:{pid}"
    return f"{owner}:{pid}"


def prepare_row(source, row, labels):
    """Reject reconstructed official prompts before considering their text."""
    owner = source.split("/")[0]
    if owner in EXCLUDED_OWNERS:
        return None, "source_provenance_not_established"
    if owner == "prvsiyan" and "split" in row and row["split"] not in {"new", "new_kimi_nothink", "proxy_v1"}:
        return None, "excluded_prvsiyan_split:" + row["split"]
    label = row.get("label") or row.get("model")
    if label not in labels:
        raise ValueError(f"Unknown external label: {label}")
    for key in ("is_truncated", "is_empty", "is_refusal", "has_markdown"):
        if row.get(key, "").lower() == "true":
            return None, key
    if row.get("served_matches", "true").lower() != "true":
        return None, "served_model_mismatch"
    reason = row.get("finish_reason_norm", row.get("finish_reason"))
    if reason is not None and reason.casefold() not in {"stop", "end_turn"}:
        return None, "unfinished_or_missing_finish"
    text = row["text"].strip()
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    if not text or "\x00" in text or re.search(r"\n\s*\n", text) or MARKDOWN.search(text):
        return None, "empty_or_multiple_paragraphs_or_markdown"
    text = " ".join(text.split())
    if not 30 <= len(text.split()) <= 300:
        return None, "word_count_outside_30_300"
    return {"text": text, "model": label, "group": canonical_group(owner, row), "source": source}, None


def remove_overlaps(rows, development_texts):
    """Label-free copy exclusion; entire known external prompt groups are removed."""
    exact = set(map(normalized, development_texts))
    index = defaultdict(list)
    for i, text in enumerate(development_texts):
        for shingle in shingles(text):
            index[shingle].append(i)
    bad_groups, direct = set(), 0
    for row in rows:
        matches = Counter()
        for shingle in shingles(row["text"]):
            matches.update(index.get(shingle, ()))
        if normalized(row["text"]) in exact or max(matches.values(), default=0) >= 3:
            bad_groups.add(row["group"])
            direct += 1
    kept = [r for r in rows if r["group"] not in bad_groups]
    return kept, {"direct_overlap_rows": direct, "excluded_groups": sorted(bad_groups), "excluded_group_rows": len(rows)-len(kept)}


def prepare(manifest, labels, development_texts):
    rows, rejections = [], Counter()
    for spec in manifest:
        if not spec["name"].endswith(".csv"):
            continue
        path = REFERENCE / spec["local_path"]
        if digest(path) != spec["sha256"]:
            raise ValueError("Raw external file identity changed")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for original in csv.DictReader(handle):
                row, reason = prepare_row(spec["source"], original, labels)
                if reason:
                    rejections[spec["source"] + ":" + reason] += 1
                else:
                    rows.append(row)
    rows, overlap = remove_overlaps(rows, development_texts)
    by_text = defaultdict(list)
    for row in rows:
        by_text[normalized(row["text"])].append(row)
    unique, duplicate_rows, conflicts = [], 0, 0
    for key in sorted(by_text):
        copies = by_text[key]
        if len({r["model"] for r in copies}) != 1:
            conflicts += len(copies)
            continue
        copies.sort(key=lambda r: (r["source"], r["group"], r["text"]))
        unique.append(copies[0])
        duplicate_rows += len(copies)-1
    kept = unique
    if set(r["model"] for r in kept) != set(labels):
        raise ValueError("Prepared external pool lacks a class")
    info = {"raw_manifest_sha256": digest(REFERENCE / "raw_download_manifest.json"),
            "source_rejections": dict(sorted(rejections.items())), "normalized_duplicate_rows_removed": duplicate_rows,
            "conflicting_label_rows_removed": conflicts, "copy_check": overlap,
            "prepared_rows": len(kept), "prepared_groups": len({r["group"] for r in kept}),
            "class_counts": dict(sorted(Counter(r["model"] for r in kept).items())),
            "source_counts": dict(sorted(Counter(r["source"] for r in kept).items())),
            "official_rows_checked": len(development_texts), "official_labels_used_for_cleaning": False,
            "protected_holdout_evaluations": 0,
            "limitations": "Source authors used official-training summaries and some upstream filters; not an independent validation set. Three shared 8-grams is a copy screen, not an exhaustive semantic-prompt audit."}
    return kept, info


def load_pool(path, expected_sha256, labels):
    if digest(path) != expected_sha256:
        raise ValueError("Prepared external pool changed")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    texts = np.asarray([r["text"] for r in rows])
    truth = np.asarray([labels.index(r["model"]) for r in rows], dtype=np.int64)
    return {"texts": texts, "truth": truth, "groups": np.asarray([r["group"] for r in rows]),
            "ids": np.asarray([hashlib.sha256((r["model"]+"\0"+r["text"]).encode()).hexdigest() for r in rows])}


def fold_pool(pool, training_texts, training_truth, classes=18):
    present = np.zeros((classes, len(MARKS)), dtype=bool)
    for label in range(classes):
        texts = training_texts[training_truth == label]
        if len(texts) == 0:
            raise ValueError("Training fold lacks a class")
        present[label] = [any(mark in t for t in texts) for mark in MARKS]
    keep = np.ones(len(pool["truth"]), dtype=bool)
    for i, (text, label) in enumerate(zip(pool["texts"], pool["truth"])):
        keep[i] = not any(mark in text and not present[label, j] for j, mark in enumerate(MARKS))
    if set(pool["truth"][keep]) != set(range(classes)):
        raise ValueError("Fold-filtered pool lacks a class")
    return np.flatnonzero(keep), present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("Use --prepare for data curation; this script never trains a model")
    import windows_modernbert_dev as dev
    base = json.loads((MODULE / "config/windows_modernbert_20261023.json").read_bytes())
    data = dev.load_development(MODULE / "data/raw/train.csv", base["input"], MODULE / "config/labels.json")
    manifest = json.loads((REFERENCE / "raw_download_manifest.json").read_bytes())
    rows, info = prepare(manifest, data["labels"].tolist(), data["texts"])
    path = REFERENCE / "prepared_pool.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "model", "group", "source"])
        writer.writeheader()
        writer.writerows(rows)
    info["prepared_csv_sha256"] = digest(path)
    info["preparer_sha256"] = digest(Path(__file__))
    write_json(REFERENCE / "prepared_pool_receipt.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
