#!/usr/bin/env python3
"""Audit complete local competition files; emit aggregates, never source rows.

From the workspace root (uses the already installed local environment):
  enveda-casmi-2026/.venv/bin/python name-that-llm/scripts/profile_data.py

For attached official data in a CPU Kaggle Notebook:
  python scripts/profile_data.py --data-dir /kaggle/input/competitions/name-that-llm \
      --output /kaggle/working/local_data_profile.json

Exit 0: audit completed with no error findings (review findings can remain).
Exit 1: audit completed and found integrity errors; read the JSON report.
Exit 2: missing/unreadable inputs or invalid configuration; no new report.
Synthetic fixture tests validate this tool, not the real competition data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "train.csv": ["id", "prompt_id", "text", "model"],
    "test.csv": ["id", "text"],
    "sample_submission.parquet": ["id", "model"],
}


class AuditInputError(Exception):
    """A safe, value-free diagnostic for an input that cannot be audited."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def identifier(value: Any) -> str | None:
    # CSV preserves lexical IDs; Parquet integer IDs match their CSV strings.
    # Never trim, case-fold, or coerce floating point IDs to integers.
    return None if is_blank(value) else str(value)


@dataclass
class TextGroup:
    rows: int = 0
    prompts: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)


class Accumulator:
    def __init__(self, name: str, labels: list[str]):
        self.name = name
        self.labels = labels
        self.allowed = set(labels)
        self.rows = 0
        self.fields = {key: Counter() for key in EXPECTED[name]}
        self.ids: Counter[str] = Counter()
        self.label_counts: Counter[str] = Counter()
        self.unknown_labels: set[str] = set()
        self.missing_labels = 0
        self.unknown_label_rows = 0
        self.prompt_rows: Counter[str] = Counter()
        self.prompt_labels: dict[str, Counter[str]] = defaultdict(Counter)
        self.prompt_unknown: set[str] = set()
        self.text_groups: dict[str, dict[bytes, TextGroup]] = {
            "exact": {}, "whitespace_normalized": {}
        }

    def add(self, row: dict[str, Any]) -> None:
        self.rows += 1
        for key, counts in self.fields.items():
            value = row.get(key)
            if value is None:
                counts["null_or_missing"] += 1
            elif value == "":
                counts["empty_string"] += 1
            elif isinstance(value, str) and not value.strip():
                counts["whitespace_only"] += 1
            if value is not None and not isinstance(value, str):
                counts["non_string"] += 1

        row_id = identifier(row.get("id"))
        if row_id is not None:
            self.ids[row_id] += 1
        label = row.get("model")
        if "model" in self.fields:
            if is_blank(label):
                self.missing_labels += 1
            elif isinstance(label, str) and label in self.allowed:
                self.label_counts[label] += 1
            else:
                self.unknown_label_rows += 1
                self.unknown_labels.add(str(label))

        prompt = identifier(row.get("prompt_id"))
        if self.name == "train.csv" and prompt is not None:
            self.prompt_rows[prompt] += 1
            if not is_blank(label):
                self.prompt_labels[prompt][str(label)] += 1
                if not isinstance(label, str) or label not in self.allowed:
                    self.prompt_unknown.add(prompt)

        text = row.get("text")
        if isinstance(text, str) and text.strip():
            for mode, value in (("exact", text),
                                ("whitespace_normalized", " ".join(text.split()))):
                key = hashlib.sha256(value.encode("utf-8")).digest()
                group = self.text_groups[mode].setdefault(key, TextGroup())
                group.rows += 1
                if prompt is not None:
                    group.prompts.add(prompt)
                if not is_blank(label):
                    group.labels.add(str(label))

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rows": self.rows,
            "required_field_quality": {
                key: {metric: counts[metric] for metric in
                      ("null_or_missing", "empty_string", "whitespace_only", "non_string")}
                for key, counts in self.fields.items()
            },
            "ids": {
                "nonblank_unique": len(self.ids),
                "duplicate_groups": sum(count > 1 for count in self.ids.values()),
                "rows_in_duplicate_groups": sum(count for count in self.ids.values() if count > 1),
                "extra_duplicate_rows": sum(count - 1 for count in self.ids.values() if count > 1),
            },
        }
        if "model" in self.fields:
            result["labels"] = {
                "official_class_counts": {label: self.label_counts[label] for label in self.labels},
                "missing_or_blank_rows": self.missing_labels,
                "unknown_nonblank_rows": self.unknown_label_rows,
                "unknown_distinct_values": len(self.unknown_labels),
                "reference_labels_absent": [label for label in self.labels if not self.label_counts[label]],
                "matches_reference_label_set": (not self.unknown_labels and
                                                set(self.label_counts) == self.allowed),
                "unknown_values_suppressed": True,
            }
        if "text" in self.fields:
            result["text_duplicates"] = {
                mode: summarize_text_groups(groups) for mode, groups in self.text_groups.items()
            }
        if self.name == "train.csv":
            covered = {prompt: set(self.prompt_labels[prompt]) & self.allowed
                       for prompt in self.prompt_rows}
            pair_counts = [count for counter in self.prompt_labels.values()
                           for count in counter.values()]
            result["prompt_coverage"] = {
                "nonblank_unique_prompts": len(self.prompt_rows),
                "rows_per_prompt_histogram": histogram(self.prompt_rows.values()),
                "official_classes_per_prompt_histogram": histogram(map(len, covered.values())),
                "prompts_missing_any_reference_class": sum(len(values) < len(self.allowed)
                                                            for values in covered.values()),
                "missing_class_prompt_counts": {
                    label: sum(label not in values for values in covered.values())
                    for label in self.labels
                },
                "prompts_with_unknown_labels": len(self.prompt_unknown),
                "prompt_label_pairs_with_multiple_rows": sum(count > 1 for count in pair_counts),
                "extra_rows_for_repeated_prompt_label_pairs": sum(count - 1 for count in pair_counts if count > 1),
                "prompt_ids_suppressed": True,
            }
        return result


def histogram(values: Any) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items())}


def summarize_text_groups(groups: dict[bytes, TextGroup]) -> dict[str, int]:
    duplicated = [group for group in groups.values() if group.rows > 1]
    conflicts = [group for group in duplicated if len(group.labels) > 1]
    cross_prompt = [group for group in duplicated if len(group.prompts) > 1]
    return {
        "nonblank_string_rows": sum(group.rows for group in groups.values()),
        "unique_text_groups": len(groups),
        "duplicate_groups": len(duplicated),
        "rows_in_duplicate_groups": sum(group.rows for group in duplicated),
        "extra_duplicate_rows": sum(group.rows - 1 for group in duplicated),
        "groups_with_label_conflicts": len(conflicts),
        "rows_in_label_conflict_groups": sum(group.rows for group in conflicts),
        "cross_prompt_duplicate_groups": len(cross_prompt),
        "rows_in_cross_prompt_groups": sum(group.rows for group in cross_prompt),
        "cross_prompt_groups_with_label_conflicts": sum(len(group.labels) > 1 for group in cross_prompt),
    }


def read_input(path: Path, labels: list[str]) -> tuple[dict[str, Any], Accumulator]:
    accumulator = Accumulator(path.name, labels)
    before = path.stat()
    digest = sha256_file(path)
    malformed: Counter[str] = Counter()
    if path.suffix == ".csv":
        # Permit complete long responses; csv's default field limit is too small.
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                break
            except OverflowError:
                limit //= 10
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                columns = next(reader, [])
                for values in reader:
                    if len(values) < len(columns):
                        malformed["rows_with_missing_fields"] += 1
                    if len(values) > len(columns):
                        malformed["rows_with_extra_fields"] += 1
                    # First occurrence is used if a duplicate header exists;
                    # duplicate headers are separately flagged as integrity errors.
                    row: dict[str, Any] = {}
                    for index, key in enumerate(columns):
                        if key not in row:
                            row[key] = values[index] if index < len(values) else None
                    accumulator.add(row)
        except (UnicodeError, csv.Error) as exc:
            raise AuditInputError(f"Cannot fully parse {path.name}; check encoding and CSV structure.") from exc
        schema = [{"name": name, "type": "CSV text"} for name in columns]
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise AuditInputError("pyarrow is required; use an environment containing the audit dependency.") from exc
        try:
            parquet = pq.ParquetFile(path)
            columns = parquet.schema_arrow.names
            schema = [{"name": item.name, "type": str(item.type)} for item in parquet.schema_arrow]
            for batch in parquet.iter_batches(batch_size=65536):
                for row in batch.to_pylist():
                    accumulator.add(row)
        except Exception as exc:
            # Arrow diagnostics can contain source values: do not echo them.
            raise AuditInputError(f"Cannot fully parse {path.name}; check Parquet structure.") from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AuditInputError(f"Input changed while auditing {path.name}; retry on stable files.")
    summary = accumulator.summary()
    summary.update({
        "file": path.name,
        "size_bytes": before.st_size,
        "sha256": digest,
        "schema": schema,
        "missing_required_columns": [key for key in EXPECTED[path.name] if key not in columns],
        "extra_columns": [key for key in columns if key not in EXPECTED[path.name]],
        "duplicate_column_names": sorted(key for key, count in Counter(columns).items() if count > 1),
        "csv_structure": {key: malformed[key] for key in
                          ("rows_with_missing_fields", "rows_with_extra_fields")}
                         if path.suffix == ".csv" else None,
    })
    return summary, accumulator


def build_report(data_dir: Path, labels_path: Path) -> dict[str, Any]:
    paths = [data_dir / name for name in EXPECTED]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise AuditInputError("Missing required input files: " + ", ".join(missing))
    try:
        reference = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = reference["labels"]
        if (not isinstance(labels, list) or len(labels) != 18
                or not all(isinstance(label, str) and label.strip() == label and label for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AuditInputError("Label configuration must contain exactly 18 unique nonblank string labels.") from exc
    reports: dict[str, dict[str, Any]] = {}
    accumulators: dict[str, Accumulator] = {}
    for path in paths:
        try:
            reports[path.name], accumulators[path.name] = read_input(path, labels)
        except OSError as exc:
            raise AuditInputError(f"Cannot read required input file: {path.name}") from exc
    train = accumulators["train.csv"]
    test = accumulators["test.csv"]
    sample = accumulators["sample_submission.parquet"]
    overlap = {}
    for mode in train.text_groups:
        shared = train.text_groups[mode].keys() & test.text_groups[mode].keys()
        overlap[mode] = {
            "shared_text_groups": len(shared),
            "train_rows_in_shared_groups": sum(train.text_groups[mode][key].rows for key in shared),
            "test_rows_in_shared_groups": sum(test.text_groups[mode][key].rows for key in shared),
        }
    test_ids, sample_ids = set(test.ids), set(sample.ids)
    coverage = {
        "test_unique_nonblank_ids": len(test_ids),
        "sample_unique_nonblank_ids": len(sample_ids),
        "missing_test_ids": len(test_ids - sample_ids),
        "extra_sample_ids": len(sample_ids - test_ids),
        "same_unique_nonblank_id_set": test_ids == sample_ids,
        "exactly_one_sample_row_per_test_id": (
            bool(test_ids) and test_ids == sample_ids
            and all(count == 1 for count in test.ids.values())
            and all(count == 1 for count in sample.ids.values())
            and test.rows == len(test_ids) and sample.rows == len(sample_ids)
        ),
    }
    findings = collect_findings(reports, overlap, coverage)
    return {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "review_required" if findings else "no_findings",
        "error_finding_count": sum(item["severity"] == "error" for item in findings),
        "scope": "Complete supplied local files; public test data cannot establish hidden-test quality or model accuracy.",
        "privacy": "Aggregate counts only; source text, row IDs, prompt IDs, unknown label values, and text hashes are omitted.",
        "definitions": {
            "null_or_missing": "CSV absent fields or Parquet nulls; CSV empty strings are counted separately.",
            "whitespace_only": "Nonempty strings containing only Unicode whitespace; exclusive of empty strings.",
            "text_groups": "Nonblank strings grouped by SHA-256 of UTF-8 text; no text digests are emitted.",
            "whitespace_normalized": "Unicode whitespace collapsed to one ASCII space and trimmed; case and punctuation preserved.",
            "id_comparison": "Exact string rendering, preserving CSV lexical values; no whitespace trimming or numeric rounding.",
            "prompt_coverage": "Aggregated over nonblank prompt IDs, with no per-prompt identifiers emitted.",
        },
        "label_reference": {**reference, "sha256": sha256_file(labels_path)},
        "files": reports,
        "train_test_text_overlap": overlap,
        "train_test_shared_nonblank_ids": len(set(train.ids) & test_ids),
        "sample_id_coverage": coverage,
        "findings": findings,
    }


def collect_findings(reports: dict[str, dict[str, Any]], overlap: dict[str, Any],
                     coverage: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []

    def add(code: str, severity: str, file: str, count: int) -> None:
        if count:
            findings.append({"code": code, "severity": severity, "file": file, "count": count})

    for name, report in reports.items():
        add("missing_required_columns", "error", name, len(report["missing_required_columns"]))
        add("duplicate_column_names", "error", name, len(report["duplicate_column_names"]))
        add("empty_input_file", "error", name, int(report["rows"] == 0))
        for code, count in (report["csv_structure"] or {}).items():
            add(code, "error", name, count)
        for key, counts in report["required_field_quality"].items():
            add(f"blank_required_field:{key}", "error", name,
                sum(counts[metric] for metric in ("null_or_missing", "empty_string", "whitespace_only")))
            if key in {"text", "model"}:
                add(f"non_string_field:{key}", "error", name, counts["non_string"])
        add("duplicate_ids", "error", name, report["ids"]["extra_duplicate_rows"])
        if "labels" in report:
            add("labels_outside_prior_official_reference_review_required", "review", name,
                report["labels"]["unknown_nonblank_rows"])
            if name == "train.csv":
                add("reference_classes_absent_from_training_review_required", "review", name,
                    len(report["labels"]["reference_labels_absent"]))
        for mode, counts in report.get("text_duplicates", {}).items():
            add(f"duplicate_text:{mode}", "review", name, counts["duplicate_groups"])
            add(f"text_label_conflict:{mode}", "review", name, counts["groups_with_label_conflicts"])
            add(f"cross_prompt_text_duplicate:{mode}", "review", name, counts["cross_prompt_duplicate_groups"])
    for mode, counts in overlap.items():
        add(f"train_test_text_overlap:{mode}", "review", "train.csv + test.csv", counts["shared_text_groups"])
    add("sample_id_coverage_mismatch", "error", "sample_submission.parquet",
        int(not coverage["exactly_one_sample_row_per_test_id"]))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--labels", type=Path, default=PROJECT / "config" / "labels.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports" / "local_data_profile.json")
    args = parser.parse_args(argv)
    # Prevent accidental replacement of an official input or the label reference.
    protected = [(args.data_dir / name).resolve() for name in EXPECTED] + [args.labels.resolve()]
    if args.output.resolve() in protected:
        parser.exit(2, "error: output must not overwrite an input file or label configuration.\n")
    try:
        report = build_report(args.data_dir, args.labels)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except AuditInputError as exc:
        parser.exit(2, f"error: {exc}\n")
    except OSError:
        parser.exit(2, "error: could not write audit report.\n")
    print(json.dumps({"audit_status": report["audit_status"],
                      "rows": {name: item["rows"] for name, item in report["files"].items()},
                      "error_findings": report["error_finding_count"],
                      "report": str(args.output)}, ensure_ascii=False))
    return 1 if report["error_finding_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
