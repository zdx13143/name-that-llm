"""Synthetic tool tests only; these do not audit the real competition files."""

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "profile_data.py"
LABELS_FILE = PROJECT / "config" / "labels.json"
LABELS = json.loads(LABELS_FILE.read_text(encoding="utf-8"))["labels"]
SPEC = importlib.util.spec_from_file_location("ntl_profile_data", SCRIPT)
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


class ProfileDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ntl-synthetic-tool-test-")
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write_csv(self, name, columns, rows):
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def write_inputs(self, train, test, sample):
        self.write_csv("train.csv", ["id", "prompt_id", "text", "model"], train)
        self.write_csv("test.csv", ["id", "text"], test)
        pq.write_table(pa.table({"id": [row[0] for row in sample],
                                "model": [row[1] for row in sample]}),
                       self.root / "sample_submission.parquet")

    def complete_inputs(self, test_count=7):
        train = [[f"private-train-{prompt}-{index}", f"private-prompt-{prompt}",
                  f"Synthetic response {prompt}/{index}, with a quoted newline\nNext line.", label]
                 for prompt in range(2) for index, label in enumerate(LABELS)]
        test = [[str(index), f"Unseen synthetic response {index}"] for index in range(test_count)]
        # Integer Parquet IDs must match lexical integer IDs in the official CSV.
        sample = [[index, LABELS[index % len(LABELS)]] for index in range(test_count)]
        self.write_inputs(train, test, sample)

    def audit(self):
        return PROFILE.build_report(self.root, LABELS_FILE)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(self.root), *args],
                              capture_output=True, text=True, check=False)

    def test_full_counts_prompt_coverage_and_hashes_for_arbitrary_test_size(self):
        self.complete_inputs(test_count=7)
        report = self.audit()
        self.assertEqual(report["audit_status"], "no_findings")
        self.assertEqual(report["files"]["train.csv"]["rows"], 36)
        self.assertEqual(report["files"]["test.csv"]["rows"], 7)
        self.assertEqual(report["files"]["sample_submission.parquet"]["rows"], 7)
        prompts = report["files"]["train.csv"]["prompt_coverage"]
        self.assertEqual(prompts["nonblank_unique_prompts"], 2)
        self.assertEqual(prompts["official_classes_per_prompt_histogram"], {"18": 2})
        self.assertEqual(prompts["rows_per_prompt_histogram"], {"18": 2})
        self.assertTrue(report["sample_id_coverage"]["exactly_one_sample_row_per_test_id"])
        expected = hashlib.sha256((self.root / "train.csv").read_bytes()).hexdigest()
        self.assertEqual(report["files"]["train.csv"]["sha256"], expected)
        self.assertTrue(report["files"]["train.csv"]["labels"]["matches_reference_label_set"])

    def test_duplicate_leakage_conflicts_coverage_and_privacy(self):
        self.write_inputs(
            [["private-id-a", "private-prompt-a", "same  text", LABELS[0]],
             ["private-id-a", "private-prompt-b", "same  text", LABELS[1]],
             ["private-id-c", "private-prompt-c", " same\ttext ", LABELS[0]],
             ["private-id-d", "", " \t ", ""],
             ["private-id-e", "private-prompt-a", "private-text-sentinel", "private-unknown-label"]],
            [["private-test-a", "same text"], ["private-test-b", "other response"]],
            [["private-test-a", LABELS[0]], ["private-test-a", LABELS[1]],
             ["private-extra-id", "private-unknown-label"]],
        )
        report = self.audit()
        train = report["files"]["train.csv"]
        self.assertEqual(train["ids"]["extra_duplicate_rows"], 1)
        exact = train["text_duplicates"]["exact"]
        normalized = train["text_duplicates"]["whitespace_normalized"]
        self.assertEqual(exact["duplicate_groups"], 1)
        self.assertEqual(exact["rows_in_duplicate_groups"], 2)
        self.assertEqual(exact["cross_prompt_groups_with_label_conflicts"], 1)
        self.assertEqual(normalized["rows_in_duplicate_groups"], 3)
        self.assertEqual(normalized["groups_with_label_conflicts"], 1)
        self.assertEqual(report["train_test_text_overlap"]["exact"]["shared_text_groups"], 0)
        self.assertEqual(report["train_test_text_overlap"]["whitespace_normalized"]["train_rows_in_shared_groups"], 3)
        self.assertEqual(train["required_field_quality"]["text"]["whitespace_only"], 1)
        self.assertEqual(train["required_field_quality"]["prompt_id"]["empty_string"], 1)
        self.assertEqual(train["labels"]["unknown_nonblank_rows"], 1)
        self.assertEqual(train["labels"]["missing_or_blank_rows"], 1)
        coverage = report["sample_id_coverage"]
        self.assertEqual(coverage["missing_test_ids"], 1)
        self.assertEqual(coverage["extra_sample_ids"], 1)
        self.assertFalse(coverage["exactly_one_sample_row_per_test_id"])
        serialized = json.dumps(report)
        for secret in ("private-id-a", "private-prompt-a", "private-text-sentinel",
                       "private-test-a", "private-extra-id", "private-unknown-label", "same  text"):
            self.assertNotIn(secret, serialized)
        self.assertGreater(report["error_finding_count"], 0)

    def test_missing_files_fail_without_creating_a_report(self):
        output = self.root / "local_profile.json"
        result = self.cli("--output", str(output))
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing required input files", result.stderr)
        self.assertFalse(output.exists())

    def test_missing_columns_and_malformed_rows_are_reported(self):
        self.complete_inputs()
        self.write_csv("train.csv", ["id", "prompt_id", "model"],
                       [["x", "p", LABELS[0], "private-extra-value"], ["y"]])
        report = self.audit()
        train = report["files"]["train.csv"]
        self.assertEqual(train["missing_required_columns"], ["text"])
        self.assertEqual(train["rows"], 2)
        self.assertEqual(train["csv_structure"], {
            "rows_with_missing_fields": 1, "rows_with_extra_fields": 1})
        self.assertEqual(train["required_field_quality"]["text"]["null_or_missing"], 2)
        self.assertNotIn("private-extra-value", json.dumps(report))

    def test_null_and_blank_sample_ids_do_not_pass_coverage(self):
        self.complete_inputs(test_count=2)
        pq.write_table(pa.table({"id": ["0", "1", None, " "],
                                "model": [LABELS[0], LABELS[1], None, ""]}),
                       self.root / "sample_submission.parquet")
        report = self.audit()
        self.assertTrue(report["sample_id_coverage"]["same_unique_nonblank_id_set"])
        self.assertFalse(report["sample_id_coverage"]["exactly_one_sample_row_per_test_id"])
        quality = report["files"]["sample_submission.parquet"]["required_field_quality"]
        self.assertEqual(quality["id"]["null_or_missing"], 1)
        self.assertEqual(quality["id"]["whitespace_only"], 1)
        self.assertEqual(quality["model"]["null_or_missing"], 1)
        self.assertEqual(quality["model"]["empty_string"], 1)

    def test_long_multiline_text_is_read_completely_and_cli_writes_aggregate_report(self):
        self.complete_inputs(test_count=11)
        self.write_csv("test.csv", ["id", "text"],
                       [[str(index), ("private-long-text\n" * 10000) + str(index)] for index in range(11)])
        output = self.root / "reports" / "local_profile.json"
        result = self.cli("--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["files"]["test.csv"]["rows"], 11)
        self.assertEqual(report["files"]["test.csv"]["text_duplicates"]["exact"]["unique_text_groups"], 11)
        self.assertNotIn("private-long-text", output.read_text(encoding="utf-8"))

    def test_cli_cannot_overwrite_official_inputs(self):
        self.complete_inputs()
        source = self.root / "train.csv"
        before = source.read_bytes()
        result = self.cli("--output", str(source))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(source.read_bytes(), before)

    def test_cli_reports_integrity_errors_with_nonzero_exit_and_saved_findings(self):
        self.complete_inputs(test_count=2)
        self.write_csv("test.csv", ["id", "text"], [["0", " "], ["0", "different"]])
        output = self.root / "local_profile.json"
        result = self.cli("--output", str(output))
        self.assertEqual(result.returncode, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertGreater(report["error_finding_count"], 0)
        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("duplicate_ids", codes)
        self.assertIn("sample_id_coverage_mismatch", codes)


if __name__ == "__main__":
    unittest.main()
