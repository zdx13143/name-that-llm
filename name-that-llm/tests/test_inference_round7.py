import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inference_round7 import digest, read_test, verify_assets


class InferenceContractTests(unittest.TestCase):
    def test_test_rows_preserve_ids_and_reject_label_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            path.write_text('id,text\nb,"second, text"\na,first\n', encoding="utf-8")
            self.assertEqual(read_test(path), (["b", "a"], ["second, text", "first"]))
            path.write_text("id,text,model\na,first,forbidden\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns only"):
                read_test(path)

    def test_duplicate_ids_and_row_limit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            path.write_text("id,text\na,one\na,two\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_test(path)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "text"])
                writer.writerows((str(i), "artificial") for i in range(4097))
            with self.assertRaises(ValueError):
                read_test(path)

    def test_manifest_anchor_and_extra_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"candidate": "windows-external-weighting-requested-final-20261026-r7",
                        "official_weight": 4, "development_gate_passed": False, "files": {}}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            expected = digest(path)
            self.assertEqual(verify_assets(root, expected), manifest)
            with self.assertRaisesRegex(ValueError, "Manifest identity"):
                verify_assets(root, "changed")
            (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set"):
                verify_assets(root, expected)


if __name__ == "__main__":
    unittest.main()
