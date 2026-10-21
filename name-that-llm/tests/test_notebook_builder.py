"""Execute the generated Notebook on synthetic files, including runtime JSON."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_cpu_notebook import build
from model_cpu import config_id, file_hash
from model_cpu_round2 import make_pipeline


class NotebookTests(unittest.TestCase):
    def test_builder_rejects_changed_selection_and_summary_before_export(self):
        with tempfile.TemporaryDirectory(prefix="ntl-notebook-binding-") as folder:
            folder = Path(folder)
            config = {"classifier": "svm", "C": 1.}
            summary = {"selected": {"id": "changed", "config": config}}
            (folder / "search_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "configuration identifier"):
                build(folder, folder / "rejected.ipynb")
            summary["selected"]["id"] = config_id(config)
            (folder / "search_summary.json").write_text(json.dumps(summary))
            (folder / "artifact_contract.json").write_text(json.dumps({"search_summary_sha256": "old"}))
            with self.assertRaisesRegex(ValueError, "summary hash"):
                build(folder, folder / "rejected.ipynb")
            self.assertFalse((folder / "rejected.ipynb").exists())

    def test_generated_style_notebook_executes_and_matches_local_model(self):
        with tempfile.TemporaryDirectory(prefix="ntl-notebook-synthetic-") as folder:
            folder = Path(folder)
            labels = json.loads((ROOT / "config" / "labels.json").read_text())["labels"]
            config = {"representation": "char", "classifier": "svm", "C": 1.,
                      "char_ngram": [1, 2], "min_df": 1, "style_weight": .25}
            train_rows = [[str(index), str(index // 18), f"Synthetic class {index % 18}. Example {index}!", labels[index % 18]]
                          for index in range(36)]
            test_rows = [[f"test-{index}", f"Synthetic class {index}. A new response!"] for index in range(7)]
            for name, columns, rows in (("train.csv", ["id", "prompt_id", "text", "model"], train_rows),
                                        ("test.csv", ["id", "text"], test_rows)):
                with (folder / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    writer.writerows(rows)
            summary = {"selected": {"id": config_id(config), "config": config}, "train_sha256": file_hash(folder / "train.csv"),
                       "label_reference_sha256": file_hash(ROOT / "config" / "labels.json"),
                       "code_sha256": {name: file_hash(ROOT / "scripts" / name)
                                       for name in ("model_cpu.py", "model_cpu_round2.py")},
                       "environment": {"python": "3.13.2", "scikit_learn": sklearn.__version__, "numpy": np.__version__}}
            (folder / "search_summary.json").write_text(json.dumps(summary))
            notebook = folder / "test.ipynb"
            build(folder, notebook)
            document = json.loads(notebook.read_text())
            namespace = {"__name__": "synthetic_notebook"}
            with patch.dict(os.environ, {"NTL_INPUT_DIR": str(folder), "NTL_OUTPUT_DIR": str(folder / "result")}):
                for cell in document["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell["outputs"], [])
                        self.assertIsNone(cell["execution_count"])
                        exec(compile("".join(cell["source"]), "generated_notebook", "exec"), namespace)
            runtime = json.loads((folder / "result" / "runtime.json").read_text())
            self.assertEqual(runtime["test_rows"], 7)
            model = make_pipeline(config)
            model.fit([row[2] for row in train_rows], [labels.index(row[3]) for row in train_rows])
            local_predictions = [labels[index] for index in model.predict([row[1] for row in test_rows])]
            notebook_predictions = pq.read_table(folder / "result" / "submission.parquet").column("model").to_pylist()
            self.assertEqual(notebook_predictions, local_predictions)


if __name__ == "__main__":
    unittest.main()
