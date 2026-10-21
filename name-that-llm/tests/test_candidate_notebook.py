"""Synthetic checks for the fixed submission candidate; no competition fitting."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_candidate_notebook as builder
from compare_prompt_centering import cluster_batch, feature_pair, soft_center
from compare_team_cpu import canonical_probabilities
from model_cpu import file_hash
from model_cpu_round2 import make_pipeline


def load_runtime():
    document = json.loads((ROOT / "notebooks/candidate_blend_offline.ipynb").read_text())
    namespace = {"__name__": "synthetic_candidate_notebook"}
    assert "run_candidate()" in "".join(document["cells"][-1]["source"])
    with redirect_stdout(io.StringIO()):
        for cell in document["cells"][:-1]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), "candidate_notebook", "exec"), namespace)
    return document, namespace


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fixture(namespace, folder, rows=7):
    labels = namespace["LABELS"]
    train = [{"id": f"train-{index}", "prompt_id": str(index // 18),
              "text": f"Synthetic class{index % 18} answer. Item{index % 18}! Topic{index // 18}.",
              "model": labels[index % 18]} for index in range(54)]
    test = [{"id": f"test-{rows-index:04d}",
             "text": f"Synthetic class{index % 18} answer. Item{index % 18}! New topic."}
            for index in range(rows)]
    write_csv(folder / "train.csv", ["id", "prompt_id", "text", "model"], train)
    write_csv(folder / "test.csv", ["id", "text"], test)
    namespace.update(INPUT=folder, OUTPUT=folder / "result",
                     EXPECTED_TRAIN_SHA256=file_hash(folder / "train.csv"),
                     EXPECTED_TRAIN_ROWS=len(train), EXPECTED_TRAIN_GROUPS=3)
    return train, test


class CandidateNotebookTests(unittest.TestCase):
    def test_clean_clone_builder_and_frozen_source_identity(self):
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-build-") as temporary:
            clone = Path(temporary) / "module"
            for relative in builder.PINNED:
                destination = clone / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            with patch.object(builder, "ROOT", clone):
                document = builder.build(clone / "candidate.ipynb")
            self.assertFalse((clone / "outputs").exists())
            committed, namespace = load_runtime()
            self.assertEqual(document, committed)
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    ast.parse("".join(cell["source"]))
            self.assertEqual(namespace["MODEL_SOURCES"]["model_cpu.py"],
                             (ROOT / "scripts/model_cpu.py").read_text())
            self.assertEqual(namespace["MODEL_SOURCES"]["model_cpu_round2.py"],
                             (ROOT / "scripts/model_cpu_round2.py").read_text())
            center_config = json.loads((ROOT / "config/prompt_centering_20261021.json").read_text())
            original_config = json.loads((ROOT / "config/team_dev_20261021.json").read_text())["original_config"]
            self.assertEqual(namespace["CENTER_CONFIG"], center_config)
            self.assertEqual(namespace["ORIGINAL_CONFIG"], original_config)
            self.assertEqual(namespace["EXPECTED_TRAIN_SHA256"], center_config["train_sha256"])
            # Embedded helpers are exact definitions, not similarly named rewrites.
            embedded = "".join(document["cells"][3]["source"])
            for filename, names in (
                    ("compare_prompt_centering.py", ("soft_center", "cluster_batch", "feature_pair")),
                    ("compare_team_cpu.py", ("validate_probabilities", "canonical_probabilities"))):
                for name in names:
                    self.assertIn(builder.extract_functions(ROOT / "scripts" / filename, (name,)).strip(), embedded)

    def test_builder_refuses_source_and_configuration_drift(self):
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-drift-") as temporary:
            output = Path(temporary) / "rejected.ipynb"
            for target in ("scripts/model_cpu_round2.py", "config/prompt_centering_20261021.json",
                           "config/labels.json"):
                real_digest = builder.digest
                def altered_digest(path):
                    return "0" * 64 if Path(path) == ROOT / target else real_digest(path)
                with self.subTest(target=target), patch.object(builder, "digest", side_effect=altered_digest):
                    with self.assertRaisesRegex(ValueError, "Frozen source/config drift"):
                        builder.build(output)
                self.assertFalse(output.exists())

    def test_seven_row_inference_matches_independent_probability_formula(self):
        _, namespace = load_runtime()
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-synthetic-") as temporary:
            folder = Path(temporary)
            train, test = fixture(namespace, folder)
            with redirect_stdout(io.StringIO()):
                runtime = namespace["run_candidate"]()
            texts = np.asarray([row["text"] for row in train])
            test_texts = np.asarray([row["text"] for row in test])
            ids = np.asarray([row["id"] for row in test])
            groups = np.asarray([row["prompt_id"] for row in train])
            truth = np.asarray([namespace["LABELS"].index(row["model"]) for row in train])
            config = namespace["CENTER_CONFIG"]
            with threadpool_limits(limits=2):
                original = make_pipeline(namespace["ORIGINAL_CONFIG"], seed=20261018).fit(texts, truth)
                original_probs = canonical_probabilities(original.predict_proba(test_texts), 7, 18)
                a, b = feature_pair(texts, test_texts, config)
                a = soft_center(a, groups, .5, config["max_centered_csr_bytes"])
                classifier = LogisticRegression(C=16., class_weight="balanced", max_iter=3000,
                                                random_state=20261021).fit(a, truth)
                test_groups = cluster_batch(test_texts, ids, .65, config["cluster_max_features"])
                b = soft_center(b, test_groups, .5, config["max_centered_csr_bytes"])
                partial_probs = canonical_probabilities(classifier.predict_proba(b), 7, 18)
            expected = canonical_probabilities(.75 * original_probs + .25 * partial_probs, 7, 18)
            with np.load(folder / "result/probabilities.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["ids"], ids)
                np.testing.assert_array_equal(saved["labels"], namespace["LABELS"])
                np.testing.assert_allclose(saved["original_style"], original_probs, rtol=0, atol=1e-14)
                np.testing.assert_allclose(saved["partial"], partial_probs, rtol=0, atol=1e-14)
                np.testing.assert_allclose(saved["blend"], expected, rtol=0, atol=1e-14)
            submission = pq.read_table(folder / "result/submission.parquet").to_pydict()
            self.assertEqual(submission["id"], ids.tolist())
            self.assertEqual(submission["model"], [namespace["LABELS"][i] for i in expected.argmax(axis=1)])
            self.assertEqual(runtime["test_rows"], 7)
            self.assertEqual(runtime["full_training_classifier_fits"], 2)
            self.assertEqual(runtime["final_holdout_evaluations"], 0)
            self.assertFalse(runtime["formal_submission"])
            self.assertFalse(runtime["is_three_row_interface_example"])
            before = {path.name: file_hash(path) for path in (folder / "result").iterdir()}
            with self.assertRaises(FileExistsError):
                namespace["run_candidate"]()
            self.assertEqual(before, {path.name: file_hash(path) for path in (folder / "result").iterdir()})

    def test_official_train_hash_checked_before_any_fit_or_claim(self):
        _, namespace = load_runtime()
        official_hash = namespace["EXPECTED_TRAIN_SHA256"]
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-hash-") as temporary:
            folder = Path(temporary)
            fixture(namespace, folder)
            namespace["EXPECTED_TRAIN_SHA256"] = official_hash
            with self.assertRaisesRegex(ValueError, "Training hash"):
                namespace["run_candidate"]()
            self.assertFalse((folder / "result").exists())

    def test_bad_schema_ids_text_and_labels_rejected_before_fit(self):
        _, namespace = load_runtime()
        mutations = ("missing_test_text", "extra_test_label", "duplicate_test_id", "blank_test_text", "duplicate_train_id",
                     "missing_train_model", "unknown_label", "blank_prompt")
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-schema-") as temporary:
            folder = Path(temporary)
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    train, test = fixture(namespace, folder)
                    train_columns = ["id", "prompt_id", "text", "model"]
                    test_columns = ["id", "text"]
                    if mutation == "missing_test_text":
                        test_columns.remove("text")
                        for row in test:
                            row.pop("text")
                    elif mutation == "extra_test_label":
                        test_columns.append("model")
                        for row in test:
                            row["model"] = "unexpected-test-label"
                    elif mutation == "duplicate_test_id":
                        test[1]["id"] = test[0]["id"]
                    elif mutation == "blank_test_text":
                        test[0]["text"] = "   "
                    elif mutation == "duplicate_train_id":
                        train[1]["id"] = train[0]["id"]
                    elif mutation == "missing_train_model":
                        train_columns.remove("model")
                        for row in train:
                            row.pop("model")
                    elif mutation == "unknown_label":
                        train[0]["model"] = "unknown"
                    elif mutation == "blank_prompt":
                        train[0]["prompt_id"] = " "
                    write_csv(folder / "train.csv", train_columns, train)
                    write_csv(folder / "test.csv", test_columns, test)
                    # Fixture-only hash substitution reaches the schema checks.
                    namespace["EXPECTED_TRAIN_SHA256"] = file_hash(folder / "train.csv")
                    with self.assertRaises(ValueError):
                        namespace["run_candidate"]()
                    self.assertFalse((folder / "result").exists())

    def test_existing_claim_or_output_never_overwritten(self):
        _, namespace = load_runtime()
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-existing-") as temporary:
            folder = Path(temporary)
            fixture(namespace, folder)
            for name in ("candidate_attempt.json", "submission.parquet", "probabilities.npz", "runtime.json"):
                with self.subTest(name=name):
                    output = folder / name.replace(".", "-")
                    output.mkdir()
                    namespace["OUTPUT"] = output
                    sentinel = output / name
                    sentinel.write_text("retain this prior artifact")
                    with self.assertRaises(FileExistsError):
                        namespace["run_candidate"]()
                    self.assertEqual(sentinel.read_text(), "retain this prior artifact")
                    self.assertEqual(list(output.iterdir()), [sentinel])

    def test_failed_fit_is_persisted_and_cannot_silently_restart(self):
        _, namespace = load_runtime()
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-failure-") as temporary:
            folder = Path(temporary)
            fixture(namespace, folder)
            def fail_fit(*args, **kwargs):
                raise RuntimeError("synthetic deliberate failure")
            with patch.dict(namespace, {"make_pipeline": lambda *a, **kw: types.SimpleNamespace(fit=fail_fit)}):
                with self.assertRaisesRegex(RuntimeError, "synthetic deliberate failure"):
                    namespace["run_candidate"]()
            claim = json.loads((folder / "result/candidate_attempt.json").read_text())
            self.assertEqual(claim["state"], "failed")
            self.assertEqual(claim["attempted_fits"], 1)
            self.assertEqual(claim["completed_fits"], 0)
            with self.assertRaises(FileExistsError):
                namespace["run_candidate"]()

    def test_class_order_and_probability_contracts(self):
        _, namespace = load_runtime()
        namespace["assert_classes"](types.SimpleNamespace(classes_=np.arange(18)))
        for order in (np.arange(18)[::-1], np.arange(17)):
            with self.assertRaisesRegex(ValueError, "class order"):
                namespace["assert_classes"](types.SimpleNamespace(classes_=order))
        for values in (np.full((2, 18), np.nan), np.zeros((2, 18)),
                       np.full((2, 17), 1/17), np.full((2, 18), -.1)):
            with self.assertRaises(ValueError):
                namespace["canonical_probabilities"](values, 2, 18)

    def test_clustering_admission_precedes_fit_and_does_not_assume_three_rows(self):
        _, namespace = load_runtime()
        with tempfile.TemporaryDirectory(prefix="ntl-candidate-batch-") as temporary:
            folder = Path(temporary)
            fixture(namespace, folder, rows=4096)
            _, test, _ = namespace["validate_inputs"]()
            self.assertEqual(len(test), 4096)
            fixture(namespace, folder, rows=4097)
            with self.assertRaisesRegex(MemoryError, "clustering admission"):
                namespace["run_candidate"]()
            self.assertFalse((folder / "result").exists())


if __name__ == "__main__":
    unittest.main()
