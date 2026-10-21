"""Failure-path tests for immutable artifacts and the project-wide holdout claim."""
import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import artifact_guards as guards
import finalize_cpu as final
import train_cpu
import model_cpu
from model_cpu import config_id, file_hash


class ArtifactGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ntl-artifact-guards-")
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.run = self.root / "outputs" / "round1"
        (self.run / "trials").mkdir(parents=True)
        self.labels = [f"synthetic-label-{i}" for i in range(18)]
        self.labels_path = self.root / "labels.json"
        self.labels_path.write_text(json.dumps({"labels": self.labels}))
        with (self.data / "train.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "prompt_id", "text", "model"])
            for group in range(2):
                for index, label in enumerate(self.labels):
                    writer.writerow([f"id-{group}-{index}", f"prompt-{group}", f"synthetic-{group}-{index}", label])
        self.config = {"representation": "char", "classifier": "svm", "C": 1.}
        self.identity = config_id(self.config)
        self.manifest = {"train_sha256": file_hash(self.data / "train.csv"),
                         "development_indices": list(range(18)), "holdout_indices": list(range(18, 36))}
        (self.run / "split_manifest.json").write_text(json.dumps(self.manifest))
        self.oof = self.run / "trials" / f"{self.identity}-20261018-oof.npz"
        np.savez_compressed(self.oof, row_index=np.arange(18), truth=np.arange(18), prediction=np.arange(18))
        self.summary = {"seed": 20261018, "selected": {"id": self.identity, "config": self.config, "balanced_accuracy": 1.},
                        "train_sha256": self.manifest["train_sha256"],
                        "label_reference_sha256": file_hash(self.labels_path),
                        "split_sha256": file_hash(self.run / "split_manifest.json")}
        (self.run / "search_summary.json").write_text(json.dumps(self.summary))

    def tearDown(self):
        self.temporary.cleanup()

    def test_prior_labels_truth_and_oof_schema_are_checked(self):
        checked = guards.validate_prior(self.run, self.data, self.labels_path)
        self.assertEqual(checked["contract"]["oof_schema_sha256"], guards.object_hash(checked["contract"]["oof_schema"]))
        self.assertTrue((self.run / "artifact_contract.json").exists())
        np.savez_compressed(self.oof, row_index=np.arange(18), truth=np.roll(np.arange(18), 1), prediction=np.arange(18))
        with self.assertRaisesRegex(ValueError, "truth differs"):
            guards.validate_prior(self.run, self.data, self.labels_path)
        np.savez_compressed(self.oof, row_index=np.arange(18), truth=np.arange(18), prediction=np.arange(18).astype(float))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            guards.validate_prior(self.run, self.data, self.labels_path)
        self.labels_path.write_text(json.dumps({"labels": list(reversed(self.labels))}))
        with self.assertRaisesRegex(ValueError, "label reference hash"):
            guards.validate_prior(self.run, self.data, self.labels_path)

    def test_contract_detects_changed_schema_fingerprint(self):
        guards.validate_prior(self.run, self.data, self.labels_path)
        path = self.run / "artifact_contract.json"
        changed = json.loads(path.read_text())
        changed["training_schema_sha256"] = "altered"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "schema or artifact hash"):
            guards.validate_prior(self.run, self.data, self.labels_path)

    def test_shared_claim_blocks_other_parent_directory_and_legacy_search(self):
        other = self.root / "elsewhere" / "round2"
        other.mkdir(parents=True)
        with patch.object(final, "ROOT", self.root):
            claim, _ = final.claim_holdout(self.run, self.summary["train_sha256"], self.summary["split_sha256"])
            self.assertTrue(claim.exists())
            self.assertTrue((self.run / "holdout_opened.json").exists())
            with self.assertRaisesRegex(ValueError, "already claimed"):
                final.claim_holdout(other, self.summary["train_sha256"], "different-split-hash")
        with self.assertRaisesRegex(ValueError, "already claimed"):
            guards.assert_holdout_available(self.summary["train_sha256"], self.root)

    def test_marker_write_failure_leaves_durable_claim(self):
        other = self.root / "different-parent" / "round2"
        other.mkdir(parents=True)
        with patch.object(final, "ROOT", self.root):
            with patch.object(final, "save", side_effect=OSError("synthetic marker write failure")):
                with self.assertRaises(OSError):
                    final.claim_holdout(self.run, self.summary["train_sha256"], self.summary["split_sha256"])
            self.assertFalse((self.run / "holdout_opened.json").exists())
            with self.assertRaisesRegex(ValueError, "already claimed"):
                final.claim_holdout(other, self.summary["train_sha256"], self.summary["split_sha256"])

    def test_legacy_finalize_delegates_to_the_single_guarded_entry(self):
        args = argparse.Namespace(output=self.run, data_dir=self.data)
        with patch.object(final, "finalize_once", return_value="shared-entry") as delegated:
            self.assertEqual(train_cpu.finalize(args), "shared-entry")
            delegated.assert_called_once_with(args)

    def test_runtime_drift_is_rejected_before_holdout_claim(self):
        summary = {"environment": {"python": platform.python_version(),
                    "scikit_learn": sklearn.__version__, "numpy": np.__version__}}
        final.verify_runtime(summary)
        summary["environment"]["numpy"] = "unverified-version"
        with self.assertRaisesRegex(ValueError, "Runtime version differs"):
            final.verify_runtime(summary)
        self.assertFalse((self.root / "outputs" / "holdout_claims").exists())

    def test_failure_after_holdout_prediction_cannot_reopen_through_legacy_entry(self):
        (self.root / "scripts").mkdir()
        shutil.copy2(Path(model_cpu.__file__), self.root / "scripts" / "model_cpu.py")
        (self.root / "config").mkdir()
        shutil.copy2(self.labels_path, self.root / "config" / "labels.json")
        self.summary["code_sha256"] = {"model_cpu.py": file_hash(Path(model_cpu.__file__))}
        self.summary["environment"] = {"python": platform.python_version(),
            "scikit_learn": sklearn.__version__, "numpy": np.__version__}
        (self.run / "search_summary.json").write_text(json.dumps(self.summary))
        owner = self
        calls = []
        class SyntheticEstimator:
            def fit(self, x, y):
                owner.assertTrue((owner.run / "holdout_opened.json").exists())
                calls.append(len(x))
                if len(calls) == 2:
                    raise RuntimeError("synthetic full-train failure")
                return self
            def predict(self, x):
                return np.arange(len(x)) % 18
        args = argparse.Namespace(output=self.run, data_dir=self.data)
        with patch.object(final, "ROOT", self.root), patch.object(model_cpu, "make_pipeline", return_value=SyntheticEstimator()):
            with self.assertRaisesRegex(RuntimeError, "full-train failure"):
                final.finalize_once(args)
            self.assertTrue((self.run / "holdout_metrics.json").exists())
            claim = self.root / "outputs" / "holdout_claims" / f"{self.summary['train_sha256']}.json"
            self.assertEqual(json.loads(claim.read_text())["status"], "failed_after_claim")
            with self.assertRaisesRegex(ValueError, "already opened"):
                train_cpu.finalize(args)
        self.assertEqual(calls, [18, 36])


if __name__ == "__main__":
    unittest.main()
