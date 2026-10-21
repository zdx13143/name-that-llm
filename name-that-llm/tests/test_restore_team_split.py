"""Metadata-only restoration safeguards; no real data or model fitting."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / "scripts"))
import restore_team_split as restore


class SplitRestorationTests(unittest.TestCase):
    def test_wrong_training_hash_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory(prefix="ntl-restore-test-") as folder:
            folder = Path(folder)
            train = folder / "train.csv"
            train.write_text("id,prompt_id,text,model\nsynthetic,prompt,unused,unused\n")
            output = folder / "outputs" / "split_manifest.json"
            with patch.object(restore, "DESTINATION", output):
                with self.assertRaisesRegex(ValueError, "training file SHA-256 mismatch"):
                    restore.restore_split(train)
            self.assertFalse(output.exists())

    def test_bad_reconstruction_and_conflicting_existing_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ntl-restore-test-") as folder:
            destination = Path(folder) / "split_manifest.json"
            payload = b'{"synthetic_metadata": true}\n'
            digest = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ValueError, "Reconstructed manifest SHA-256"):
                restore._persist_verified_manifest(destination, payload, "incorrect-hash")
            self.assertFalse(destination.exists())
            original = b'{"different_existing_metadata": true}\n'
            destination.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                restore._persist_verified_manifest(destination, payload, digest)
            self.assertEqual(destination.read_bytes(), original)

    def test_identical_manifest_is_verified_without_rewriting(self):
        with tempfile.TemporaryDirectory(prefix="ntl-restore-test-") as folder:
            destination = Path(folder) / "split_manifest.json"
            payload = b'{"synthetic_metadata": true}\n'
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(restore._persist_verified_manifest(destination, payload, digest), "created")
            original_stat = destination.stat()
            self.assertEqual(restore._persist_verified_manifest(destination, payload, digest), "verified_existing")
            current_stat = destination.stat()
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(current_stat.st_ino, original_stat.st_ino)
            self.assertEqual(current_stat.st_mtime_ns, original_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
