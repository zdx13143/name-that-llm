"""No network, Torch, competition data or real model required."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_modernbert as probe


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest = []
        for name in probe.FILES:
            data = ("synthetic-" + name).encode()
            (self.directory / name).write_bytes(data)
            item = {"rfilename": name, "size": len(data)}
            if name == "model.safetensors":
                item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
            else:
                item["blobId"] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            self.manifest.append(item)

    def test_verifies_both_git_blob_and_lfs_identity(self):
        self.assertEqual(len(probe.validate_files(self.directory, self.manifest)), 6)

    def test_same_size_tampering_is_rejected_for_tokenizer_and_weights(self):
        for name in ("tokenizer.json", "model.safetensors"):
            with self.subTest(name=name):
                path = self.directory / name
                original = path.read_bytes()
                path.write_bytes(b"x" * len(original))
                with self.assertRaises(ValueError):
                    probe.validate_files(self.directory, self.manifest)
                path.write_bytes(original)

    def test_incomplete_file_set_is_rejected(self):
        with self.assertRaises(ValueError):
            probe.validate_files(self.directory, self.manifest[:-1])

    def test_extra_loader_file_is_rejected(self):
        (self.directory / "added_tokens.json").write_text("{}")
        with self.assertRaises(ValueError):
            probe.validate_files(self.directory, self.manifest)

    def test_receipt_cannot_supply_changed_tokenizer_identity(self):
        output = self.directory / "output"
        claim = self.directory / "claims/synthetic.json"
        config = {"revision": "fixed", "model_id": "synthetic", "model_files": self.manifest}
        changed = json.loads(json.dumps(self.manifest))
        changed[0]["blobId"] = "changed"
        probe.write_json(output / "modernbert_files.json", {"revision": "fixed", "model_id": "synthetic", "upstream_manifest": changed})
        with patch.object(probe, "OUTPUT", output), patch.object(probe, "CLAIM", claim):
            with self.assertRaisesRegex(ValueError, "preregistered file identities"):
                probe.smoke(config)
        self.assertEqual(json.loads(claim.read_bytes())["attempted_forward_calls"], 0)

    def test_existing_receipt_is_not_overwritten(self):
        path = self.directory / "receipt.json"
        probe.write_json(path, {"state": "failed"})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            probe.write_json(path, {"state": "completed"})
        self.assertEqual(before, path.read_bytes())

    def test_missing_files_consume_claim_and_block_repeat_before_torch_import(self):
        output = self.directory / "output"
        claim = self.directory / "claims/synthetic.json"
        with patch.object(probe, "OUTPUT", output), patch.object(probe, "CLAIM", claim):
            with self.assertRaises(FileNotFoundError):
                probe.smoke({"synthetic": "test"})
            first = claim.read_bytes()
            self.assertEqual(json.loads(first)["state"], "failed")
            self.assertEqual(json.loads(first)["attempted_forward_calls"], 0)
            with self.assertRaises(FileExistsError):
                probe.smoke({"synthetic": "test"})
            self.assertEqual(first, claim.read_bytes())


if __name__ == "__main__":
    unittest.main()
