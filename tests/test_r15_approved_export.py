"""Artificial data only; no raw source, actual quarantine, training or scoring."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_r15_dev_export import fixture
spec = importlib.util.spec_from_file_location('r15_approved', ROOT / 'scripts/r15_admit_approved_export.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ApprovedExportTests(unittest.TestCase):
    def setUp(self):
        self.data, self.protocol, self.blob, _, _, _ = fixture()
        self.receipt = {'schema': 'ntl-approved-raw-development-extraction-v1',
            'source_kind': 'approved_raw_csv_extraction', 'approval_reference': mod.APPROVAL,
            'source_sha256': mod.TRAIN_SHA256, 'raw_official_csv_opened': True,
            'legacy_preisolated_intake_compatible': False, 'csv_memory_parse_passes': 2,
            'source_rows': 2699, 'protected_rows': 540, 'protected_prompts': 30,
            'protected_record_parse_events': 1080, 'protected_rows_persisted': 0,
            'protected_holdout_evaluations': 0, 'training_fits': 0, 'training_admitted': False,
            'development_rows': 2159, 'development_prompts': 120,
            'development_sha256': mod.sha(self.blob)}
        for key in ('row_identity_sha256', 'text_sha256', 'fold_sha256', 'historical_split_sha256'):
            self.receipt[key] = self.protocol[key]
        self.policy = {'schema': 'ntl-approved-extraction-admission-policy-v1',
            'approval_reference': mod.APPROVAL, 'approval': 'granted', 'producer': 'artificial-producer',
            'protocol_sha256': mod.canonical_hash(self.protocol),
            'raw_source_rereads_allowed': 0,
            'single_source_directory': 'outputs/r15_raw_export_quarantine',
            'training_enabled': False, 'real_scoring_enabled': False}
        self.rebind()

    def rebind(self):
        self.policy.update(extraction_receipt_sha256=mod.canonical_hash(self.receipt),
                           development_sha256=self.receipt['development_sha256'])
        self.review = {'schema': 'ntl-approved-extraction-review-v1', 'reviewer': 'artificial-reviewer',
            'conclusion': 'approved_extraction_identity_verified',
            'policy_sha256': mod.canonical_hash(self.policy),
            'extraction_receipt_sha256': mod.canonical_hash(self.receipt),
            'development_sha256': self.receipt['development_sha256']}

    def load(self):
        self.reads = []
        def reader(directory, name, maximum):
            self.reads.append(name)
            return json.dumps(self.receipt).encode() if name == 'extraction_receipt.json' else self.blob
        with patch.object(mod, 'bounded_file', side_effect=reader):
            return mod.load_approved_export('artificial', self.policy, self.review, self.protocol)

    def test_approved_artificial_export_preserves_bytes(self):
        data, blob, receipt = self.load()
        self.assertEqual((data, blob, receipt), (self.data, self.blob, self.receipt))

    def test_missing_approval_stops_before_payload(self):
        self.policy['approval'] = 'pending'
        with self.assertRaises(ValueError): self.load()
        self.assertNotIn('development.json', self.reads)

    def test_missing_stale_or_nonindependent_review_stops_before_payload(self):
        for key, value in [('conclusion', 'pending'), ('policy_sha256', '0' * 64),
                           ('reviewer', self.policy['producer'])]:
            self.rebind()
            self.review[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.load()
            self.assertNotIn('development.json', self.reads)

    def test_cannot_hide_protected_access_or_enable_training(self):
        for key, value in [('raw_official_csv_opened', False), ('protected_record_parse_events', 0),
                           ('protected_rows_persisted', 1), ('training_admitted', True)]:
            original = copy.deepcopy(self.receipt)
            self.receipt[key] = value
            self.rebind()
            with self.subTest(key=key), self.assertRaises(ValueError): self.load()
            self.assertNotIn('development.json', self.reads)
            self.receipt = original

    def test_policy_cannot_enable_scoring(self):
        self.policy['real_scoring_enabled'] = True
        self.rebind()
        with self.assertRaises(ValueError): self.load()
        self.assertNotIn('development.json', self.reads)

    def test_changed_bytes_fail_before_json_parse(self):
        self.blob = b'not json'
        with self.assertRaisesRegex(ValueError, 'bytes changed'): self.load()

    def test_raw_reread_permission_and_source_redirect_rejected(self):
        for key, value in [('raw_source_rereads_allowed', 1), ('raw_source_rereads_allowed', False),
                           ('single_source_directory', 'another-source')]:
            original = copy.deepcopy(self.policy)
            self.policy[key] = value
            self.rebind()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.load()
            self.assertNotIn('development.json', self.reads)
            self.policy = original

    def test_rehashed_changes_still_fail_frozen_identity(self):
        for key, value in [('texts', 'changed'), ('truth', 17), ('fold_id', 3), ('row_index', True)]:
            data = copy.deepcopy(self.data)
            data[key][0] = value
            self.blob = json.dumps(data).encode()
            self.receipt['development_sha256'] = mod.sha(self.blob)
            self.rebind()
            with self.subTest(key=key), self.assertRaises(ValueError): self.load()

    def test_write_preserves_history_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'admitted'
            plan = mod.nested_plan(self.data, self.protocol)
            admission = mod.write_admitted(target, self.blob, self.receipt, self.policy, self.review, plan)
            self.assertEqual((target / 'development.json').read_bytes(), self.blob)
            self.assertEqual(json.loads((target / 'extraction_receipt.json').read_bytes()), self.receipt)
            self.assertTrue(admission['data_admitted'])
            self.assertFalse(admission['training_enabled'])
            self.assertFalse(admission['real_scoring_enabled'])
            with self.assertRaises(FileExistsError):
                mod.write_admitted(target, self.blob, self.receipt, self.policy, self.review, plan)


if __name__ == '__main__':
    unittest.main()
