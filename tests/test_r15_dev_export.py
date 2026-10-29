"""Artificial fixtures only: no official CSV, cache, OOF or GPU access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r15_dev_export', ROOT / 'scripts/r15_validate_dev_export.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
from r15_typography_replay import LABELS, canonical_hash


def fixture():
    data = {k: [] for k in mod.FIELDS}
    data['labels'] = list(LABELS)
    for g in range(120):
        for c in range(18):
            if g == 0 and c == 0:
                continue
            data['row_index'].append(g * 18 + c)
            data['ids'].append(f'artificial-row-{g}-{c}')
            data['groups'].append(f'artificial-prompt-{g:03d}')
            data['truth'].append(c)
            data['texts'].append(f'Artificial test passage for group {g}, class {c}.')
            data['fold_id'].append(g % 5)
    protocol = {'development_rows': 2159, 'development_prompts': 120, 'folds': 5,
                'historical_split_sha256': '1' * 64,
                'row_identity_sha256': canonical_hash({k: data[k] for k in
                    ('row_index', 'ids', 'groups', 'truth', 'labels')}),
                'text_sha256': canonical_hash(data['texts']),
                'fold_sha256': canonical_hash(data['fold_id'])}
    blob = json.dumps(data).encode()
    evidence = b'Artificial source-isolation evidence. Not real provenance.\n'
    receipt = dict(protocol, schema='ntl-development-only-export-v1', producer='synthetic-producer',
                   source_kind='preexisting_development_only_artifact',
                   all_input_sources_development_only=True, raw_official_csv_opened=False,
                   protected_content_reads=0, historical_train_sha256=mod.TRAIN_SHA256,
                   development_file_sha256=hashlib.sha256(blob).hexdigest(),
                   source_evidence_sha256=hashlib.sha256(evidence).hexdigest())
    review = {'schema': 'ntl-development-export-review-v1', 'reviewer': 'synthetic-reviewer',
              'receipt_sha256': canonical_hash(receipt),
              'source_evidence_sha256': receipt['source_evidence_sha256'],
              'conclusion': 'preisolated_development_source_verified'}
    return data, protocol, blob, evidence, receipt, review


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.data, self.protocol, self.blob, self.evidence, self.receipt, self.review = fixture()

    def mocked_reader(self, receipt=None, review=None, blob=None, evidence=None):
        values = {'development_export_receipt.json': json.dumps(receipt or self.receipt).encode(),
                  'development_export_review.json': json.dumps(review or self.review).encode(),
                  'source_evidence.md': self.evidence if evidence is None else evidence,
                  'development.json': self.blob if blob is None else blob}
        calls = []
        def reader(directory, name, maximum):
            calls.append(name)
            return values[name]
        return reader, calls

    def test_valid_artificial_export(self):
        reader, calls = self.mocked_reader()
        with patch.object(mod, 'bounded_file', side_effect=reader):
            result, _ = mod.load_export('unused', self.protocol)
        self.assertEqual(result, self.data)
        self.assertEqual(calls[-1], 'development.json')

    def test_nonobject_sidecars_are_rejected(self):
        for value in (None, [], 'wrong'):
            for receipt, review in ((value, self.review), (self.receipt, value)):
                with self.assertRaisesRegex(ValueError, 'JSON objects'):
                    mod.metadata_gate(receipt, review, self.protocol)

    def test_cli_protocol_is_semantically_pinned(self):
        actual = json.loads((ROOT / 'configs/r15_protocol_20261029.json').read_bytes())
        # Harmless whitespace is portable; changed identities are never accepted.
        with patch.object(mod, 'bounded_file', return_value=json.dumps(actual, indent=4).encode()):
            self.assertEqual(mod.load_frozen_protocol(), actual)
        actual['text_sha256'] = 'f' * 64
        with patch.object(mod, 'bounded_file', return_value=json.dumps(actual).encode()), self.assertRaisesRegex(ValueError, 'protocol changed'):
            mod.load_frozen_protocol()

    def test_unsafe_source_rejected_before_payload_read(self):
        mutations = [{'source_kind': 'raw_csv_filter'}, {'raw_official_csv_opened': True},
                     {'protected_content_reads': False}, {'protected_content_reads': 1},
                     {'all_input_sources_development_only': False}, {'development_rows': 2699}]
        for changes in mutations:
            with self.subTest(changes=changes):
                receipt = dict(self.receipt, **changes)
                reader, calls = self.mocked_reader(receipt=receipt)
                with patch.object(mod, 'bounded_file', side_effect=reader), self.assertRaises(ValueError):
                    mod.load_export('unused', self.protocol)
                self.assertNotIn('development.json', calls)

    def test_review_rejected_before_payload_read(self):
        for changes in ({'reviewer': 'synthetic-producer'}, {'conclusion': 'pending'},
                        {'receipt_sha256': '0' * 64}):
            reader, calls = self.mocked_reader(review=dict(self.review, **changes))
            with patch.object(mod, 'bounded_file', side_effect=reader), self.assertRaises(ValueError):
                mod.load_export('unused', self.protocol)
            self.assertNotIn('development.json', calls)

    def test_evidence_tampering_stops_before_data(self):
        reader, calls = self.mocked_reader(evidence=b'changed')
        with patch.object(mod, 'bounded_file', side_effect=reader), self.assertRaises(ValueError):
            mod.load_export('unused', self.protocol)
        self.assertNotIn('development.json', calls)

    def test_payload_byte_hash_checked_before_parsing(self):
        reader, _ = self.mocked_reader(blob=b'not json')
        with patch.object(mod, 'bounded_file', side_effect=reader), self.assertRaisesRegex(ValueError, 'bytes changed'):
            mod.load_export('unused', self.protocol)

    def test_extra_fields_and_boolean_indices_rejected(self):
        for change in ('extra', 'bool'):
            data = copy.deepcopy(self.data)
            if change == 'extra':
                data['other_rows'] = []
            else:
                data['row_index'][0] = True
            blob = json.dumps(data).encode()
            receipt = dict(self.receipt, development_file_sha256=hashlib.sha256(blob).hexdigest())
            review = dict(self.review, receipt_sha256=canonical_hash(receipt))
            reader, _ = self.mocked_reader(receipt=receipt, review=review, blob=blob)
            with patch.object(mod, 'bounded_file', side_effect=reader), self.assertRaises(ValueError):
                mod.load_export('unused', self.protocol)

    def test_frozen_text_and_fold_hashes_are_checked(self):
        for key in ('texts', 'fold_id'):
            data = copy.deepcopy(self.data)
            data[key][0] = 'changed' if key == 'texts' else 4
            blob = json.dumps(data).encode()
            receipt = dict(self.receipt, development_file_sha256=hashlib.sha256(blob).hexdigest())
            review = dict(self.review, receipt_sha256=canonical_hash(receipt))
            reader, _ = self.mocked_reader(receipt=receipt, review=review, blob=blob)
            with patch.object(mod, 'bounded_file', side_effect=reader), self.assertRaises(ValueError):
                mod.load_export('unused', self.protocol)

    def test_nested_groups_and_coverage(self):
        plan = mod.nested_plan(self.data, self.protocol)
        self.assertFalse(plan['training_enabled'])
        all_rows = set(range(2159))
        for outer in plan['folds']:
            train, valid = set(outer['train_rows']), set(outer['valid_rows'])
            self.assertEqual(train | valid, all_rows)
            self.assertFalse(train & valid)
            seen = []
            for inner in outer['inner']:
                fit, test = set(inner['train_rows']), set(inner['valid_rows'])
                self.assertEqual(fit | test, train)
                self.assertFalse(fit & test)
                self.assertFalse((fit | test) & valid)
                fit_groups = {self.data['groups'][i] for i in fit}
                test_groups = {self.data['groups'][i] for i in test}
                self.assertFalse(fit_groups & test_groups)
                seen.extend(test)
            self.assertEqual(sorted(seen), sorted(train))
        self.assertEqual(plan, mod.nested_plan(self.data, self.protocol))

    def test_bounded_files_and_no_missing_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'no raw-data fallback'):
                mod.bounded_file(directory, 'development.json', 8)
            Path(directory, 'development.json').write_bytes(b'0123456789')
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                mod.bounded_file(directory, 'development.json', 8)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                mod.bounded_file(directory, '../train.csv', 8)


if __name__ == '__main__':
    unittest.main()
