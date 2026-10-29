"""Generated CSVs only. No official source, cached arrays or protected rows read."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments'))
import r15_prepare_raw_dev_export as exporter
from r15_typography_replay import DEVELOPMENT_LABELS, canonical_hash
from r15_validate_dev_export import metadata_gate


def artificial_source():
    """Construct expected retained rows before passing anything to the exporter."""
    names = np.asarray([f'generated-group-{i:03}' for i in range(150)])
    excluded_groups = set(np.random.default_rng(20261018).permutation(names)[:30])
    first_development = next(g for g in names if g not in excluded_groups)
    records, expected = [], {key: [] for key in
                            ('row_index', 'ids', 'groups', 'truth', 'labels', 'texts', 'fold_id')}
    expected['labels'] = list(DEVELOPMENT_LABELS)
    for g in names:
        for c, label in enumerate(DEVELOPMENT_LABELS):
            if g == first_development and c == 0:
                continue
            excluded = g in excluded_groups
            text = ('ARTIFICIAL_EXCLUDED_SENTINEL' if excluded else 'Retained artificial passage, "quoted"\nsecond line')
            text += f' {g} class {c}'
            row = {'id': f'generated-id-{len(records)}', 'prompt_id': str(g), 'text': text, 'model': label}
            if not excluded:
                expected['row_index'].append(len(records))
                expected['ids'].append(row['id'])
                expected['groups'].append(str(g))
                expected['truth'].append(c)
                expected['texts'].append(text)
            records.append(row)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['id', 'prompt_id', 'text', 'model'], lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(records)
    blob = stream.getvalue().encode('utf-8-sig')
    digest = hashlib.sha256(blob).hexdigest()
    held = [i for i, row in enumerate(records) if row['prompt_id'] in excluded_groups]
    seeded_development_groups = np.random.default_rng(20261018).permutation(np.unique(expected['groups']))
    group_to_fold = {str(group): fold for fold, groups in enumerate(np.array_split(seeded_development_groups, 5))
                     for group in groups}
    manifest = {'seed': 20261018, 'stability_seeds': [20261019, 20261020], 'train_sha256': digest,
                'development_indices': expected['row_index'], 'holdout_indices': held,
                'rows': [{'row_index': i, 'id': row['id'], 'prompt_id': row['prompt_id'],
                          'role': 'holdout' if i in held else 'development',
                          'development_fold': group_to_fold.get(row['prompt_id'])}
                         for i, row in enumerate(records)]}
    split_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    fold_id = np.full(2159, -1, dtype=np.int64)
    for fold, (_, valid) in enumerate(GroupKFold(5).split(np.zeros(2159), groups=expected['groups'])):
        fold_id[valid] = fold
    expected['fold_id'] = fold_id.tolist()
    protocol = {'development_rows': 2159, 'development_prompts': 120, 'folds': 5,
                'row_identity_sha256': canonical_hash({k: expected[k] for k in
                    ('row_index', 'ids', 'groups', 'truth', 'labels')}),
                'text_sha256': canonical_hash(expected['texts']),
                'fold_sha256': canonical_hash(expected['fold_id']),
                'historical_split_sha256': hashlib.sha256(split_bytes).hexdigest()}
    return blob, digest, protocol, expected


class RawExportProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blob, cls.digest, cls.protocol, cls.expected = artificial_source()
        cls.proposal = json.loads(exporter.PROPOSAL.read_text(encoding='utf-8'))

    def test_checked_in_proposal_blocks_before_source_or_claim_io(self):
        self.assertEqual(self.proposal['approval'], 'pending_user_boundary_amendment')
        with mock.patch.object(Path, 'open', side_effect=AssertionError('No file open')):
            with mock.patch.object(Path, 'stat', side_effect=AssertionError('No source stat')):
                with self.assertRaisesRegex(ValueError, 'permission.*pending'):
                    exporter.run_approved(self.proposal)

    def test_granted_flag_without_reference_is_not_enough(self):
        proposal = dict(self.proposal, approval='granted')
        with mock.patch.object(Path, 'open', side_effect=AssertionError('No file open')):
            with self.assertRaisesRegex(ValueError, 'permission.*pending'):
                exporter.run_approved(proposal)

    def test_source_output_or_admission_scope_cannot_be_redirected(self):
        for changes in ({'source': 'elsewhere.csv'}, {'output': 'data'}, {'training_admitted': True},
                        {'legacy_preisolated_intake_compatible': True}, {'source_sha256': '0' * 64}):
            proposal = dict(self.proposal, approval='granted', approval_reference='artificial approval only', **changes)
            with mock.patch.object(Path, 'open', side_effect=AssertionError('No file open')):
                with self.assertRaisesRegex(ValueError, 'scope'):
                    exporter.run_approved(proposal)

    def test_source_hash_and_bounds_checked_before_csv_parsing(self):
        with mock.patch.object(exporter, 'rows', side_effect=AssertionError('Do not parse')):
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                exporter.extract_from_bytes(self.blob + b'x', self.protocol, self.digest)
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                exporter.extract_from_bytes(b'x' * (exporter.MAX_SOURCE_BYTES + 1), self.protocol, self.digest)

    def test_only_expected_rows_persist_with_exact_multiline_text_and_historical_labels(self):
        payload, receipt = exporter.extract_from_bytes(self.blob, self.protocol, self.digest)
        self.assertEqual(json.loads(payload), self.expected)
        self.assertNotIn(b'ARTIFICIAL_EXCLUDED_SENTINEL', payload)
        self.assertTrue(receipt['raw_official_csv_opened'])
        self.assertEqual(receipt['protected_record_parse_events'], 1080)
        self.assertEqual(receipt['protected_rows_persisted'], 0)
        self.assertFalse(receipt['training_admitted'])
        self.assertFalse(receipt['legacy_preisolated_intake_compatible'])
        with self.assertRaises(ValueError):
            metadata_gate(receipt, {}, self.protocol)

    def test_frozen_row_text_fold_or_split_mismatch_rejected(self):
        for field in ('row_identity_sha256', 'text_sha256', 'fold_sha256', 'historical_split_sha256'):
            with self.subTest(field=field):
                protocol = dict(self.protocol, **{field: '0' * 64})
                with self.assertRaises(ValueError):
                    exporter.extract_from_bytes(self.blob, protocol, self.digest)

    def test_duplicate_header_and_malformed_row_are_rejected(self):
        for blob in (b'id,id,text,model\nx,p,t,m\n', b'id,prompt_id,text,model\nx,p,t,m,extra\n'):
            with self.assertRaises(ValueError):
                list(exporter.rows(blob))

    def test_quarantine_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'quarantine'
            exporter.write_quarantine(destination, b'{"synthetic":true}\n', {'synthetic': True})
            before = (destination / 'development.json').read_bytes()
            with self.assertRaises(FileExistsError):
                exporter.write_quarantine(destination, b'changed', {})
            self.assertEqual((destination / 'development.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
