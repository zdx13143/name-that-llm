import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts import r21_extract_members, r21_members, r21_scorebank_cv as cv


class ScorebankTest(unittest.TestCase):
    def test_frozen_blocks_are_exact_notebook_sources(self):
        expected = {'r16': '38204e9f00cb7da783932a917929d5d7af8c146f02f5e487bb033d5fb8b48f37',
                    'areeb': 'adbde4ef757f3ae11d5bf9a748a10ce48d4e5d00f792ed22277f3b94449077a5'}
        for condition, path in r21_extract_members.SOURCES.items():
            sha, blocks = r21_extract_members.extract(path)
            self.assertEqual(sha, expected[condition])
            for name, code in blocks.items():
                self.assertEqual((r21_members.SOURCE / f'{condition}_{name}.py.txt').read_text(encoding='utf-8'), code)

    def test_member_output_parity_both_extracted_sources(self):
        labels = cv.R16_LABELS
        official = [(f'Common synthetic style marker word{i} unique{i} repeated repeated.', label)
                    for i, label in enumerate(labels) for _ in range(4)]
        extra = [(f'Extra common style marker word{i} distinct{i} repeated repeated.', label)
                 for i, label in enumerate(labels)]
        validation = ['Common synthetic style marker word0 repeated.', 'Common synthetic style marker word2 repeated.']
        for member in cv.MEMBERS:
            first = r21_members.run_member('r16', member, official, extra, [], validation, labels)
            second = r21_members.run_member('areeb', member, official, extra, [], validation, labels)
            np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.exp(first).sum(axis=1), 1., rtol=1e-6)

    def test_global_sample_is_not_resampled_by_fold(self):
        pool = pd.DataFrame({'text': [f'x{i}' for i in range(12)],
                             'model': ['a'] * 6 + ['b'] * 6,
                             'match_weight': np.ones(12)})
        extra = pd.DataFrame({'model': ['a', 'a', 'b']})
        sampled = cv.sample_v15(pool, extra)
        expected = pd.concat([g.sample(min(len(g), int(2 * extra.model.value_counts().get(m, 0))),
                                       weights='match_weight', random_state=0)
                              for m, g in pool.groupby('model')])[['text', 'model']]
        pd.testing.assert_frame_equal(sampled, expected)
        self.assertEqual(len(sampled), 6)

    def test_fold_filter_includes_normalized_exact_key(self):
        rows = [('HELLO　world', 'a'), ('different tokens entirely', 'b')]
        with patch.object(cv, 'filter_source', return_value=(rows, 0)):
            kept, counts = cv.filter_fold(rows, ['HELLO world'])
        self.assertEqual(kept, [rows[1]])
        self.assertEqual(counts['exact_after_cosine'], 1)

    def test_flag_removal_fails_closed(self):
        frame = pd.DataFrame({'text': ['x'], 'model': ['a']})
        with patch.object(cv, 'read_pool', side_effect=[(frame, 0), (frame, 0), (frame, 1)]):
            with self.assertRaisesRegex(ValueError, 'nonzero'):
                cv.load_sources()

    def test_insufficient_memory_does_not_create_claim(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cv, 'OUT', Path(tmp)), \
             patch.object(cv.psutil, 'virtual_memory') as memory:
            memory.return_value.available = 0
            with self.assertRaisesRegex(RuntimeError, 'no claim'):
                cv.supervise()
            self.assertFalse((Path(tmp) / 'claim.json').exists())

    def test_rss_cap_kills_tree_and_claim_blocks_retry(self):
        supervisor, process, child = MagicMock(), MagicMock(), MagicMock()
        supervisor.cpu_times.return_value = (0, 0)
        supervisor.memory_info.return_value = SimpleNamespace(rss=1)
        process.children.return_value = [child]
        process.is_running.return_value = child.is_running.return_value = True
        process.memory_info.return_value = SimpleNamespace(rss=cv.MAX_RSS)
        child.memory_info.return_value = SimpleNamespace(rss=1)
        process.cpu_times.return_value = child.cpu_times.return_value = (0, 0)
        proc = MagicMock(pid=1234)
        proc.poll.return_value = None
        proc.wait.return_value = -9
        with tempfile.TemporaryDirectory() as tmp, patch.object(cv, 'OUT', Path(tmp)), \
             patch.object(cv, 'validate_data'), \
             patch.object(cv.psutil, 'virtual_memory', return_value=SimpleNamespace(available=cv.MIN_AVAIL)), \
             patch.object(cv.psutil, 'Process', side_effect=[supervisor, process]), \
             patch.object(cv.psutil, 'wait_procs'), \
             patch.object(cv.subprocess, 'Popen', return_value=proc) as spawn:
            with self.assertRaises(RuntimeError):
                cv.supervise()
            process.kill.assert_called_once()
            child.kill.assert_called_once()
            receipt = json.loads((cv.OUT / 'receipt.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertIn('RSS cap', receipt['reason'])
            with self.assertRaises(FileExistsError):
                cv.supervise()
            self.assertEqual(spawn.call_count, 1)

    def test_full_synthetic_smoke(self):
        cv.smoke()

    def test_interrupted_synthetic_resume_only_computes_missing_units(self):
        data = {'labels': cv.R16_LABELS, 'texts': [], 'truth': [], 'fold_id': [], 'groups': []}
        for fold in range(5):
            for label in range(18):
                data['texts'].append(f'synthetic fold{fold} class{label}')
                data['truth'].append(label)
                data['fold_id'].append(fold)
                data['groups'].append(f'group{fold}')
        sources = {key: [] for key in ('baseline', 'extra', 'open', 'v15_extra')}
        calls = []
        def fake_member(condition, member, official, extra, pool, validation, labels):
            calls.append((condition, member))
            if len(calls) == 4:
                raise RuntimeError('synthetic interruption')
            return np.tile(np.arange(18, dtype=float), (len(validation), 1))
        def consistent_member(condition, member, official, extra, pool, validation, labels):
            calls.append((condition, member))
            return np.tile(np.arange(18, dtype=float), (len(validation), 1))
        with tempfile.TemporaryDirectory() as tmp:
            partial, resumed = Path(tmp) / 'partial', Path(tmp) / 'resumed'
            partial.mkdir()
            resumed.mkdir()
            with patch.object(cv, 'run_member', side_effect=fake_member):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    cv.calculate(data, sources, partial, {'synthetic': True})
            self.assertEqual(calls, [('r16', 'tfidf'), ('r16', 'distort'), ('r16', 'svc4'), ('areeb', 'tfidf')])
            progress = json.loads((partial / 'progress.json').read_text())
            self.assertEqual(len(progress['completed_units']), 3)
            self.assertFalse((partial / 'result.json').exists())
            calls.clear()
            with patch.object(cv, 'run_member', side_effect=consistent_member):
                cv.calculate(data, sources, resumed, {'synthetic': True}, resume_from=partial)
            self.assertEqual(len(calls), 27)
            self.assertEqual(calls[0], ('areeb', 'tfidf'))
            result = json.loads((resumed / 'result.json').read_text())
            self.assertEqual(result['resumed_units'], ['baseline__f0__tfidf', 'baseline__f0__distort', 'baseline__f0__svc4'])
            self.assertEqual(len(list((resumed / 'parts').glob('*.npz'))), 30)
            with np.load(resumed / 'scores.npz', allow_pickle=False) as scores:
                np.testing.assert_array_equal(scores['r16_tfidf'][0], np.arange(18))

    def test_checkpoint_identity_and_alignment_are_required(self):
        identity = {'config_sha256': 'a' * 64, 'input_sha256': {'dev': 'b' * 64}}
        stage = {'status': 'ok', 'wall_seconds': 1, 'sampled_peak_rss_bytes': 100}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline__f0__tfidf.npz'
            rows = np.arange(3, dtype=np.int64)
            cv.save_part(path, np.ones((3, 18)), rows, identity, path.stem, stage)
            cv.read_part(path, identity, path.stem, rows)
            with self.assertRaisesRegex(ValueError, 'identity'):
                cv.read_part(path, {**identity, 'config_sha256': 'c' * 64}, path.stem, rows)
            with self.assertRaisesRegex(ValueError, 'identity'):
                cv.read_part(path, {**identity, 'input_sha256': {'dev': 'd' * 64}}, path.stem, rows)
            with self.assertRaisesRegex(ValueError, 'alignment'):
                cv.read_part(path, identity, path.stem, rows + 1)
            path.write_bytes(b'truncated')
            with self.assertRaisesRegex(ValueError, 'invalid checkpoint'):
                cv.read_part(path, identity, path.stem, rows)

    def test_timeout_receipt_counts_parts_without_partial_gate(self):
        supervisor, process = MagicMock(), MagicMock()
        supervisor.cpu_times.return_value = process.cpu_times.return_value = (0, 0)
        supervisor.memory_info.return_value = process.memory_info.return_value = SimpleNamespace(rss=1)
        process.children.return_value = []
        process.is_running.return_value = True
        proc = MagicMock(pid=1234)
        proc.poll.return_value = None
        proc.wait.return_value = -9
        with tempfile.TemporaryDirectory() as tmp, patch.object(cv, 'OUT', Path(tmp)), \
             patch.object(cv, 'LIMIT', 1), patch.object(cv, 'validate_data'), \
             patch.object(cv.psutil, 'virtual_memory', return_value=SimpleNamespace(available=cv.MIN_AVAIL)), \
             patch.object(cv.psutil, 'Process', side_effect=[supervisor, process]), \
             patch.object(cv.psutil, 'wait_procs'), patch.object(cv.time, 'monotonic', side_effect=[0, 2, 3]):
            def spawn(*args, **kwargs):
                parts = cv.OUT / 'parts'
                parts.mkdir()
                cv.save_part(parts / 'baseline__f0__tfidf.npz', np.ones((1, 18)), np.array([0]),
                             {'config_sha256': 'a', 'input_sha256': {}}, 'baseline__f0__tfidf', {'status': 'ok'})
                return proc
            with patch.object(cv.subprocess, 'Popen', side_effect=spawn):
                with self.assertRaises(RuntimeError):
                    cv.supervise()
            receipt = json.loads((cv.OUT / 'receipt.json').read_text())
            self.assertEqual(receipt['completed_units'], 1)
            self.assertIn('budget exceeded', receipt['reason'])
            self.assertFalse((cv.OUT / 'result.json').exists())

    def test_kaggle_package_is_synchronized_and_private(self):
        from scripts.build_r21_package import ROOT, SCRIPTS, DEV_SHA
        package = ROOT / 'kaggle/r21_dataset'
        self.assertEqual(hashlib.sha256((package / 'development.json').read_bytes()).hexdigest(), DEV_SHA)
        for name in SCRIPTS:
            self.assertEqual((package / name).read_bytes(), (ROOT / 'scripts' / name).read_bytes())
        for path in (ROOT / 'scripts/r21_source').glob('*.txt'):
            self.assertEqual(path.read_bytes(), (package / 'r21_source' / path.name).read_bytes())
        kernel = json.loads((ROOT / 'notebooks/r21/kernel-metadata.json').read_text())
        self.assertTrue(kernel['is_private'])
        self.assertFalse(kernel['enable_gpu'])
        self.assertFalse(kernel['enable_internet'])
        self.assertEqual(kernel['id'], 'zephyrpong/ntl-r21-scorebank')
        notebook = ROOT / 'notebooks/r21/r21_scorebank_kaggle.ipynb'
        self.assertFalse((ROOT / 'notebooks/r21_scorebank_kaggle.ipynb').exists())
        cell = json.loads(notebook.read_text())['cells'][0]
        compile(''.join(cell['source']), str(notebook), 'exec')


if __name__ == '__main__':
    unittest.main()
