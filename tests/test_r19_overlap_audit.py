import unittest
import time
import tempfile
import json
import hashlib
import os
import re
import gc
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scripts.r19_overlap_audit import similarities, THRESHOLD, DIAGNOSTIC, read_pool
from scripts.r19_svc_component import fit_predict, build_fit_frame, R16_LABELS
from scripts import r19_cpu_cv


class OverlapAuditTest(unittest.TestCase):
    def test_exact_and_unrelated(self):
        scores = similarities(['one two three four'], ['one two three four', 'entirely unrelated content'])
        self.assertGreaterEqual(scores[0], THRESHOLD)
        self.assertLess(scores[1], DIAGNOSTIC)

    def test_fold_specific_and_empty(self):
        a = similarities(['alpha beta gamma'], ['alpha beta gamma', '', 'delta epsilon zeta'])
        b = similarities(['delta epsilon zeta'], ['alpha beta gamma', '', 'delta epsilon zeta'])
        self.assertEqual((a >= THRESHOLD).tolist(), [True, False, False])
        self.assertEqual((b >= THRESHOLD).tolist(), [False, False, True])

    def test_synthetic_four_view_fit(self):
        official = [('The apple is red.', 'a'), ('The apple is ripe.', 'a'),
                    ('The ocean is blue.', 'b'), ('The ocean is deep.', 'b')]
        start = time.perf_counter()
        prediction = fit_predict(official, [], [], ['The apple is red.'], ['a', 'b'])
        self.assertEqual(prediction.shape, (1,))
        print('synthetic four-view smoke seconds:', round(time.perf_counter() - start, 3))

    def test_filter_removes_validation_match(self):
        rows = [('alpha beta gamma', 'a'), ('unrelated other words', 'b')]
        kept, count = r19_cpu_cv.filter_source(rows, ['alpha beta gamma'])
        self.assertEqual(count, 1)
        self.assertEqual(kept, [rows[1]])

    def test_extracted_r16_svc_matches_training_rows_and_weights(self):
        path = Path('notebooks/team_e654_r16_typography_half_runnable.ipynb')
        raw = path.read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '8b04a036a5aa43c41907fd6404849179cd6bee1d4588b6c866484487f6bd6075')
        notebook = json.loads(raw)
        source = next(''.join(c['source']) for c in notebook['cells']
                      if 'if a.svc_w > 0:' in ''.join(c.get('source', [])))
        label_code = source[source.index('LABELS = ['):source.index('L2I = ')]
        self.assertEqual(R16_LABELS, ast.literal_eval(label_code.split('=', 1)[1].strip()))
        extra_code = source[source.index('EXTRA_DF = pd.DataFrame'):source.index('FIT_T = np.concatenate')]
        svc_code = source[source.index('if a.svc_w > 0:'):source.index('if a.distort_w > 0:')]
        labels = ['a', 'b', 'c']
        official = [('The apple repeated.', 'a'), ('The ocean repeated.', 'b'), ('The pear repeated.', 'c')]
        extra = [('The apple repeated.', 'a'), ('Unique  extra.', 'b'), ('Unique extra.', 'c'),
                 ('Held　answer', 'c'), ('different extra', 'a'), ('bad label', 'z'), ('', 'a'), (None, 'a')]
        pool = [('Unique extra.', 'b'), ('Held answer', 'a'), ('another pool', 'a'),
                ('another  pool', 'b'), ('different extra', 'c'), ('third pool', 'c'), ('bad label', 'z'), (None, 'c')]
        validation = ['Held answer']
        class StubVectorizer:
            def __init__(self, **kwargs): pass
            def fit_transform(self, texts): return csr_matrix(np.ones((len(texts), 1)))
            def transform(self, texts): return csr_matrix(np.ones((len(texts), 1)))
        class StubSVC:
            def __init__(self, **kwargs): pass
            def fit(self, matrix, y, sample_weight): return self
            def decision_function(self, matrix): return np.zeros((matrix.shape[0], 3))
        for sim in ('', '1'):
            env = {'pd': pd, 'np': np, 're': re, 'os': os, 'gc': gc,
                   'train': pd.DataFrame(official, columns=['text', 'model']),
                   'test': pd.DataFrame({'text': validation}), 'LABELS': labels,
                   'L2I': {label: i for i, label in enumerate(labels)},
                   'a': SimpleNamespace(extra='extra.parquet', pool='pool.parquet', extra_w=0.5, sim=sim, svc_w=1.5),
                   'TfidfVectorizer': StubVectorizer, 'LP': {}, 'log': lambda *args: None}
            def fake_read(path):
                return pd.DataFrame(extra if path == 'extra.parquet' else pool, columns=['text', 'model'])
            with patch('pandas.read_parquet', side_effect=fake_read), \
                 patch('os.path.exists', return_value=True), \
                 patch('sklearn.svm.LinearSVC', StubSVC):
                exec(extra_code, env)
                exec(svc_code, env)
            actual = build_fit_frame(official, extra, pool, validation, labels, 0.5, bool(sim))
            self.assertEqual(actual.text.tolist(), env['_fit'])
            self.assertEqual(actual.model.map(env['L2I']).tolist(), env['_fy'].tolist())
            np.testing.assert_array_equal(actual['_srcw'].to_numpy(dtype=np.float32), env['_sw'])

    def test_fit_passes_sample_weights(self):
        official = [('Repeated apple text.', 'a'), ('Repeated ocean text.', 'b')]
        svc = MagicMock()
        svc.predict.return_value = np.array([0])
        with patch('scripts.r19_svc_component.LinearSVC', return_value=svc):
            fit_predict(official, [('extra repeated text.', 'a')], [('pool repeated text.', 'b')], ['unrelated'], ['a', 'b'])
        args, kwargs = svc.fit.call_args
        self.assertEqual(args[0].shape[0], 4)
        np.testing.assert_array_equal(kwargs['sample_weight'], np.array([1, 1, .5, .35], dtype=np.float32))

    def test_open_pool_excludes_flag_before_dropping_column(self):
        frame = pd.DataFrame({'text': ['one', 'two'], 'model': ['a', 'b'],
                              'flag_train_prompt_regen': [True, False]})
        with patch('scripts.r19_overlap_audit.pd.read_parquet', return_value=frame):
            got, removed = read_pool('open')
        self.assertEqual(removed, 1)
        self.assertEqual(got.to_dict('records'), [{'text': 'two', 'model': 'b'}])

    def test_synthetic_five_fold_pipeline(self):
        labels = [f'class{i}' for i in range(18)]
        data = {'labels': labels, 'texts': [], 'truth': [], 'fold_id': [], 'groups': []}
        for fold in range(5):
            for i in range(18):
                data['texts'].append(f'The synthetic common passage number {i} and class token{i}.')
                data['truth'].append(i)
                data['fold_id'].append(fold)
                data['groups'].append(f'group{fold}')
        sources = [('A small unrelated synthetic passage.', 'class0')]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(r19_cpu_cv, 'OUT', Path(tmp)), \
             patch.object(r19_cpu_cv, 'validate_data', return_value=data), \
             patch.object(r19_cpu_cv, 'R16_LABELS', labels), \
             patch.object(r19_cpu_cv, 'load_source', return_value=(sources, 0)), \
             patch.object(r19_cpu_cv, 'metrics', return_value={'synthetic_only': True}):
            started = time.perf_counter()
            r19_cpu_cv.worker()
            self.assertTrue((Path(tmp) / 'result.json').is_file())
            print('synthetic five-fold pipeline seconds:', round(time.perf_counter() - started, 3))

    def test_claim_survives_failure_and_blocks_retry(self):
        child = MagicMock(pid=12345, returncode=1)
        child.poll.return_value = 1
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(r19_cpu_cv, 'OUT', Path(tmp)), \
             patch.object(r19_cpu_cv, 'validate_data'), \
             patch.object(r19_cpu_cv.subprocess, 'Popen', return_value=child) as spawn, \
             patch.object(r19_cpu_cv.psutil, 'Process'):
            with self.assertRaises(RuntimeError):
                r19_cpu_cv.supervise()
            self.assertTrue((Path(tmp) / 'claim.json').exists())
            with self.assertRaises(FileExistsError):
                r19_cpu_cv.supervise()
            self.assertEqual(spawn.call_count, 1)

    def test_paired_bootstrap_and_gate(self):
        truth = np.tile(np.arange(18), 5)
        groups = np.repeat(np.arange(5), 18)
        result = r19_cpu_cv.metrics(truth, (truth + 1) % 18, truth, groups, groups)
        self.assertEqual(result['delta_ba'], 1.0)
        self.assertEqual(result['bootstrap_95'], [1.0, 1.0])
        self.assertTrue(result['adopt'])


if __name__ == '__main__':
    unittest.main()
