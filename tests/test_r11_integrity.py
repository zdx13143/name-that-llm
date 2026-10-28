"""Synthetic failure injection and execution of real baseline/patched fusion code."""

import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments'))
import build_r11_integrity as builder
import r11_integrity as integrity


@contextmanager
def working(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def member_manifest(n, grouped):
    result = {key: {'weight': weight, 'rows': n} for key, weight in integrity.BASE_WEIGHTS.items()}
    if grouped:
        result.update({key: {'weight': weight, 'rows': grouped} for key, weight in integrity.GROUP_WEIGHTS.items()})
    return result


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = json.loads(builder.BASELINE.read_bytes())
        cls.original = [builder.source(c) for c in original['cells'] if c['cell_type'] == 'code']
        cls.notebook = builder.build()
        cls.patched = [builder.source(c) for c in cls.notebook['cells'] if c['cell_type'] == 'code'][2:]
        cls.base_infer = cls.original[2].split('\n', 1)[1]
        cls.fixed_infer = cls.patched[2].split('\n', 1)[1]

    def setUp(self):
        temp_root = ROOT / 'outputs/r11_test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.path = self.dir / 'comp_logit.npy'
        self.run_id = 'synthetic-run-11'
        self.rng = np.random.default_rng(11)

    def fixture(self, n=18, grouped=0, permute=False):
        logits = self.rng.normal(size=(n, 18))
        labels = list(reversed(integrity.LABELS)) if permute else integrity.LABELS
        ids = ['synthetic-' + str(i) for i in range(n)]
        integrity.write_comp(self.path, logits, labels, ids, 1.0, member_manifest(n, grouped), grouped, self.run_id)
        return logits, labels, ids

    def metadata(self):
        return json.loads(self.path.with_name('comp_meta.json').read_text())

    def save_metadata(self, meta):
        self.path.with_name('comp_meta.json').write_text(json.dumps(meta))

    def fusion(self, code, n, grouped, atlas_failure=False, nonfinite=False, confident=False):
        begin = code.index('COMP_P = None\n')
        end = code.index('if a.sim:\n', begin)
        block = code[begin:end]
        rng = np.random.default_rng(n)
        lp = {'synthetic_member_' + str(i): rng.normal(0, 0.4, size=(n, 18)) for i in range(6)}
        if nonfinite:
            lp['synthetic_member_0'][0, 0] = np.nan
        if confident:
            for value in lp.values():
                value[:, 0] = 100.0
        atlas = rng.uniform(0.1, 1, size=(n, 18))
        atlas /= atlas.sum(1, keepdims=True)
        def atlas_scores(bundle, rows, budget):
            if atlas_failure:
                raise RuntimeError('synthetic ATLAS failure')
            return atlas[rows]
        groups = np.arange(n)
        if grouped:
            groups[:min(n, 18)] = 0
        env = {
            'np': np, 'pd': pd, 'json': json, 'os': os,
            'LABELS': integrity.LABELS, 'LP': lp, 'tot': sum(lp.values()),
            'test': pd.DataFrame({'id': ['synthetic-' + str(i) for i in range(n)],
                                  'text': ['Synthetic — example “ text.' if i % 2 else 'Synthetic sentence.' for i in range(n)]}),
            'Gte': groups, 'linear_sum_assignment': linear_sum_assignment,
            'atlas_scores': atlas_scores, 'log': lambda *args: None, 'gpu_mem': lambda: '',
            'a': SimpleNamespace(comp_w=0.15, comp_path=str(self.path), atlas_dir='synthetic-atlas',
                                 atlas_w='0.10', atlas_margin=0.3, atlas_budget=100,
                                 sinkhorn=1.0, sk_div=1, post=1, perpassage=1),
        }
        tree = ast.parse(code)
        guards = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GUARDS' for t in node.targets))
        exec(compile(ast.Module(body=[guards], type_ignores=[]), '<baseline-guards>', 'exec'), env)
        with patch.dict(os.environ, {'R11_RUN_ID': self.run_id}):
            exec(compile(block, '<real-fusion-code>', 'exec'), env)
        return env['tot'], env['pred']

    def test_valid_path_bitwise_parity(self):
        self.parity_cases = 0
        for n in (1, 3, 17, 18, 36, 73):
            for grouped in (False, True):
                for permute in (False, True):
                    with self.subTest(rows=n, grouped=grouped, reordered_labels=permute):
                        self.fixture(n, n if grouped and n > 1 else 0, permute)
                        before = self.fusion(self.base_infer, n, grouped)
                        after = self.fusion(self.fixed_infer, n, grouped)
                        np.testing.assert_array_equal(before[0], after[0])
                        np.testing.assert_array_equal(before[1], after[1])
                        self.parity_cases += 1
        self.assertEqual(self.parity_cases, 24)

    def test_atlas_empty_selected_rows_parity(self):
        self.fixture(73)
        before = self.fusion(self.base_infer, 73, False, confident=True)
        after = self.fusion(self.fixed_infer, 73, False, confident=True)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])

    def test_missing_comp_previously_fell_back(self):
        before = self.fusion(self.base_infer, 18, False)
        self.assertEqual(before[1].shape, (18,))
        with self.assertRaises(FileNotFoundError):
            self.fusion(self.fixed_infer, 18, False)

    def test_atlas_failure_previously_fell_back(self):
        self.fixture()
        self.assertEqual(self.fusion(self.base_infer, 18, False, atlas_failure=True)[1].shape, (18,))
        with self.assertRaisesRegex(RuntimeError, 'required ATLAS failed'):
            self.fusion(self.fixed_infer, 18, False, atlas_failure=True)

    def test_nonfinite_final_scores_stop(self):
        self.fixture()
        with np.errstate(invalid='ignore'):
            with self.assertRaisesRegex(RuntimeError, 'nonfinite final ensemble'):
                self.fusion(self.fixed_infer, 18, False, nonfinite=True)

    def test_metadata_faults_fail(self):
        mutations = {
            'stale': lambda m: m.update(r11_run_id='previous-run'),
            'schema': lambda m: m.update(r11_schema=0),
            'reordered_ids': lambda m: m['ids'].reverse(),
            'duplicate_ids': lambda m: m['ids'].__setitem__(1, m['ids'][0]),
            'missing_class': lambda m: m['labels'].pop(),
            'duplicate_class': lambda m: m['labels'].__setitem__(1, m['labels'][0]),
            'foreign_class': lambda m: m['labels'].__setitem__(1, 'not-a-class'),
            'missing_member': lambda m: m['members'].pop('ultrabert'),
            'unexpected_member': lambda m: m['members'].update(lora24_public={'rows': 18, 'weight': 0.177}),
            'weight_drift': lambda m: m['members']['stylegram'].update(weight=1.0),
            'coverage': lambda m: m['members']['stylegram'].update(rows=17),
            'missing_group_members': lambda m: m.update(grouped_rows=18),
            'invalid_group_count': lambda m: m.update(grouped_rows=1),
            'elapsed': lambda m: m.update(seconds=float('nan')),
            'hash': lambda m: m.update(logit_sha256='bad'),
        }
        for name, mutate in mutations.items():
            with self.subTest(fault=name):
                self.fixture()
                meta = self.metadata()
                mutate(meta)
                self.save_metadata(meta)
                with self.assertRaises(RuntimeError):
                    self.fusion(self.fixed_infer, 18, False)

    def test_bad_array_faults_fail(self):
        for fault in ('nan', 'inf', 'shape', 'corrupt'):
            with self.subTest(fault=fault):
                logits, labels, ids = self.fixture()
                if fault in ('nan', 'inf'):
                    logits[0, 0] = np.nan if fault == 'nan' else np.inf
                if fault == 'shape':
                    logits = logits[:17]
                if fault == 'corrupt':
                    self.path.write_bytes(b'not-a-numpy-file')
                else:
                    np.save(self.path, logits)
                meta = self.metadata()
                meta['logit_sha256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()
                self.save_metadata(meta)
                with self.assertRaises((RuntimeError, ValueError)):
                    integrity.load_comp(self.path, ids, labels, self.run_id)

    def test_comp_writer_rejects_partial_ensemble(self):
        with self.assertRaisesRegex(RuntimeError, 'member set differs'):
            integrity.write_comp(self.path, np.zeros((3, 18)), integrity.LABELS, ['a', 'b', 'c'], 1, {}, 0, self.run_id)
        self.assertFalse(self.path.exists())

    def test_real_add_member_parity_and_faults(self):
        def load_function(code):
            tree = ast.parse(code.split('\n', 1)[1])
            node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'add_member')
            env = {'np': np, 'LOGIT': np.zeros((18, 18)), 'W': integrity.BASE_WEIGHTS,
                   'TE': list(range(18)), 'K': 18, 'T0': 0, 'time': SimpleNamespace(time=lambda: 1),
                   'MEMBER_AUDIT': {}, 'require': integrity.require, 'print': lambda *a, **k: None}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-add-member>', 'exec'), env)
            return env
        for rows in (None, np.arange(18) < 3):
            before, after = load_function(self.original[3]), load_function(self.patched[3])
            prob = self.rng.uniform(0.1, 1, size=(18 if rows is None else 3, 18))
            prob /= prob.sum(1, keepdims=True)
            before['add_member']('stylegram', prob, rows)
            after['add_member']('stylegram', prob, rows)
            np.testing.assert_array_equal(before['LOGIT'], after['LOGIT'])
            with self.assertRaisesRegex(RuntimeError, 'duplicate member'):
                after['add_member']('stylegram', prob, rows)
        for fault in ('shape', 'nan', 'negative', 'zero_sum'):
            after = load_function(self.patched[3])
            prob = np.full((18, 18), 1 / 18)
            if fault == 'shape':
                prob = prob[:17]
            elif fault == 'nan':
                prob[0, 0] = np.nan
            elif fault == 'negative':
                prob[0, 0] = -0.1
            else:
                prob[0] = 0
            with self.assertRaises(RuntimeError):
                after['add_member']('stylegram', prob)

    def test_notebook_initialization_removes_stale_outputs(self):
        init = [builder.source(c) for c in self.notebook['cells'] if c['cell_type'] == 'code'][1]
        with working(self.dir):
            names = ['submission.parquet', 'comp_logit.npy', 'comp_meta.json']
            for name in names:
                Path(name).write_bytes(b'stale')
            exec(init, {})
            self.assertTrue(all(not Path(name).exists() for name in names))

    def test_child_failure_timeout_and_stale_files(self):
        with working(self.dir):
            out = Path('submission.parquet')
            for mode in ('exit', 'timeout', 'missing', 'budget', 'validator'):
                with self.subTest(fault=mode):
                    out.write_bytes(b'old-submission')
                    code = "from pathlib import Path; Path('submission.parquet').write_bytes(b'partial')"
                    timeout = 10
                    validator = None
                    if mode == 'exit':
                        code += '; raise SystemExit(7)'
                    elif mode == 'timeout':
                        code += '; import time; time.sleep(5)'
                        timeout = 0.2
                    elif mode == 'missing':
                        code = 'pass'
                    elif mode == 'budget':
                        timeout = 0
                    elif mode == 'validator':
                        validator = lambda: integrity.require(False, 'synthetic invalid submission')
                    with self.assertRaises((subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError)):
                        integrity.run_checked([sys.executable, '-c', code], [out], timeout, validator)
                    self.assertFalse(out.exists())

    def test_clear_outputs_respects_working_directory(self):
        outside = self.dir / 'outside.txt'
        outside.write_text('preserve')
        inside = self.dir / 'child'
        inside.mkdir()
        with working(inside):
            with self.assertRaises(RuntimeError):
                integrity.clear_outputs([outside])
        self.assertEqual(outside.read_text(), 'preserve')

    def test_submission_schema_and_successful_child(self):
        with working(self.dir):
            pd.DataFrame({'id': ['a', 'b']}).to_csv('test.csv', index=False)
            sub = pd.DataFrame({'id': ['a', 'b'], 'model': integrity.LABELS[:2]})
            sub.to_parquet('submission.parquet', index=False)
            integrity.validate_submission('submission.parquet', 'test.csv')
            for fault in ('order', 'label', 'columns', 'duplicate'):
                bad = sub.copy()
                if fault == 'order':
                    bad = bad.iloc[::-1]
                elif fault == 'label':
                    bad.loc[0, 'model'] = 'invalid'
                elif fault == 'columns':
                    bad['unexpected'] = 0
                elif fault == 'duplicate':
                    bad.loc[1, 'id'] = 'a'
                bad.to_parquet('submission.parquet', index=False)
                with self.assertRaises(RuntimeError):
                    integrity.validate_submission('submission.parquet', 'test.csv')
            integrity.run_checked([sys.executable, '-c', "from pathlib import Path; Path('ok.txt').write_text('ok')"], ['ok.txt'], 10)
            self.assertEqual(Path('ok.txt').read_text(), 'ok')

    def test_build_source_and_launch_preservation(self):
        self.assertEqual(self.patched[0], self.original[0])
        self.assertEqual(self.patched[1], self.original[1])
        for source in self.original[2:4]:
            self.assertTrue(source.startswith('%%writefile'))
        # Generated argv is executed with fake subprocess dispatch and baseline variables.
        original_line = next(line for line in self.original[5].splitlines() if line.startswith('!python infer.py '))
        expected_tokens = re.findall(r"'[^']*'|\S+", original_line)[2:]
        config_prefix = self.original[5].split('import pandas as pd')[0]
        env = {name: name for name in ('B14', 'B8', 'BL', 'B25_14', 'N14B', 'Q25_14', 'DEB', 'DATA', 'EXTRA', 'POOL', 'ATLAS')}
        env.update(AQ14V5=['14'], AQ8V5=['8'], ALL8V5=['ll'], COMP_SEC=1, sys=sys, os=os, json=json)
        for name in ('N14BC', 'Q25_14C'):
            env[name] = {'keep': 24, 'layers': [2, 4], 'max_len': 384, 'eos_id': 1}
        exec(config_prefix, env)
        env['ATLAS_BUDGET'] = int(8.3 * 3600 - env['COMP_SEC'] - 600)
        expected = [str(eval(t.strip("'")[1:-1], env)) if t.strip("'").startswith('{') else t.strip("'") for t in expected_tokens]
        with patch.object(integrity, 'run_checked') as run, patch.object(pd, 'read_parquet', return_value=pd.DataFrame()):
            exec(self.patched[5], env)
        self.assertEqual(run.call_args.args[0], [sys.executable, 'infer.py'] + expected)
        self.assertEqual(builder.build(), builder.build())
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        # A modified baseline is rejected before the patch can silently drift.
        changed = json.loads(builder.BASELINE.read_bytes())
        changed['cells'] = [builder.code_cell(c + ('\n# drift' if i == 0 else '')) for i, c in enumerate(self.original)]
        bad = self.dir / 'changed.ipynb'
        bad.write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError, 'Baseline source differs'):
            builder.build(bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
