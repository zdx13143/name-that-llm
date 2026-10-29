"""Exercise the actual frozen source blocks with synthetic data only."""

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_r14_reproduction as builder


def execute_nodes(nodes, env):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-source-block>', 'exec'), env)


class R14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(builder.REFERENCE.read_bytes())
        cls.raw = [builder.source(c) for c in cls.reference['cells'] if c['cell_type'] == 'code']
        cls.notebook = builder.build()
        cls.code = [builder.source(c) for c in cls.notebook['cells'] if c['cell_type'] == 'code']
        cls.labels = builder.labels_from_source(cls.raw[2])
        cls.infer_tree = ast.parse(cls.raw[2].split('\n', 1)[1])
        cls.comp_tree = ast.parse(cls.raw[3].split('\n', 1)[1])

    def setUp(self):
        base = ROOT / 'outputs/r14_test_tmp'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.validation = {'_R14_LABELS': self.labels}
        exec(builder.VALIDATION, self.validation)

    def test_frozen_scripts_inputs_and_main_ast(self):
        self.assertEqual(self.code[1:5], self.raw[:4])
        self.assertEqual(self.notebook['metadata']['kaggle']['dataSources'], self.reference['metadata']['kaggle']['dataSources'])
        self.assertEqual(builder.inputs_digest(self.notebook['metadata']['kaggle']['dataSources']), builder.INPUT_SHA256)
        self.assertEqual(len(self.notebook['metadata']['kaggle']['dataSources']), 31)
        main_body = ast.parse(self.code[6]).body[0].body[:-1]
        self.assertEqual(ast.dump(ast.Module(body=main_body, type_ignores=[])), ast.dump(ast.parse(self.raw[5])))
        comp_body = self.code[5].removeprefix('if RERUN:\n')
        import textwrap
        self.assertEqual(textwrap.dedent(comp_body).replace(builder.NEW_COMP, builder.OLD_COMP), self.raw[4])

    def test_pin_tampering_rejected(self):
        for change in ('source', 'inputs', 'image'):
            nb = json.loads(json.dumps(self.reference))
            if change == 'source':
                next(c for c in nb['cells'] if c['cell_type'] == 'code')['source'].append('\n# drift\n')
            elif change == 'inputs':
                nb['metadata']['kaggle']['dataSources'][1]['sourceId'] += 1
            else:
                nb['metadata']['kaggle']['dockerImageVersionId'] += 1
            path = self.dir / 'tampered.ipynb'
            path.write_text(json.dumps(nb), encoding='utf-8')
            with self.assertRaises(ValueError):
                builder.build(path)

    def test_reproducible_output_free_package(self):
        packed = (json.dumps(builder.build(), ensure_ascii=False, indent=1) + '\n').encode()
        self.assertEqual(packed, builder.OUTPUT.read_bytes())
        recipe = json.loads((ROOT / 'configs/r14_frozen_recipe_20261029.json').read_bytes())
        self.assertEqual(hashlib.sha256(packed).hexdigest(), recipe['candidate_sha256'])
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        self.assertTrue(self.notebook['metadata']['kaggle']['isGpuEnabled'])
        self.assertFalse(self.notebook['metadata']['kaggle']['isInternetEnabled'])
        self.assertEqual(self.notebook['metadata']['kaggle']['dockerImageVersionId'], 31499)

    def test_comp_validation_fails_closed(self):
        valid = {'ids': ['a', 'b'], 'labels': self.labels, 'seconds': 2}
        logits = np.zeros((2, 18))
        check = self.validation['_r14_validate_comp']
        check(valid, logits, ['a', 'b'])
        for meta, values in [({**valid, 'ids': ['b', 'a']}, logits),
                             ({**valid, 'labels': ['bad'] * 18}, logits),
                             ({**valid, 'seconds': float('nan')}, logits),
                             (valid, np.zeros((1, 18))), (valid, np.full((2, 18), np.nan))]:
            with self.assertRaises(ValueError):
                check(meta, values, ['a', 'b'])
        check({**valid, 'labels': list(reversed(self.labels))}, logits, ['a', 'b'])

    def test_final_validation_rejects_wrong_ids_and_labels(self):
        sub = pd.DataFrame({'id': ['a', 'b'], 'model': self.labels[:2]})
        check = self.validation['_r14_validate_submission']
        check(sub, ['a', 'b'])
        for wrong in [sub.iloc[::-1], sub.assign(model='bad'), sub.assign(id=['a', 'a']), sub.assign(extra=1)]:
            with self.assertRaises(ValueError):
                check(wrong, ['a', 'b'])

    def test_normal_save_only_placeholder(self):
        ids = ['s0', 's1', 's2']
        env = {**self.validation, 'RERUN': False, '_R14_TEST': pd.DataFrame({'id': ids}), '_r14pd': pd, '_R14_LABELS': self.labels}
        code = self.code[6].replace('/kaggle/working', self.dir.as_posix())
        with contextlib.redirect_stdout(io.StringIO()), mock.patch('subprocess.run', side_effect=AssertionError('No inference in Save')):
            exec(code, env)
        sub = pd.read_parquet(self.dir / 'submission.parquet')
        self.assertEqual(sub.id.tolist(), ids)
        self.assertEqual(sub.model.nunique(), 1)

    def test_normal_save_wrong_row_count_stops(self):
        pd.DataFrame({'id': range(4)}).to_csv(self.dir / 'test.csv', index=False)
        init = self.code[0].replace('/kaggle/input/competitions/name-that-llm', self.dir.as_posix())
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'three public placeholder'):
            exec(init, {})

    def test_comp_subprocess_failure_is_not_swallowed(self):
        (self.dir / 'comp_stack.py').write_text('raise SystemExit(7)\n', encoding='utf-8')
        real_run = subprocess.run
        env = {**self.validation, 'RERUN': True}
        with mock.patch.dict(os.environ), mock.patch('subprocess.run', side_effect=lambda args, **kw: real_run(args, cwd=self.dir, **kw)):
            with self.assertRaises(subprocess.CalledProcessError):
                exec(self.code[5], env)

    def test_hidden_launcher_matches_reference_on_fake_process(self):
        ids = ['s0', 's1', 's2', 's3']
        pd.DataFrame({'id': ids}).to_csv(self.dir / 'test.csv', index=False)
        fake = "import argparse,pandas as pd\np=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--out');a,_=p.parse_known_args();d=pd.read_csv(a.data+'/test.csv');d['model']='claude-fable-5';d.to_parquet(a.out,index=False)\n"
        (self.dir / 'infer.py').write_text(fake, encoding='utf-8')
        env = {name: name for name in ['B14', 'B8', 'BL', 'B25_14', 'N14B', 'Q25_14', 'DEB', 'EXTRA', 'POOL', 'ATLAS']}
        env.update({name: [name] for name in ['AQ14V5', 'AQ8V5', 'ALL8V5']})
        env.update({name: {'keep': 24, 'layers': [2,4], 'max_len': 384, 'eos_id': 1} for name in ['N14BC', 'Q25_14C']})
        env.update(self.validation)
        env.update({'DATA': str(self.dir), 'COMP_SEC': 2, 'RERUN': True, '_R14_TEST': pd.DataFrame({'id': ids}), 'json': json, 'os': os})
        real_run = subprocess.run
        outputs, commands = [], []
        for code in [self.raw[5], self.code[6]]:
            ns = env.copy()
            with contextlib.redirect_stdout(io.StringIO()), mock.patch('subprocess.run', side_effect=lambda args, **kw: real_run(args, cwd=self.dir, **kw)):
                exec(code.replace('/kaggle/working', self.dir.as_posix()), ns)
            outputs.append((self.dir / 'submission.parquet').read_bytes())
            commands.append(ns['cmd'])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(commands[0], commands[1])

    def test_family_gate_preserves_mass_and_confident_rows(self):
        nodes = [n for n in self.comp_tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_np_logsoftmax', '_apply_conditional_family_gate')]
        rng = np.random.default_rng(14)
        before = rng.normal(0, .1, (12,18))
        before[0,0] = 20
        env = {'np': np, 'LOGIT': before.copy(), 'GATES': {'claude': rng.normal(0,1,(12,5))}, 'FAMILY_IDX': {'claude': list(range(5))}, 'TE': list(range(12))}
        execute_nodes(nodes, env)
        with contextlib.redirect_stdout(io.StringIO()):
            env['_apply_conditional_family_gate']('claude', 'claude', 1.5)
        np.testing.assert_allclose(logsumexp(env['LOGIT'][:,:5],axis=1), logsumexp(before[:,:5],axis=1), atol=1e-12)
        np.testing.assert_array_equal(env['LOGIT'][:,5:], before[:,5:])
        np.testing.assert_allclose(env['LOGIT'][0], before[0], atol=1e-12)
        self.assertGreater(np.abs(env['LOGIT'][1:,:5]-before[1:,:5]).max(), 1e-4)

    def test_atlas_value_schedule_is_stable_and_cost_aware(self):
        function = next(n for n in self.infer_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'atlas_scores')
        node = next(n for n in ast.walk(function) if isinstance(n, ast.If) and 'a.atlas_schedule' in ast.unparse(n.test))
        def run(count, confidence=False):
            probs = np.full((count,18),1/18)
            if confidence:
                probs[:count//2] = 0.001
                probs[:count//2,0] = .983
            tokens = [[0]*64 for _ in range(count)]
            if not confidence:
                tokens[0] = [0]*384
            env = {'a': SimpleNamespace(atlas_schedule='value'), 'rows': np.arange(count), 'P': probs, 'COMP_P': None, 'LABELS': self.labels, 'np': np, '_enc': tokens, '_ord': np.arange(count), 'log': lambda *a: None}
            execute_nodes([node], env)
            return env['_ord']
        np.testing.assert_array_equal(run(49), np.arange(49))
        np.testing.assert_array_equal(run(60), np.r_[np.arange(1,60),0])
        np.testing.assert_array_equal(run(60,True), np.r_[np.arange(30,60),np.arange(30)])

    def test_partial_atlas_keeps_unscored_rows(self):
        code = self.raw[2]
        block = code[code.index('_bundles = '):code.index('    Q = P.copy()', code.index('_bundles = '))]
        rng = np.random.default_rng(14)
        scores = rng.normal(size=(9,18))
        atlas = softmax(rng.normal(size=(9,18)),axis=1)
        for scored in [np.zeros(9,bool), np.ones(9,bool), np.arange(9)%2==0]:
            env = {'a': SimpleNamespace(atlas_dir='fixed',atlas_w='.10',sinkhorn=.45,sk_div=1,atlas_margin=0,atlas_budget=100,comp_w=.15), 'test': list(range(9)), 'LP': {'one':scores}, 'tot':scores.copy(), 'COMP_P':None, 'np':np, 'log':lambda *a:None, 'gpu_mem':lambda:'', 'atlas_scores':lambda *args:(atlas.copy(),scored.copy())}
            exec(block,env)
            reference = softmax(scores,axis=1)
            expected = reference.copy()
            expected[scored] += .1*(atlas[scored]-reference[scored])
            np.testing.assert_allclose(env['P'],expected,atol=1e-15)
            np.testing.assert_allclose(env['P'][~scored],reference[~scored],atol=1e-15)

    def test_damped_balancing_respects_nonuniform_marginal(self):
        outer = next(n for n in self.infer_tree.body if isinstance(n,ast.If) and '_bundles' in ast.unparse(n.test) and 'COMP_P' in ast.unparse(n.test))
        node = next(n for n in outer.body if isinstance(n,ast.If) and ast.unparse(n.test)=='a.sinkhorn > 0 and len(test) >= 18')
        P = np.tile(np.r_[.5,np.full(17,.5/17)],(72,1))
        for rho in [0,.35]:
            env = {'a':SimpleNamespace(sinkhorn=.45,sinkhorn_target_mix=rho), 'test':list(range(72)), 'np':np, 'P':P.copy(),'Q':P.copy(),'y':np.arange(180)%18}
            execute_nodes([node],env)
            np.testing.assert_allclose(env['Q'].sum(1),1,atol=1e-12)
            np.testing.assert_allclose(env['Q'].mean(0),env['_target'],atol=1e-12)
            self.assertTrue(np.isfinite(env['tot']).all())
            if rho==0:
                np.testing.assert_allclose(env['Q'],P,atol=1e-12)
            else:
                self.assertGreater(env['_target'][0],1/18)

    def test_hierarchical_typography_is_bounded_and_gated(self):
        block = self.raw[2][self.raw[2].index('if not a.sim:\n'):]
        for confident,uniform in [(False,False),(True,False),(False,True)]:
            raw = np.zeros((4,18))
            if confident:
                raw[:,3] = 20
            texts = ['Same sentence.']*360 if uniform else [('However, — “text” ' if i%2 else 'Overall, simple text. ')*4 for i in range(360)]
            env={'a':SimpleNamespace(sim='',out=str(self.dir/'submission.parquet'),sinkhorn=.45,sinkhorn_target_mix=.35),'train':pd.DataFrame({'text':texts}),'test':pd.DataFrame({'id':list('abcd'),'text':['short','However, — “text” '*30,'plain text '*10,'### ``` > ']}),'y':np.arange(360)%18,'LABELS':self.labels,'np':np,'pd':pd,'re':re,'_v38_before_hard':raw,'log':lambda *a:None}
            exec(block,env)
            self.assertTrue(np.isfinite(env['_safe_logits']).all())
            self.assertLessEqual(np.abs(env['_evidence']).max(),.30)
            self.assertTrue(((env['_gate']>=0)&(env['_gate']<=1)).all())
            self.assertEqual(env['_safe'].id.tolist(),list('abcd'))
            if confident:
                np.testing.assert_array_equal(env['_safe_logits'],raw)
            if uniform:
                np.testing.assert_allclose(env['_evidence'],0,atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
