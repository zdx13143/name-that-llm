"""Verify fixed public recipe fidelity and synthetic postprocessing behavior."""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments'))
import build_r12_reproduction as builder
from r11_integrity import LABELS


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads(builder.REFERENCE.read_bytes())
        cls.raw = [builder.source(c) for c in cls.ref['cells'] if c['cell_type'] == 'code']
        cls.nb = builder.build()
        cls.candidate = [builder.source(c) for c in cls.nb['cells'] if c['cell_type'] == 'code'][1:]

    def setUp(self):
        temp_root = ROOT / 'outputs/r12_test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def test_exact_score_source_and_inputs_preserved(self):
        for i in (0, 1, 2, 4, 6):
            self.assertEqual(self.candidate[i], self.raw[i])
        self.assertEqual(self.candidate[3].replace(builder.NEW_SIGNAL, builder.OLD_SIGNAL), self.raw[3])
        self.assertEqual(self.candidate[5].replace(builder.NEW_COMP, builder.OLD_COMP), self.raw[5])
        self.assertEqual(self.nb['metadata']['kaggle']['dataSources'], self.ref['metadata']['kaggle']['dataSources'])
        self.assertEqual(len(self.nb['metadata']['kaggle']['dataSources']), 31)
        self.assertEqual(self.nb['metadata']['kaggle']['dockerImageVersionId'], 31481)
        self.assertFalse(self.nb['metadata']['kaggle']['isInternetEnabled'])

    def test_reproducible_build_and_empty_outputs(self):
        content = (json.dumps(builder.build(), ensure_ascii=False, indent=1) + '\n').encode()
        self.assertEqual(content, (ROOT / 'notebooks/team_e651_r12_public654.ipynb').read_bytes())
        receipt = json.loads((ROOT / 'configs/r12_frozen_recipe_20261028.json').read_bytes())
        self.assertEqual(hashlib.sha256(content).hexdigest(), receipt['candidate_sha256'])
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])

    def typography(self, source, confident=False, uniform=False):
        block = source[source.index('if not a.sim:\n'):]
        train_text = ['Synthetic — example “text”; ' + ('However,' if i % 2 else 'Overall,') for i in range(360)]
        if uniform:
            train_text = ['Identical synthetic training sentence.'] * 360
        test_text = ['Synthetic “ example.', 'No special markers', 'However, synthetic — text;', 'Overall, synthetic text.']
        rng = np.random.default_rng(12)
        raw = rng.normal(0, 0.2, size=(4, 18))
        if confident:
            raw[:, 3] = 20
        test = pd.DataFrame({'id': ['s0', 's1', 's2', 's3'], 'text': test_text})
        env = {'a': SimpleNamespace(sim='', out=str(self.dir / 'submission.parquet')),
               'sub': pd.DataFrame({'id': test.id, 'model': [LABELS[i] for i in raw.argmax(1)]}),
               'train': pd.DataFrame({'text': train_text}), 'test': test,
               'y': np.arange(360) % 18, 'LABELS': LABELS, 'np': np, 'pd': pd,
               '_v38_before_hard': raw.copy(), 'log': lambda *args: None}
        exec(compile(block, '<actual-typography>', 'exec'), env)
        return env

    def test_typography_fidelity_clipping_and_confidence(self):
        for confident in (False, True):
            before = self.typography(self.raw[3], confident)
            after = self.typography(self.candidate[3], confident)
            np.testing.assert_array_equal(before['_safe_logits'], after['_safe_logits'])
            self.assertEqual(before['_safe']['model'].tolist(), after['_safe']['model'].tolist())
            self.assertTrue(np.isfinite(after['_safe_logits']).all())
            self.assertTrue((np.abs(after['_evidence']) <= 0.35).all())
            self.assertTrue(((after['_gate'] >= 0) & (after['_gate'] <= 1)).all())
            self.assertTrue((after['_rate'] >= 0.015).all() and (after['_rate'] <= 0.985).all())
            self.assertEqual(after['_safe'].id.tolist(), ['s0', 's1', 's2', 's3'])
            if confident:
                np.testing.assert_array_equal(after['_safe_logits'], after['_v38_before_hard'])
        # Zero evidence when every class has identical marker rates.
        uniform = self.typography(self.candidate[3], uniform=True)
        self.assertTrue(np.allclose(uniform['_evidence'], 0))

    def test_partial_atlas_fusion_preserves_unscored_rows(self):
        code = self.candidate[3]
        block = code[code.index('_bundles = '):code.index('    Q = P.copy()', code.index('_bundles = '))]
        n = 9
        rng = np.random.default_rng(18)
        scores = rng.normal(size=(n, 18))
        atlas = rng.uniform(0.1, 1, size=(n, 18)); atlas /= atlas.sum(1, keepdims=True)
        for scored in (np.zeros(n, dtype=bool), np.ones(n, dtype=bool), np.arange(n) % 2 == 0):
            env = {'a': SimpleNamespace(atlas_dir='fixed', atlas_w='0.10', sinkhorn=1, sk_div=1,
                                        atlas_margin=0, atlas_budget=100, comp_w=0.15),
                   'test': list(range(n)), 'LP': {'one': scores}, 'tot': scores.copy(),
                   'COMP_P': None, 'np': np, 'log': lambda *args: None, 'gpu_mem': lambda: '',
                   'atlas_scores': lambda *args: (atlas.copy(), scored.copy())}
            exec(compile(block, '<actual-partial-atlas-fusion>', 'exec'), env)
            reference = softmax(scores, axis=1)
            expected = reference.copy()
            expected[scored] += 0.10 * (atlas[scored] - reference[scored])
            expected /= expected.sum(1, keepdims=True)
            np.testing.assert_allclose(env['P'], expected, rtol=0, atol=2e-16)
            np.testing.assert_allclose(env['P'][~scored], reference[~scored], rtol=0, atol=2e-16)

    def test_zero_weight_tfidf_has_no_fit(self):
        code = self.candidate[3].split('\n', 1)[1]
        tree = ast.parse(code)
        node = next(node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'a.tfidf_w > 0')
        env = {'a': SimpleNamespace(tfidf_w=0), 'LP': {}, 'test': list(range(17)), 'np': np, 'log': lambda *args: None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<zero-weight-tfidf>', 'exec'), env)
        np.testing.assert_array_equal(env['LP']['tfidf'], np.zeros((17, 18)))

    def test_normal_save_is_only_placeholder(self):
        train = pd.DataFrame({'model': LABELS * 2})
        test = pd.DataFrame({'id': ['s1', 's2', 's3']})
        train.to_csv(self.dir / 'train.csv', index=False)
        test.to_csv(self.dir / 'test.csv', index=False)
        # Execute only the actual save-path branch with an absolute output target.
        node = ast.parse(self.candidate[6]).body[0]
        code = ast.unparse(ast.Module(body=node.orelse, type_ignores=[]))
        code = code.replace("'submission.parquet'", repr(str(self.dir / 'submission.parquet')))
        import os
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, {'DATA': str(self.dir), 'os': os})
        sub = pd.read_parquet(self.dir / 'submission.parquet')
        self.assertEqual(sub.id.tolist(), ['s1', 's2', 's3'])
        self.assertEqual(sub.model.nunique(), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
