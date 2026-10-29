"""Synthetic envelope and source-transform checks; no model or official payload."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_r15_dev_export import fixture, mod as dev
spec = importlib.util.spec_from_file_location('s7_adapter', ROOT / 'scripts/build_r15_s7_adapter.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class S7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / 'reference_sources/r15_p0/s7_v1.py.txt').read_bytes()
        cls.original = mod.archived_worker(cls.source)
        cls.worker = mod.adapt_worker(cls.source)

    def setUp(self):
        self.data, self.protocol, *_ = fixture()
        self.plan = dev.nested_plan(self.data, self.protocol)

    def test_all_thirty_envelopes_exclude_correct_groups(self):
        outer_predictions = []
        for outer in range(5):
            outer_valid = set(self.plan['folds'][outer]['valid_rows'])
            inner_predictions = []
            for inner in [None, 0, 1, 2, 3, 4]:
                env = mod.materialize_envelope(self.data, self.plan, outer, inner)
                fit, pred = set(env['fit_positions']), set(env['predict_positions'])
                self.assertFalse(fit & pred)
                self.assertFalse({r['prompt_id'] for r in env['fit']} & {r['prompt_id'] for r in env['predict']})
                self.assertEqual(env['anchor_positions'], env['fit_positions'][:3])
                self.assertEqual(env['labels'], list(mod.LABELS))
                self.assertTrue(all(set(r) == {'id', 'prompt_id', 'text'} for r in env['predict']))
                if inner is None:
                    self.assertEqual(pred, outer_valid)
                    outer_predictions.extend(pred)
                else:
                    self.assertFalse((fit | pred) & outer_valid)
                    inner_predictions.extend(pred)
            self.assertEqual(sorted(inner_predictions), sorted(self.plan['folds'][outer]['train_rows']))
        self.assertEqual(sorted(outer_predictions), list(range(2159)))

    def test_prediction_truth_is_never_read(self):
        for i in self.plan['folds'][0]['valid_rows']:
            self.data['truth'][i] = 100000  # would fail if converted to a model name
        env = mod.materialize_envelope(self.data, self.plan, 0)
        self.assertEqual(len(env['predict']), len(self.plan['folds'][0]['valid_rows']))
        for row, i in zip(env['fit'], env['fit_positions']):
            self.assertEqual(row['model'], self.data['labels'][self.data['truth'][i]])

    def test_invalid_ids_and_inner_escape_rejected(self):
        for outer, inner in [(True, None), (5, None), (-1, None), (0, True), (0, 5)]:
            with self.subTest(outer=outer, inner=inner), self.assertRaises(ValueError):
                mod.job_positions(self.plan, outer, inner)
        changed = copy.deepcopy(self.plan)
        changed['folds'][0]['inner'][0]['train_rows'].append(changed['folds'][0]['valid_rows'][0])
        with self.assertRaisesRegex(ValueError, 'escapes'):
            mod.job_positions(changed, 0, 0)

    def test_training_core_and_sampler_unchanged(self):
        def sampler(source):
            return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'triplets'))
        self.assertEqual(sampler(self.worker), sampler(self.original))
        def core(source):
            return source.split("    mc = AutoConfig", 1)[1].split('    model.eval(); norms.eval()', 1)[0]
        self.assertEqual(core(self.worker), core(self.original))
        compile(self.worker, '<synthetic-source-test>', 'exec')
        with self.assertRaisesRegex(ValueError, 'trainer changed'):
            mod.adapt_worker(self.source + b'\n')

    def test_label_free_predictions_paired_save_and_early_closed_gate(self):
        self.assertNotIn('pd.read_csv', self.worker)
        self.assertNotIn('fold_seed', self.worker)
        self.assertNotIn('valid_acc', self.worker)
        self.assertNotIn('y[valid]', self.worker)
        self.assertIn("features=Z[:3].astype(np.float16)", self.worker)
        self.assertIn("logits=lg[3:]", self.worker)
        self.assertIn("model.save_pretrained", self.worker)
        self.assertLess(self.worker.index('cfg, envelope = worker_inputs'), self.worker.index('import numpy as np, pandas as pd, torch'))
        with patch('builtins.open', side_effect=AssertionError('unexpected I/O')), self.assertRaisesRegex(ValueError, 'release is not implemented'):
            mod.worker_inputs('never-open-this-path')

    def test_original_sampler_operates_only_on_fit(self):
        for inner in [None, 0]:
            env = mod.materialize_envelope(self.data, self.plan, 0, inner)
            report = mod.verify_sampling(self.worker, env)
            self.assertTrue(report['sampler_fit_only_verified'])
            prompts = len({r['prompt_id'] for r in env['fit']})
            self.assertEqual(report['optimizer_steps_planned'], prompts * 3)

    def test_even_conservative_fp16_matrix_floor_exceeds_laptop(self):
        cfg = {'model_type': 'qwen3', 'hidden_size': 4096, 'num_hidden_layers': 36,
               'head_dim': 128, 'num_attention_heads': 32, 'num_key_value_heads': 8,
               'intermediate_size': 12288, 'vocab_size': 151936}
        result = mod.fp16_weight_lower_bound(cfg)
        self.assertGreater(result['fp16_bytes'], 8188 * 1024**2)
        self.assertGreater(result['fp16_gib'], 9.7)
        self.assertIn('activations', result['excludes'])


if __name__ == '__main__':
    unittest.main()
