"""Artificial CPU fixtures only; executes the actual extracted l24 function."""
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import contextlib
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_r18_candidate as builder
import r18_runtime_contract as contract
sys.modules['r18_contract'] = contract
torch.set_num_threads(1)
REPORT = {'real_encoder_forwards': 0, 'training_fits': 0, 'official_rows_opened': 0,
          'protected_holdout_accesses': 0, 'fixture_forwards': 0, 'fixture_base_loads': 0}
BASE_WEIGHT = torch.diag(torch.tensor([1.0, 1.1, 1.2, 1.3], dtype=torch.float16))


class Toy(torch.nn.Module):
    def __init__(self, overflow=False):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float16)
        self.proj.weight.data.copy_(BASE_WEIGHT)
        self.overflow = overflow

    def forward(self, input_ids, attention_mask, **kwargs):
        REPORT['fixture_forwards'] += 1
        x = input_ids.to(self.proj.weight.dtype)
        x = torch.stack([x, x * 0.25 + 0.3, x * 0.5 + 0.7, x * 0.125 - 0.4], -1)
        z = self.proj(x)
        if self.overflow:
            self.overflow = False
            z = z * float('inf')
        return SimpleNamespace(hidden_states=(z, z))


class Tokenizer:
    pad_token = 'p'
    eos_token = 'e'
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        return {'input_ids': [[1, len(x) % 4 + 2, 3] for x in texts]}


def execute(code, adapters, groups=True, train=True, overflow=False, trace=None):
    starts, refs = [], []

    def load(*args, **kwargs):
        gc.collect()
        if any(ref() is not None for ref in refs):
            raise AssertionError('prior fixture model still alive before load')
        m = Toy(overflow)
        refs.append(weakref.ref(m))
        REPORT['fixture_base_loads'] += 1
        return m

    from safetensors.torch import load_file as actual_load
    def load_adapter(path):
        starts.append(refs[-1]().proj.weight.detach().clone())
        return actual_load(path)

    namespace = {'torch': torch, 'np': np, 'gc': gc, 'os': os, 'json': json,
        'AutoTokenizer': SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        'AutoModel': SimpleNamespace(from_pretrained=load), 'log': lambda *a: None}
    cfg = SimpleNamespace(from_pretrained=lambda *a: SimpleNamespace(num_hidden_layers=1))
    with patch.dict(sys.modules, {'transformers': SimpleNamespace(AutoConfig=cfg)}), \
         patch('safetensors.torch.load_file', load_adapter), \
         patch.dict(os.environ, {'R18_TRACE_PATH': str(trace) if trace else ''}), contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)
        result = namespace['l24_predict']({'name': 'toy', 'path': 'artificial-only', 'keep': 1,
            'layers': [1], 'adapters': [str(p) for p in adapters], 'bs': 1},
            ['short', 'artificial longer', 'third'], groups=np.array([0, 0, 1]) if groups else None,
            train_texts=['artificial train', 'other'] if train else None)
    gc.collect()
    assert all(ref() is None for ref in refs)
    return result, starts


class R18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.adapters = []
        for i in range(6):
            folder = cls.directory / f'toy_{i}'
            folder.mkdir()
            (folder / 'adapter_config.json').write_text(json.dumps({'lora_alpha': 1, 'r': 1}))
            save_file({'base_model.model.proj.lora_A.weight': torch.eye(4),
                'base_model.model.proj.lora_B.weight': torch.eye(4) * (0.0004 * (i + 1))},
                str(folder / 'adapter_model.safetensors'))
            h = {'norms.0.weight': torch.ones(4), 'norms.0.bias': torch.zeros(4)}
            for prefix in ('plain', 'centered'):
                h[prefix + '.weight'] = torch.arange(72, dtype=torch.float32).reshape(18, 4).sin()
                h[prefix + '.bias'] = torch.arange(18, dtype=torch.float32) * 0.01
            torch.save(h, folder / 'heads.pt')
            cls.adapters.append(folder)
        nb = json.loads(builder.frozen_bytes(builder.BASE, builder.BASE_SHA256))
        cells = [builder.source(c) for c in nb['cells'] if c['cell_type'] == 'code']
        cls.original = builder.function_source(cells[3].split('\n', 1)[1])
        cls.candidate = builder.isolate_function(cls.original)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_fp16_noninvertibility(self):
        w = torch.tensor([1.0], dtype=torch.float16)
        delta = torch.tensor([0.0004], dtype=torch.float16)
        recovered = (w + delta) - delta
        self.assertFalse(torch.equal(w, recovered))
        REPORT['numeric_example'] = {'base': w.item(), 'delta': delta.item(), 'after_add_subtract': recovered.item()}

    def test_02_actual_baseline_drift_and_candidate_freshness(self):
        _, old_starts = execute(self.original, self.adapters)
        _, starts = execute(self.candidate, self.adapters)
        self.assertTrue(all(torch.equal(s, BASE_WEIGHT) for s in starts))
        self.assertTrue(any(not torch.equal(s, BASE_WEIGHT) for s in old_starts[1:]))
        REPORT['baseline_adapter_start_max_abs_drift'] = [float((s - BASE_WEIGHT).abs().max()) for s in old_starts]
        REPORT['candidate_adapter_start_max_abs_drift'] = [float((s - BASE_WEIGHT).abs().max()) for s in starts]

    def test_03_matches_independent_fresh_reference_and_features(self):
        actual, _ = execute(self.candidate, self.adapters)
        references = [execute(self.original, [ad])[0] for ad in self.adapters]
        for index in (0, 1):
            expected = sum(r[index] / 6 for r in references)
            np.testing.assert_array_equal(actual[index], expected)
        for a, b in zip(actual[2], references[-1][2]):
            np.testing.assert_array_equal(a, b)

    def test_04_single_adapter_exact_parity(self):
        a, _ = execute(self.original, self.adapters[:1], groups=False, train=False)
        b, _ = execute(self.candidate, self.adapters[:1], groups=False, train=False)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertIsNone(b[1])

    def test_05_bf16_retry_isolated_between_adapters(self):
        actual, starts = execute(self.candidate, self.adapters[:2], overflow=True)
        references = [execute(self.original, [ad], overflow=True)[0] for ad in self.adapters[:2]]
        self.assertTrue(all(torch.equal(s, BASE_WEIGHT) for s in starts))
        np.testing.assert_array_equal(actual[0], sum(r[0] / 2 for r in references))

    def test_06_fingerprint_and_lifecycle_receipt(self):
        trace = self.directory / 'trace.jsonl'
        execute(self.candidate, self.adapters, trace=trace)
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual([e['event'] for e in events], ['before_load', 'loaded', 'released'] * 6)
        self.assertEqual(len({e['sample_sha256'] for e in events if e['event'] == 'loaded'}), 1)

    def test_07_frozen_non_lifecycle_source_and_inputs(self):
        old = json.loads(builder.frozen_bytes(builder.BASE, builder.BASE_SHA256))
        new = builder.build()
        self.assertEqual(old['metadata']['kaggle'], new['metadata']['kaggle'])
        get = lambda nb, name: next(builder.source(c) for c in nb['cells'] if builder.source(c).startswith('%%writefile ' + name + '\n'))
        for name in ('gauss.py', 'comp_stack.py'):
            self.assertEqual(get(old, name), get(new, name))
        self.assertEqual(get(old, 'infer.py').replace(self.original, self.candidate), get(new, 'infer.py'))
        self.assertEqual(len([c for c in new['cells'] if c['cell_type'] == 'code']), 9)

    def test_08_runtime_missing_members_and_fallback_rejected(self):
        text = '\n'.join(n + ' done' for n in ['q14v5', 'q8v5', 'n14b', 'q25_14', 'deb'])
        for f in ('q14v5', 'q8v5'):
            text += '\n' + '\n'.join(f'l24 {f}: adapter {n} done' for n in [f'{f}_f{i}' for i in range(5)] + [f])
        text += '\ncomp stack loaded (3, 18)\natlas artificial mixed 2/3 rows at 0.1'
        self.assertEqual(contract.validate_main_log(text)['atlas']['scored_rows'], 2)
        for bad in (text.replace('q8v5_f3', 'wrong'), text + '\nATLAS SKIPPED: failure', text.replace('2/3', '0/3')):
            with self.assertRaises(ValueError):
                contract.validate_main_log(bad)
        with self.assertRaises(ValueError):
            contract.validate_comp_log('SKIP required: checkpoint not attached\nCOMP STACK DONE 1s')

    def test_09_live_model_guard(self):
        lifecycle = contract.Lifecycle('toy')
        with contextlib.redirect_stdout(io.StringIO()):
            lifecycle.before_load('artificial')
            model = Toy()
            lifecycle.loaded(model)
            with self.assertRaises(RuntimeError):
                lifecycle.released()
            with self.assertRaises(RuntimeError):
                lifecycle.before_load('next')

    def test_10_preflight_rejects_cpu_offload(self):
        with patch.dict(os.environ, {'R18_REQUIRE_CUDA': '1'}), contextlib.redirect_stdout(io.StringIO()):
            lifecycle = contract.Lifecycle('toy')
            lifecycle.before_load('artificial')
            with self.assertRaisesRegex(RuntimeError, 'GPU-resident'):
                lifecycle.loaded(Toy())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(R18Tests))
    REPORT.update({'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'passed': result.wasSuccessful()})
    destination = ROOT / 'outputs/r18'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'local_numeric_receipt.json').write_text(json.dumps(REPORT, indent=2) + '\n')
    print(json.dumps(REPORT, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
