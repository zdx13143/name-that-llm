"""Synthetic R17 release checks; no official data, weights or model forward."""

import ast
import contextlib
import glob
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_r17_candidate as build
import r17_runtime_contract as runtime


class R17Restore(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'outputs/r17/artificial'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'input/llama-3.1/transformers/8b/1'
        self.base.mkdir(parents=True)
        self.contract = {'kaggle_ref': 'synthetic/llama/base/1', 'files': {},
                         'adapters': [f'll8v5_f{i}' for i in range(5)] + ['ll8v5'],
                         'adapter_dataset_directory': 'nlm-l24-adapters-v5',
                         'adapter_dataset_ref': 'synthetic/nlm-l24-adapters-v5',
                         'adapter_dataset_version': 1, 'adapter_file_sizes': {},
                         'adapter_allowed_base_refs': ['meta-llama/Llama-3.1-8B']}
        config = json.dumps({'model_type': 'llama', 'hidden_size': 4096,
                             'num_hidden_layers': 32, 'vocab_size': 128256}).encode()
        for name, data, algorithm in [('config.json', config, 'git-blob-sha1'),
                                      ('fake.safetensors', b'synthetic weight fixture', 'sha256')]:
            (self.base / name).write_bytes(data)
            digest = (hashlib.sha256(data).hexdigest() if algorithm == 'sha256'
                      else hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest())
            self.contract['files'][name] = {'size': len(data), 'algorithm': algorithm, 'digest': digest}
        self.adapters = []
        for name in self.contract['adapters']:
            path = self.root / 'input/nlm-l24-adapters-v5' / name
            path.mkdir(parents=True)
            (path / 'adapter_config.json').write_text(json.dumps({'r': 8, 'lora_alpha': 16, 'base_model_name_or_path': 'meta-llama/Llama-3.1-8B'}), encoding='utf-8')
            (path / 'adapter_model.safetensors').write_bytes(b'fake adapter')
            (path / 'heads.pt').write_bytes(b'fake head')
            self.contract['adapter_file_sizes'][name] = {p.name: p.stat().st_size for p in path.iterdir()}
            self.adapters.append(str(path))

    def test_exact_bytes_and_all_six_adapters(self):
        result = runtime.verify_llama_assets(self.base, self.adapters, self.contract)
        self.assertEqual(len(result['adapters']), 6)
        self.assertFalse(result['weight_forward_executed'])
        self.assertFalse(result['adapters'][0]['files']['heads.pt']['independent_expected_digest_available'])

    def test_equal_size_corruption_is_rejected(self):
        weight = self.base / 'fake.safetensors'
        weight.write_bytes(b'X' * weight.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'content hash mismatch'):
            runtime.verify_llama_assets(self.base, self.adapters, self.contract)

    def test_missing_or_duplicate_adapter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'distinct fold adapters'):
            runtime.verify_llama_assets(self.base, self.adapters[:-1] + [self.adapters[0]], self.contract)
        (Path(self.adapters[-1]) / 'heads.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'missing adapter file'):
            runtime.verify_llama_assets(self.base, self.adapters, self.contract)

    def test_absent_base_and_expired_budget_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            runtime.verify_llama_assets(self.root / 'absent', self.adapters, self.contract)
        with self.assertRaises(TimeoutError):
            runtime.verify_llama_assets(self.base, self.adapters, self.contract, time.monotonic() - 30000)

    def test_wrong_declared_adapter_base_is_rejected(self):
        path = Path(self.adapters[0]) / 'adapter_config.json'
        cfg = json.loads(path.read_bytes())
        cfg['base_model_name_or_path'] = 'wrong-model'
        path.write_text(json.dumps(cfg), encoding='utf-8')
        self.contract['adapter_file_sizes'][path.parent.name][path.name] = path.stat().st_size
        with self.assertRaisesRegex(ValueError, 'unverified adapter base identity'):
            runtime.verify_llama_assets(self.base, self.adapters, self.contract)

    @staticmethod
    def successful_main_log(scored=3, eligible=4):
        lines = []
        for family in ('q14v5', 'q8v5', 'll8v5'):
            for suffix in ('_f0', '_f1', '_f2', '_f3', '_f4', ''):
                lines.append(f'123.4s   l24 {family}: adapter {family}{suffix} done (bf16 redo batches 0)')
        lines += [f'200.0s {name} done gpu0:0.0GB' for name in ('q14v5', 'q8v5', 'll8v5', 'n14b', 'q25_14', 'deb')]
        lines += ['comp stack loaded (123s)', f'atlas /input/atlas mixed {scored}/{eligible} rows at 0.1 gpu0:0.0GB']
        return '\n'.join(lines)

    def test_runtime_requires_all_18_v5_adapters(self):
        text = self.successful_main_log()
        result = runtime.validate_main_log(text)
        self.assertEqual(sum(len(v) for v in result['v5_adapters'].values()), 18)
        with self.assertRaisesRegex(ValueError, 'incomplete v5 adapter family'):
            runtime.validate_main_log('\n'.join(line for line in text.splitlines() if 'adapter ll8v5_f3 ' not in line))
        with self.assertRaisesRegex(ValueError, 'main member did not complete'):
            runtime.validate_main_log(text.replace('deb done', 'deb started'))

    def test_runtime_rejects_error_fallback_and_empty_atlas(self):
        for error in ('COMP SKIPPED: test', 'ATLAS SKIPPED: test', 'atlas invalid scores in batch 1'):
            with self.subTest(error=error), self.assertRaises(ValueError):
                runtime.validate_main_log(self.successful_main_log() + '\n' + error)
        with self.assertRaisesRegex(ValueError, 'empty ATLAS'):
            runtime.validate_main_log(self.successful_main_log(0, 4))
        self.assertEqual(runtime.validate_main_log(self.successful_main_log(0, 0))['atlas']['eligible_rows'], 0)

    def test_comp_allows_only_known_missing_checkpoint(self):
        good = 'SKIP lora24_public: checkpoint not attached (fixture)\nCOMP STACK DONE 123s'
        self.assertEqual(runtime.validate_comp_log(good)['missing_checkpoint_members'], ['lora24_public'])
        with self.assertRaisesRegex(ValueError, 'unexpected missing COMP'):
            runtime.validate_comp_log(good.replace('lora24_public', 'lora24_seed7'))
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            runtime.validate_comp_log('started')

    def test_subprocess_failure_and_timeout_are_not_swallowed(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIn('fixture', runtime.run_logged([sys.executable, '-c', "print('fixture')"], self.root / 'ok.log', 10))
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.run_logged([sys.executable, '-c', "print('failure');raise SystemExit(2)"], self.root / 'failure.log', 10)
            with self.assertRaises(subprocess.TimeoutExpired):
                runtime.run_logged([sys.executable, '-c', 'import time;time.sleep(5)'], self.root / 'timeout.log', 0.1)
        self.assertIn('failure', (self.root / 'failure.log').read_text())

    def test_prediction_parity_and_pending_release(self):
        base = json.loads(build.BASE.read_bytes())
        candidate = build.build()
        a = [build.source(c) for c in base['cells'] if c['cell_type'] == 'code']
        b = [build.source(c) for c in candidate['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(b), 8)
        for i in (2, 3, 4):
            self.assertEqual(a[i], b[i + 1])
        self.assertIn('_safe_logits = _raw + 0.5 * _gate[:, None] * _evidence', b[4])
        self.assertEqual(base['metadata']['kaggle'], candidate['metadata']['kaggle'])
        self.assertFalse(candidate['metadata']['r17']['release_ready'])
        original = [build.source(c) for c in json.loads(build.ORIGINAL.read_bytes())['cells'] if c['cell_type'] == 'code']
        def models(code):
            return next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'MODELS' for t in n.targets))
        self.assertEqual(ast.dump(models(original[6])), ast.dump(models(b[7])))
        self.assertIn('ATLAS_BUDGET = int(remaining_seconds(R17_START) - 600)', b[7])

    def test_only_checkout_line_endings_can_differ(self):
        frozen = build.BASE.read_bytes().replace(b'\r\n', b'\n')
        path = self.root / 'base.ipynb'
        path.write_bytes(frozen.replace(b'\n', b'\r\n'))
        with mock.patch.object(build, 'BASE', path):
            self.assertFalse(build.build()['metadata']['r17']['release_ready'])
            path.write_bytes(frozen.replace(b'0.5 * _gate', b'0.6 * _gate', 1))
            with self.assertRaisesRegex(ValueError, 'frozen scored R16'):
                build.build()
        original = build.ORIGINAL.read_bytes().replace(b'\r\n', b'\n')
        path.write_bytes(original.replace(b'"w\\": 1.0', b'"w\\": 2.0', 1))
        # Any content change to the restoration source must fail its own guard.
        path.write_bytes(path.read_bytes() + b' ')
        with mock.patch.object(build, 'ORIGINAL', path), self.assertRaises(ValueError):
            build.build()

    def test_actual_generated_initialization_with_synthetic_assets(self):
        inp = self.root / 'input'
        work = self.root / 'working'
        work.mkdir()
        def touch(name, data='{}'):
            path = inp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding='utf-8')
        for name in ('competitions/name-that-llm/train.csv', 'qwen-3/8b-base/1/config.json',
                     'qwen-3/14b-base/1/config.json', 'qwen2.5/14b/1/config.json',
                     'nlm_matched_clean.parquet', 'nlm_pool_v5.parquet',
                     'atlas/atlas_bundle.json', 'deberta_v3_large/meta.json'):
            touch(name)
        cfg = {'keep': 24, 'layers': [2, 4], 'max_len': 384, 'eos_id': 1}
        for family in ('lora_qwen3_14b', 'lora_qwen25_14b'):
            touch(family + '/meta.json', json.dumps({'cfg': cfg}))
        for family in ('q8v5', 'q14v5'):
            for suffix in ('', '_f0', '_f1', '_f2', '_f3', '_f4'):
                touch(family + suffix + '/heads.pt')
        candidate = build.build()
        codes = [build.source(c) for c in candidate['cells'] if c['cell_type'] == 'code']
        production_contract = json.loads(build.CONTRACT.read_bytes())
        init = codes[2].replace(repr(production_contract), repr(self.contract), 1)
        init = init.replace('/kaggle/input', inp.as_posix()).replace('/kaggle/working', work.as_posix())
        native_glob = glob.glob
        env = {'R17_START': time.monotonic()}
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(sys.modules, {'r17_contract': runtime}), mock.patch('glob.glob', side_effect=lambda *a, **kw: [p.replace('\\', '/') for p in native_glob(*a, **kw)]):
            exec(compile(init, '<synthetic-r17-init>', 'exec'), env)
        self.assertEqual(len(env['ALL8V5']), 6)
        self.assertEqual(len(env['AQ8V5']), 6)
        self.assertEqual(len(env['AQ14V5']), 6)
        self.assertEqual(env['BL'], self.base.as_posix())
        self.assertTrue((work / 'r17_asset_verification.json').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
