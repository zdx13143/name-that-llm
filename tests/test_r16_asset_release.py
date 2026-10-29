"""Artificial asset discovery and unchanged-member checks; no official data."""
import ast
import contextlib
import io
import glob
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import release_r16_runnable as release


class R16AssetRelease(unittest.TestCase):
    def test_initialization_with_only_active_artificial_assets(self):
        base = ROOT / 'outputs/r16/artificial_assets'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root = Path(tmp)
            inp = root / 'input'
            def touch(name, contents=''):
                p = inp / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(contents, encoding='utf-8')
            touch('competitions/name-that-llm/train.csv')
            for name in ['qwen-3/8b-base/1/config.json', 'qwen-3/14b-base/1/config.json', 'qwen2.5/14b/1/config.json',
                         'nlm_matched_clean.parquet', 'nlm_pool_v5.parquet', 'atlas/atlas_bundle.json', 'deberta_v3_large/meta.json']:
                touch(name, '{}')
            cfg = {'keep': 24, 'layers': [2, 4], 'max_len': 384, 'eos_id': 1}
            for family in ['lora_qwen3_14b', 'lora_qwen25_14b']:
                touch(family + '/meta.json', json.dumps({'cfg': cfg}))
            for family in ['q8v5', 'q14v5']:
                for suffix in ['', '_f0', '_f1', '_f2', '_f3', '_f4']:
                    touch(family + suffix + '/heads.pt')
            nb = release.build()
            codes = [release.source(c) for c in nb['cells'] if c['cell_type'] == 'code']
            init = codes[1].replace('/kaggle/input', inp.as_posix()).replace('/kaggle/working', (root / 'working').as_posix())
            env = {}
            native_glob = glob.glob
            # Kaggle uses POSIX paths; normalize the artificial Windows glob results.
            with contextlib.redirect_stdout(io.StringIO()), mock.patch('glob.glob', side_effect=lambda *a, **kw: [p.replace('\\', '/') for p in native_glob(*a, **kw)]):
                exec(compile(init, '<artificial-init>', 'exec'), env)
            self.assertEqual(len(env['AQ8V5']), 6)
            self.assertEqual(len(env['AQ14V5']), 6)
            self.assertTrue(Path(env['DEB']).is_dir())
            self.assertNotIn('BL', env)
            self.assertNotIn('AL', env)
            tree = ast.parse(codes[6])
            assignments = [n for n in tree.body[0].body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'W_N14', 'L', 'MODELS'} for t in n.targets)]
            exec(compile(ast.Module(body=assignments, type_ignores=[]), '<model-config>', 'exec'), env)
            self.assertEqual([m['name'] for m in env['MODELS']], ['q14v5', 'q8v5', 'n14b', 'q25_14', 'deb'])

    def test_prediction_scripts_and_inputs_unchanged(self):
        before = json.loads(release.SOURCE.read_bytes())
        after = release.build()
        a = [release.source(c) for c in before['cells'] if c['cell_type'] == 'code']
        b = [release.source(c) for c in after['cells'] if c['cell_type'] == 'code']
        for i in [0, 2, 3, 4, 5]:
            self.assertEqual(a[i], b[i])
        self.assertEqual(before['metadata']['kaggle'], after['metadata']['kaggle'])
        self.assertIn('_safe_logits = _raw + 0.5 * _gate[:, None] * _evidence', b[3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
