import base64
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import build_r26_notebook as builder

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5','gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max')


class PackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        builder.build(out=cls.root / 'a')
        builder.build(out=cls.root / 'b')
        cls.nb = json.loads((cls.root / 'a/ntl-r26-v15-assign.ipynb').read_text(encoding='utf-8'))
        cls.sources = [''.join(c.get('source', [])) for c in cls.nb['cells']]
        cls.infer = next(s for s in cls.sources if s.startswith('%%writefile infer.py'))
        cls.selftest = next(s for s in cls.sources if s.startswith('# R26 synthetic self-test'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_structure_solver_and_selftest(self):
        s = self.infer
        self.assertLess(s.index('_safe_logits = _raw +'), s.index('# R26_RULE_BEGIN'))
        self.assertLess(s.index('# R26_RULE_BEGIN'), s.index('_safe.to_parquet(_tmp'))
        self.assertNotIn('_safe.to_parquet(_out', s)
        self.assertNotIn('r23_rule', s)
        raw = base64.b64decode(re.search(r"base64.b64decode\('([^']+)'", self.selftest).group(1))
        self.assertEqual(hashlib.sha256(raw).digest(), hashlib.sha256(builder.SOLVER.read_bytes()).digest())
        self.assertNotIn('if not FAST_COMMIT:', self.selftest)
        self.assertLess(self.sources.index(self.selftest), next(i for i,s in enumerate(self.sources) if 'if not FAST_COMMIT:' in s and 'comp_stack.py' in s))
        code = self.selftest.replace('/kaggle/working/r24_assignment.py', (self.root / 'r24_assignment.py').as_posix()).replace("pathlib.Path('infer.py').read_text()", repr(self.infer.split('\n', 1)[1]))
        exec(compile(code, '<synthetic-selftest>', 'exec'), {})

    def test_rule_parquet_and_failure_closed(self):
        code = builder.RULE.replace("'/kaggle/working/r26_diagnostics.json'", repr(str(self.root / 'diagnostics.json')))
        code = 'if True:\n' + code
        out = self.root / 'submission.parquet'
        groups = np.repeat([1, 2], [12, 11])
        logits = np.zeros((23, 18)); logits[:, 0] = 0.3; logits[:, 1] = 0.1
        logits[:12, 1] = 0.3
        env = {'np': np, 'pd': pd, 'Gte': groups, 'test': pd.DataFrame({'id': np.arange(23)}), 'LABELS': LABELS, '_safe_logits': logits, '_out': out}
        with patch.dict('sys.modules', {'r24_assignment': __import__('scripts.r24_assignment', fromlist=['predict'])}):
            exec(code, env)
            got = pd.read_parquet(out)
            self.assertEqual(got.model.tolist(), [LABELS[i] for i in env['_after']])
            self.assertTrue((env['_after'][:12] != 0).any())
            self.assertTrue((env['_after'][12:] == 0).all())
            self.assertAlmostEqual(env['_scale'], 0.2)
            env['_safe_logits'] = np.zeros((23, 18))
            with self.assertRaises(ValueError): exec(code, env)
            self.assertFalse(out.exists())
            env['_safe_logits'] = logits
            out.write_bytes(b'stale')
            with patch.object(pd, 'read_parquet', side_effect=ValueError('synthetic read failure')):
                with self.assertRaises(ValueError): exec(code, env)
            self.assertFalse(out.exists())
            real_read = pd.read_parquet
            reads = []
            def fail_final(path):
                reads.append(path)
                if len(reads) == 2: raise ValueError('synthetic final read failure')
                return real_read(path)
            with patch.object(pd, 'read_parquet', side_effect=fail_final):
                with self.assertRaises(ValueError): exec(code, env)
            self.assertEqual(len(reads), 2)
            self.assertFalse(out.exists())

    def test_build_deterministic_metadata(self):
        for name in ('ntl-r26-v15-assign.ipynb', 'kernel-metadata.json', 'source.diff'):
            self.assertEqual((self.root / 'a' / name).read_bytes(), (self.root / 'b' / name).read_bytes())
        m = json.loads((self.root / 'a/kernel-metadata.json').read_text())
        self.assertEqual(m['id'], 'zephyrpong/ntl-r26-v15-assign')
        self.assertTrue(m['is_private']); self.assertFalse(m['enable_internet'])
        self.assertIn(builder.r23.NEW_POOL, m['dataset_sources']); self.assertIn(builder.r23.NEW_GEN, m['dataset_sources'])
        self.assertNotIn(builder.r23.OLD_POOL, m['dataset_sources']); self.assertNotIn(builder.r23.OLD_GEN, m['dataset_sources'])

if __name__ == '__main__':
    unittest.main()
