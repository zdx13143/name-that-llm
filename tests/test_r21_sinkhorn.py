import unittest
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scripts.r21_sinkhorn import balance, SOURCE


class SinkhornTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(30)
        self.members = {name: rng.normal(size=(36, 18)) * weight for name, weight in
                        [('tfidf', .25), ('svc4', 1.5), ('distort', .1)]}
        self.y = np.repeat(np.arange(18), 3)

    def test_extracted_notebook_blocks_agree(self):
        self.assertEqual((SOURCE / 'r16_sinkhorn.py.txt').read_text(),
                         (SOURCE / 'areeb_sinkhorn.py.txt').read_text())
        raw, balanced = balance(self.members, self.y, 'r16')
        np.testing.assert_allclose(raw, sum(self.members.values()))
        self.assertEqual(balanced.shape, (36, 18))
        self.assertTrue(np.isfinite(balanced).all())
        self.assertFalse(np.allclose(raw, balanced))
        _, other = balance(self.members, self.y, 'areeb')
        self.assertFalse(np.allclose(balanced, other))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            balance({'tfidf': np.zeros((36, 18))}, self.y, 'r16')
        with self.assertRaises(ValueError):
            balance(self.members, np.array([18]), 'r16')
        with self.assertRaises(ValueError):
            balance(self.members, self.y, 'other')

    def test_parity_against_full_original_notebook_blocks(self):
        paths = {'r16': Path('notebooks/team_e654_r16_typography_half_runnable.ipynb'),
                 'areeb': Path('D:/kaggle/name_that_llm/outputs/codex_runs/r21-plan/sources/areebbb__ntl-we-back-0-666/ntl-we-back-0-666.ipynb')}
        for condition, path in paths.items():
            nb = json.loads(path.read_text(encoding='utf-8'))
            source = next(''.join(cell.get('source', [])) for cell in nb['cells']
                          if 'if a.svc_w > 0:' in ''.join(cell.get('source', [])))
            begin = 'if (a.sinkhorn > 0 and len(test) >= 18) or _bundles or COMP_P is not None:'
            end = '_v38_before_hard = tot.copy()'
            block = source[source.index(begin):source.index(end)]
            strength = .45 if condition == 'r16' else .30
            env = {'np': np, 'LP': self.members, 'tot': sum(self.members.values()),
                   'y': self.y, 'test': range(36), '_bundles': [], '_aw': [], 'COMP_P': None,
                   'a': SimpleNamespace(sinkhorn=strength, sinkhorn_target_mix=.35, sk_div=1)}
            exec(block, env)
            _, actual = balance(self.members, self.y, condition)
            np.testing.assert_allclose(actual, env['tot'], rtol=1e-14, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
