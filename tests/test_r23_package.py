import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts import build_r23_notebook as b

LABELS = [f'label-{i}' for i in range(18)]
LABELS[7], LABELS[8] = 'gemini-3.7-flash', 'gemini-3.8-flash'

class R23Tests(unittest.TestCase):
    def setUp(self):
        ns = {}
        exec(b.RULE.split('\npred = r23_rule', 1)[0], ns)
        self.rule = ns['r23_rule']

    def call(self, scores, labels=LABELS, perpassage=1, post=0):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.rule(scores, labels, scores.argmax(1), perpassage, post)
        return result, json.loads(output.getvalue().split(' ', 1)[1])

    def test_rule_boundary_direction_and_counts(self):
        x = np.full((4,18), -10.)
        x[:, 7] = [0, 0, 0, 0]
        x[:, 8] = [0.999, 1.0, 1.001, 0.5]
        x[3, 1] = 3
        y, d = self.call(x)
        np.testing.assert_array_equal(y, [7, 8, 8, 1])
        self.assertEqual((d['eligible_3_8'], d['switched']), (3, 1))
        self.assertEqual(sum(d['counts_before']), 4)
        self.assertEqual(d['counts_after'][7], d['counts_before'][7]+1)
        self.assertEqual(d['counts_after'][8], d['counts_before'][8]-1)
        self.assertEqual([i for i in range(18) if d['counts_before'][i] != d['counts_after'][i]], [7,8])

    def test_empty_eligible_and_fail_closed(self):
        x = np.zeros((2,18))
        _, d = self.call(x)
        self.assertEqual(d['gap_quantiles']['5%'], None)
        x[0,3] = np.nan
        with self.assertRaises(ValueError): self.call(x)
        x[0,3] = np.inf
        with self.assertRaises(ValueError): self.call(x)
        with self.assertRaises(ValueError): self.call(np.zeros((1,18)), perpassage=0)
        with self.assertRaises(ValueError): self.call(np.zeros((1,18)), post=1)
        labels = LABELS.copy(); labels[7], labels[8] = labels[8], labels[7]
        with self.assertRaises(ValueError): self.call(np.zeros((1,18)), labels=labels)

    def test_anchor_unique(self):
        for text in ('none', 'needle needle'):
            with self.assertRaises(ValueError): b.replace_once({'cells':[{'source':[text]}]}, 'needle', 'x')

    def test_builder_deterministic_and_metadata(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as c:
            b.build(out=a); b.build(out=c)
            for name in ('ntl-r23-v15-pairwise.ipynb', 'kernel-metadata.json', 'source.diff'):
                self.assertEqual((Path(a)/name).read_bytes(), (Path(c)/name).read_bytes())
            meta=json.loads((Path(a)/'kernel-metadata.json').read_text(encoding='utf-8'))
            self.assertIn(b.NEW_POOL, meta['dataset_sources'])
            self.assertNotIn(b.OLD_POOL, meta['dataset_sources'])
            self.assertEqual(meta['id'], 'zephyrpong/ntl-r23-v15-pairwise')
            self.assertTrue(meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'])
            original=json.loads((b.SOURCE_DIR/'ntl-we-back-0-666.ipynb').read_text(encoding='utf-8'))
            built=json.loads((Path(a)/'ntl-r23-v15-pairwise.ipynb').read_text(encoding='utf-8'))
            changed=[i for i,(x,y) in enumerate(zip(original['cells'],built['cells'])) if x != y]
            self.assertEqual(changed, [6,10,16])
            diff=(Path(a)/'source.diff').read_text(encoding='utf-8')
            self.assertEqual(diff.count('\n@@ '), 3)
            for marker in ('R23_MANIFEST', 'r23_rule', '_r23_pool', '_r23_pip'):
                self.assertIn(marker,diff)

if __name__ == '__main__': unittest.main()
