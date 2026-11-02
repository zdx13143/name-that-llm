import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scripts import r20_cpu_cv as r20


class UnionTest(unittest.TestCase):
    def test_order_normalization_and_case(self):
        opened = [('Ａ  text', 'a'), ('Case', 'b')]
        baseline = [('A text', 'b'), ('case', 'a'), ('new', 'b'), ('new', 'a')]
        rows, origins = r20.union_pool(opened, baseline)
        self.assertEqual(rows, [opened[0], opened[1], baseline[1], baseline[2]])
        self.assertEqual(origins, ['open', 'open', 'baseline', 'baseline'])

    def test_label_set_preserved_and_final_sources(self):
        rows, origins = r20.union_pool([('open a', 'a'), ('open b', 'b')],
                                       [('baseline a', 'a'), ('baseline b', 'b')])
        self.assertEqual({row[1] for row in rows}, {'a', 'b'})
        counts = r20.final_counts([('official a', 'a'), ('official b', 'b')],
                                  [('extra a', 'a')], rows, ['unrelated'], ['a', 'b'], origins)
        self.assertEqual(counts, {'official': 2, 'extra': 1, 'open': 2, 'baseline': 2})

    def test_low_memory_refuses_claim(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r20, 'OUT', Path(tmp) / 'cv'), \
             patch.object(r20.psutil, 'virtual_memory', return_value=SimpleNamespace(available=r20.MIN_AVAILABLE - 1)), \
             patch.object(r20, 'validate_data') as validate:
            with self.assertRaisesRegex(RuntimeError, 'no claim'):
                r20.supervise()
            self.assertFalse((r20.OUT / 'claim.json').exists())
            validate.assert_not_called()

    def test_worker_synthetic_oof_and_source_counts(self):
        labels = ['a', 'b']
        data = {'labels': labels, 'texts': [f'official {i}' for i in range(10)],
                'truth': [i % 2 for i in range(10)], 'fold_id': [i // 2 for i in range(10)],
                'groups': [i // 2 for i in range(10)]}
        sources = {'baseline': ([('base a', 'a'), ('base b', 'b')], 0),
                   'open': ([('open a', 'a'), ('base b', 'b')], 0),
                   'extra': ([('extra a', 'a')], 0)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(r20, 'OUT', Path(tmp)), \
             patch.object(r20, 'validate_data', return_value=data), patch.object(r20, 'R16_LABELS', labels), \
             patch.object(r20, 'EXPECTED_UNION', 3), \
             patch.object(r20, 'load_source', side_effect=lambda name: sources[name]), \
             patch.object(r20, 'filter_source', side_effect=lambda rows, validation: (rows, 0)), \
             patch.object(r20, 'fit_predict', side_effect=lambda official, extra, pool, validation, labels: np.zeros(len(validation), dtype=int)), \
             patch.object(r20, 'metrics', return_value={'synthetic_only': True}):
            r20.worker()
            result = json.loads((Path(tmp) / 'result.json').read_text())
            self.assertEqual(result['union_input_count'], 3)
            self.assertEqual(result['oof_predictions'], {'baseline': [0] * 10, 'union': [0] * 10})
            self.assertEqual(result['final_training_rows'][0]['union'],
                             {'official': 8, 'extra': 1, 'open': 2, 'baseline': 1})
            self.assertEqual(result['exclusions'][0]['union_by_source'], {'open': 0, 'baseline': 0})

    def test_rss_cap_kills_tree_and_retains_failed_claim(self):
        supervisor, process, child = MagicMock(), MagicMock(), MagicMock()
        supervisor.cpu_times.return_value = (0, 0)
        supervisor.memory_info.return_value = SimpleNamespace(rss=1)
        process.children.return_value = [child]
        process.is_running.return_value = child.is_running.return_value = True
        process.memory_info.return_value = SimpleNamespace(rss=r20.MAX_RSS)
        child.memory_info.return_value = SimpleNamespace(rss=1)
        process.cpu_times.return_value = child.cpu_times.return_value = (0, 0)
        proc = MagicMock(pid=1234, returncode=-9)
        proc.poll.return_value = None
        proc.wait.return_value = -9
        with tempfile.TemporaryDirectory() as tmp, patch.object(r20, 'OUT', Path(tmp)), \
             patch.object(r20, 'validate_data'), \
             patch.object(r20.psutil, 'virtual_memory', return_value=SimpleNamespace(available=r20.MIN_AVAILABLE)), \
             patch.object(r20.psutil, 'Process', side_effect=[supervisor, process]), \
             patch.object(r20.psutil, 'wait_procs'), \
             patch.object(r20.subprocess, 'Popen', return_value=proc) as spawn:
            with self.assertRaises(RuntimeError):
                r20.supervise()
            child.kill.assert_called_once()
            process.kill.assert_called_once()
            receipt = json.loads((r20.OUT / 'receipt.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertIn('26 GiB', receipt['reason'])
            self.assertTrue((r20.OUT / 'claim.json').exists())
            with self.assertRaises(FileExistsError):
                with patch.object(r20.psutil, 'Process', return_value=supervisor):
                    r20.supervise()
            self.assertEqual(spawn.call_count, 1)


if __name__ == '__main__':
    unittest.main()
