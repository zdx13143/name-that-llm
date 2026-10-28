"""Synthetic subprocess and actual notebook-wrapper checks; no model/data loads."""

import ast
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments'))
import build_r13_observed as builder
from r13_runtime import RuntimeTrace, milestone, read_summary


class RemoveObserver(ast.NodeTransformer):
    """Independent AST equivalence check for the two notebook launch cells."""

    def visit_With(self, node):
        self.generic_visit(node)
        if len(node.items) == 1 and ast.unparse(node.items[0].context_expr).startswith('_r13_trace.stage('):
            return node.body
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            call = node.value
            if isinstance(call.func.value, ast.Name) and call.func.value.id == '_r13_trace':
                if call.func.attr in ('budget', 'finish', 'validate_submission'):
                    return None
                if call.func.attr == 'run':
                    return ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='subprocess', ctx=ast.Load()), attr='run', ctx=ast.Load()),
                        args=call.args, keywords=[ast.keyword(arg='check', value=ast.Constant(value=True))]))
        return self.generic_visit(node)


@contextmanager
def in_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb, cls.manifest = builder.build()
        cls.original = json.loads(builder.SOURCE.read_bytes())
        cls.raw = [builder.source(c) for c in cls.original['cells'] if c['cell_type'] == 'code']
        cls.observed = [builder.source(c) for c in cls.nb['cells'] if c['cell_type'] == 'code']

    def setUp(self):
        target = ROOT / 'outputs/r13_test_tmp'
        target.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.stream = io.StringIO()

    def trace(self, mode='competition_rerun', interval=0.05):
        return RuntimeTrace(self.directory / 'traces', mode=mode,
                            source_sha256=builder.SOURCE_SHA256,
                            heartbeat_seconds=interval, stream=self.stream)

    def events(self, trace):
        return [json.loads(line) for line in (trace.directory / 'events.jsonl').read_text().splitlines()]

    def test_prediction_scripts_inputs_and_launch_logic_are_preserved(self):
        # Only the helper and bootstrap are added after the existing rerun flag.
        aligned = self.observed[:2] + self.observed[4:]
        self.assertEqual(aligned[:6], self.raw[:6])
        for original, changed in zip(self.raw[6:], aligned[6:]):
            normalized = RemoveObserver().visit(ast.parse(changed))
            self.assertEqual(ast.dump(normalized), ast.dump(ast.parse(original)))
        self.assertEqual(self.nb['metadata']['kaggle'], self.original['metadata']['kaggle'])
        self.assertEqual(len(self.manifest['pinned_inputs']), 31)
        self.assertEqual(set(self.manifest['prediction_scripts_unchanged']), {'gauss.py', 'infer.py', 'comp_stack.py'})
        self.assertIn(builder.OBSERVER.read_text(encoding='utf-8'), self.observed[2])

    def test_deterministic_packaging_has_no_execution_outputs(self):
        content = json.dumps(builder.build()[0], ensure_ascii=False, indent=1) + '\n'
        self.assertEqual(content.encode(), builder.OUTPUT.read_bytes())
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        self.assertEqual(hashlib.sha256(builder.SOURCE.read_bytes()).hexdigest(), builder.SOURCE_SHA256)

    def synthetic_pipeline(self, *, observed, mode=True, invalid_comp=False, invalid_output=None):
        work = self.directory / ('observed' if observed else 'baseline')
        work.mkdir()
        pd.DataFrame({'id': ['synthetic-a', 'synthetic-b', 'synthetic-c']}).to_csv(work / 'test.csv', index=False)
        labels = builder.prediction_labels(self.raw)
        pd.DataFrame({'model': [labels[0]] * 3}).to_csv(work / 'train.csv', index=False)
        comp = (
            'import json, numpy as np, pandas as pd\n'
            "ids = pd.read_csv('test.csv').id.tolist()\n" +
            ("ids = ids[::-1]\n" if invalid_comp else '') +
            "np.save('comp_logit.npy', np.zeros((3, 18)))\n"
            "json.dump({'ids': ids, 'labels': ['synthetic-' + str(i) for i in range(18)]}, open('comp_meta.json', 'w'))\n"
            "print('COMP STACK DONE 1s', flush=True)\n"
        )
        infer = (
            'import argparse, pandas as pd\n'
            "p = argparse.ArgumentParser(); p.add_argument('--out'); a, _ = p.parse_known_args()\n"
            "ids = pd.read_csv('test.csv').id.tolist()\n" +
            ("ids = ids[::-1]\n" if invalid_output == 'ids' else '') +
            f"pd.DataFrame({{'id': ids, 'model': {('invalid-synthetic-label' if invalid_output == 'labels' else labels[0])!r}}}).to_parquet(a.out, index=False)\n"
            "print('[2s] l24 q14v5: adapter q14v5_f0 done (bf16 redo batches 0)', flush=True)\n"
        )
        (work / 'comp_stack.py').write_text(comp)
        (work / 'infer.py').write_text(infer)
        trace = self.trace(mode='competition_rerun' if mode else 'placeholder') if observed else None
        env = {name: 'synthetic-path' for name in ('B14', 'B8', 'BL', 'B25_14', 'N14B', 'Q25_14', 'DEB', 'EXTRA', 'POOL', 'ATLAS')}
        env.update({name: ['synthetic-adapter'] for name in ('AQ14V5', 'AQ8V5', 'ALL8V5')})
        env.update({name: dict(keep=1, layers=[1], max_len=4, eos_id=0) for name in ('N14BC', 'Q25_14C')})
        env.update(RERUN=mode, DATA=str(work), os=os, json=json, _r13_trace=trace, _R13_EXPECTED_LABELS=labels)
        cells = self.observed[-2:] if observed else self.raw[-2:]
        with in_directory(work), redirect_stdout(self.stream), patch.dict(os.environ):
            # The real launch/validation code is executed with fake model children
            # and synthetic CSVs. No production prediction script is executed.
            for cell in cells:
                local = cell.replace('/kaggle/working/', work.as_posix() + '/')
                exec(compile(local, '<actual-launch-cell>', 'exec'), env)
        return work / 'submission.parquet', trace

    def test_actual_wrappers_preserve_synthetic_prediction_file_bytes(self):
        baseline, _ = self.synthetic_pipeline(observed=False)
        candidate, trace = self.synthetic_pipeline(observed=True)
        self.assertEqual(baseline.read_bytes(), candidate.read_bytes())
        summary = read_summary(trace.directory)
        self.assertEqual(summary['status'], 'pipeline_validated')
        self.assertFalse(summary['external_scoring_verified'])
        self.assertIsNone(summary['public_score'])
        self.assertEqual(summary['completed_stages'], [
            'comp_stack', 'comp_output_validation', 'main_inference', 'submission_validation'])
        kinds = [item['latest_milestone']['kind'] for item in self.events(trace) if item['event'] == 'milestone']
        self.assertEqual(kinds, ['comp_complete', 'adapter_complete'])

    def test_successful_child_with_wrong_ids_fails_validation(self):
        with self.assertRaisesRegex(AssertionError, 'COMP ID mismatch'):
            self.synthetic_pipeline(observed=True, invalid_comp=True)
        receipt = next((self.directory / 'traces').glob('*/status.json'))
        summary = json.loads(receipt.read_text())
        self.assertEqual((summary['status'], summary['stage']), ('failed', 'comp_output_validation'))
        self.assertEqual(summary['error_type'], 'AssertionError')

    def test_normal_save_remains_placeholder_without_child_execution(self):
        with patch('r13_runtime.subprocess.Popen', side_effect=AssertionError('unexpected full inference')):
            artifact, trace = self.synthetic_pipeline(observed=True, mode=False)
        self.assertEqual(len(pd.read_parquet(artifact)), 3)
        self.assertEqual(read_summary(trace.directory)['status'], 'placeholder_only')

    def test_valid_shape_but_reordered_output_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ID identity/order mismatch'):
            self.synthetic_pipeline(observed=True, invalid_output='ids')
        receipt = next((self.directory / 'traces').glob('*/status.json'))
        state = json.loads(receipt.read_text())
        self.assertEqual((state['status'], state['stage']), ('failed', 'submission_validation'))

    def test_nonempty_but_invalid_output_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside frozen label set'):
            self.synthetic_pipeline(observed=True, invalid_output='labels')
        receipt = next((self.directory / 'traces').glob('*/status.json'))
        state = json.loads(receipt.read_text())
        self.assertEqual((state['status'], state['stage']), ('failed', 'submission_validation'))

    def test_silent_child_has_heartbeats_and_no_invented_progress(self):
        trace = self.trace()
        trace.run([sys.executable, '-c', 'import time; time.sleep(0.25)'], stage='comp_stack')
        events = self.events(trace)
        self.assertTrue(any(item['event'] == 'heartbeat' for item in events))
        self.assertTrue(all(item['latest_milestone'] is None for item in events))
        self.assertIn('cannot prove current process liveness', read_summary(trace.directory)['interpretation'])

    def test_nonzero_exit_and_large_stderr_are_preserved(self):
        trace = self.trace()
        cmd = [sys.executable, '-c', "import sys; sys.stderr.write('synthetic diagnostic\\n' * 15000); sys.exit(7)"]
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            trace.run(cmd, stage='main_inference')
        self.assertEqual(failure.exception.returncode, 7)
        events = self.events(trace)
        self.assertEqual([e['returncode'] for e in events if e['event'] == 'process_exit'], [7])
        self.assertEqual(read_summary(trace.directory)['status'], 'failed')
        self.assertEqual(self.stream.getvalue().count('synthetic diagnostic'), 15000)
        self.assertNotIn('synthetic diagnostic', (trace.directory / 'events.jsonl').read_text())

    def test_launch_failure_is_not_silent_or_successful(self):
        trace = self.trace()
        with self.assertRaises(OSError):
            trace.run([str(self.directory / 'nonexistent-executable')], stage='comp_stack')
        self.assertEqual(read_summary(trace.directory)['status'], 'failed')
        with self.assertRaises(RuntimeError):
            trace.finish('pipeline_validated')

    def test_stale_runs_cannot_be_confused_or_finished_early(self):
        first, second = self.trace(), self.trace(mode='placeholder')
        self.assertNotEqual(first.directory, second.directory)
        with self.assertRaises(RuntimeError):
            first.finish('pipeline_validated')
        with self.assertRaises(ValueError):
            second.finish('pipeline_validated')
        second.finish('placeholder_only')
        self.assertEqual(read_summary(first.directory)['status'], 'running')
        self.assertEqual(read_summary(second.directory)['status'], 'placeholder_only')

    def test_parser_keeps_progress_counters_only(self):
        self.assertIsNone(milestone('unrecognized passage or credential: never persist this'))
        self.assertEqual(milestone('[12s] atlas on 4/20 rows (top-2 margin < 0.3)'),
                         {'kind': 'atlas_started', 'selected_rows': 4, 'total_rows': 20})
        self.assertEqual(milestone('  + ultrabert            weight 0.095 on 3 rows  (3157s)'),
                         {'kind': 'comp_member_complete', 'member': 'ultrabert', 'rows': 3})


if __name__ == '__main__':
    unittest.main(verbosity=2)
