"""Observe a checked subprocess without loading data or changing its recipe."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import uuid


MAIN_MEMBERS = ('q14v5', 'q8v5', 'll8v5', 'n14b', 'q25_14', 'deb')


def milestone(line):
    """Keep only known structural progress; never persist arbitrary log text."""
    text = re.sub(r'^\[\s*\d+s\]\s*', '', line.strip())
    adapter = re.fullmatch(
        r'l24 (q14v5|q8v5|ll8v5|n14b|q25_14): adapter ([A-Za-z0-9_]{1,64}) '
        r'done \(bf16 redo batches (\d+)\)', text)
    if adapter:
        return {'kind': 'adapter_complete', 'member': adapter[1],
                'adapter': adapter[2], 'bf16_redo_batches': int(adapter[3])}
    for name in MAIN_MEMBERS:
        if text == 'l24 ' + name:
            return {'kind': 'model_started', 'member': name}
        if text == name + ' done':
            return {'kind': 'model_complete', 'member': name}
    comp = re.fullmatch(r'\+\s+([A-Za-z0-9_]{1,64})\s+weight\s+[\d.]+ on (\d+) rows\s+\((\d+)s\)', text)
    if comp:
        return {'kind': 'comp_member_complete', 'member': comp[1], 'rows': int(comp[2])}
    atlas = re.match(r'^atlas on (\d+)/(\d+) rows\b', text)
    if atlas:
        return {'kind': 'atlas_started', 'selected_rows': int(atlas[1]), 'total_rows': int(atlas[2])}
    if re.fullmatch(r'COMP STACK DONE [\d.]+s', text):
        return {'kind': 'comp_complete'}
    if text in ('tfidf done', 'svc4 done'):
        return {'kind': 'cpu_component_complete', 'member': text.split()[0]}
    return None


class RuntimeTrace:
    """Stage receipts and heartbeats, not a timeout controller or score receipt.

    Each session gets a new directory. No GPU synchronization, model import,
    retry, parameter change or process timeout is introduced. Child stdout and
    stderr are forwarded; structured events contain only recognized counters.
    """

    def __init__(self, root, *, mode, source_sha256, source_version=363188587,
                 heartbeat_seconds=60, stream=None):
        if mode not in ('competition_rerun', 'placeholder'):
            raise ValueError('Unknown execution mode')
        if not re.fullmatch(r'[a-f0-9]{64}', source_sha256):
            raise ValueError('Expected source SHA256')
        if heartbeat_seconds <= 0:
            raise ValueError('Heartbeat interval must be positive')
        self.run_id = uuid.uuid4().hex
        self.directory = Path(root) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.heartbeat_seconds = heartbeat_seconds
        self.stream = sys.stdout if stream is None else stream
        self.state = {'schema': 1, 'run_id': self.run_id, 'mode': mode,
                      'source_notebook_sha256': source_sha256, 'source_version': source_version,
                      'status': 'running', 'stage': None, 'latest_milestone': None,
                      'completed_stages': [], 'submission_identity_validated': False,
                      'public_score': None, 'external_scoring_verified': False}
        self.emit('run_started')

    def emit(self, event, **fields):
        stamp = datetime.now(timezone.utc).isoformat()
        self.state.update(observed_at_utc=stamp, elapsed_seconds=round(time.monotonic() - self.started, 3))
        record = dict(self.state, event=event, **fields)
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
        temporary = self.directory / 'status.tmp'
        temporary.write_text(line + '\n', encoding='utf-8')
        os.replace(temporary, self.directory / 'status.json')
        print('NTL_RUNTIME ' + line, file=self.stream, flush=True)

    @contextmanager
    def stage(self, name):
        if self.state['status'] != 'running' or self.state['stage'] is not None:
            raise RuntimeError('Cannot start stage in this trace state')
        if self.state['mode'] != 'competition_rerun':
            raise RuntimeError('Placeholder mode must not execute full-pipeline stages')
        if name not in ('comp_stack', 'comp_output_validation', 'main_inference', 'submission_validation'):
            raise ValueError('Unknown stage')
        self.state.update(stage=name, latest_milestone=None)
        started = time.monotonic()
        self.emit('stage_started')
        try:
            yield
        except BaseException as exc:
            self.state['status'] = 'failed'
            self.emit('stage_failed', error_type=type(exc).__name__,
                      stage_seconds=round(time.monotonic() - started, 3))
            raise
        else:
            self.state['completed_stages'].append(name)
            self.emit('stage_completed', stage_seconds=round(time.monotonic() - started, 3))
            self.state['stage'] = None

    def budget(self, *, comp_seconds, atlas_budget_seconds):
        self.emit('budget_observed', comp_seconds=round(comp_seconds, 3),
                  atlas_budget_seconds=int(atlas_budget_seconds), target_seconds=29880,
                  enforced_global_timeout=False)

    def validate_submission(self, *, ids, models, expected_ids, labels):
        """Check the exact ID order emitted by this recipe and its fixed label set."""
        ids, models, expected_ids, labels = map(list, (ids, models, expected_ids, labels))
        if self.state['stage'] != 'submission_validation':
            raise RuntimeError('Submission validation requires its own recorded stage')
        if len(labels) != 18 or len(set(labels)) != 18:
            raise ValueError('Expected 18 distinct frozen labels')
        if not ids or ids != expected_ids or len(set(ids)) != len(ids):
            raise ValueError('Submission ID identity/order mismatch')
        if len(models) != len(ids) or any(model not in labels for model in models):
            raise ValueError('Submission label outside frozen label set')
        self.state['submission_identity_validated'] = True
        self.emit('submission_identity_checked', rows=len(ids), class_count=len(labels))

    def run(self, command, *, stage):
        """Equivalent checked, single-attempt child execution with a log tee."""
        with self.stage(stage):
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace', env=env) as child:
                messages = queue.Queue(maxsize=256)
                stop = threading.Event()

                def enqueue(item):
                    while not stop.is_set():
                        try:
                            messages.put(item, timeout=0.1)
                            return
                        except queue.Full:
                            pass

                def read_output():
                    try:
                        # Bound each item even if a library writes a very long line.
                        for line in iter(lambda: child.stdout.readline(16384), ''):
                            enqueue(line)
                            if stop.is_set():
                                break
                    except Exception as exc:
                        enqueue(exc)
                    finally:
                        enqueue(None)

                reader = threading.Thread(target=read_output, daemon=True)
                reader.start()
                next_heartbeat = time.monotonic() + self.heartbeat_seconds
                try:
                    while True:
                        try:
                            item = messages.get(timeout=max(0.001, next_heartbeat - time.monotonic()))
                        except queue.Empty:
                            item = ''
                        if item is None:
                            # A child can close stdout before it exits. Keep observing
                            # the process instead of treating EOF as completion.
                            while child.poll() is None:
                                try:
                                    child.wait(timeout=self.heartbeat_seconds)
                                except subprocess.TimeoutExpired:
                                    self.emit('heartbeat')
                            break
                        if isinstance(item, Exception):
                            raise item
                        if item:
                            self.stream.write(item)
                            self.stream.flush()
                            progress = milestone(item)
                            if progress:
                                self.state['latest_milestone'] = progress
                                self.emit('milestone')
                        if time.monotonic() >= next_heartbeat:
                            self.emit('heartbeat')
                            next_heartbeat = time.monotonic() + self.heartbeat_seconds
                    code = child.wait()
                    self.emit('process_exit', returncode=code)
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    return subprocess.CompletedProcess(command, code)
                except BaseException:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    raise
                finally:
                    stop.set()
                    reader.join(timeout=5)

    def finish(self, outcome):
        allowed = {'competition_rerun': 'pipeline_validated', 'placeholder': 'placeholder_only'}
        if outcome != allowed[self.state['mode']]:
            raise ValueError('Outcome does not match execution mode')
        if self.state['status'] != 'running' or self.state['stage'] is not None:
            raise RuntimeError('Cannot finish an active or failed stage')
        if outcome == 'pipeline_validated' and self.state['completed_stages'] != [
                'comp_stack', 'comp_output_validation', 'main_inference', 'submission_validation']:
            raise RuntimeError('All pipeline stages and output checks must complete in order')
        if outcome == 'pipeline_validated' and not self.state['submission_identity_validated']:
            raise RuntimeError('Submission identity has not been validated')
        self.state['status'] = outcome
        self.emit('run_finished')


def read_summary(directory):
    state = json.loads((Path(directory) / 'status.json').read_text(encoding='utf-8'))
    if state.get('schema') != 1 or state.get('external_scoring_verified') is not False:
        raise ValueError('Not a local runtime observer receipt')
    state['interpretation'] = {
        'running': 'last_observed_running; this file cannot prove current process liveness',
        'failed': 'local stage failed; not a Kaggle terminal status',
        'pipeline_validated': 'local pipeline and output checks completed; Kaggle score unobserved',
        'placeholder_only': 'normal-save placeholder only; full inference not exercised',
    }[state['status']]
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read one runtime trace; no inference is executed')
    parser.add_argument('trace_directory', type=Path)
    print(json.dumps(read_summary(parser.parse_args().trace_directory), indent=2))
