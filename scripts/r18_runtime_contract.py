"""R18 subprocess limits, lifecycle measurements and fail-closed receipts."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import weakref


def remaining_seconds(start, ceiling=8 * 3600):
    remaining = ceiling - (time.monotonic() - start)
    if remaining <= 0:
        raise TimeoutError('R18 complete-execution budget exhausted')
    return remaining


def run_logged(command, log_path, timeout):
    if timeout <= 0:
        raise TimeoutError('R18 subprocess has no remaining budget')
    path = Path(log_path)
    print('R18 starting', command[1], '; remaining seconds', round(timeout), flush=True)
    try:
        with path.open('w', encoding='utf-8') as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    finally:
        if path.exists():
            print(path.read_text(encoding='utf-8', errors='replace'), flush=True)
    return path.read_text(encoding='utf-8')


class Lifecycle:
    """Weak references do not keep GPU tensors alive. Sampling only in Save."""
    def __init__(self, family):
        import torch
        self.torch = torch
        self.family = family
        self.old = None
        self.fingerprint = None
        self.trace = os.getenv('R18_TRACE_PATH')
        self.floor = self.allocated()

    def allocated(self):
        return [self.torch.cuda.memory_allocated(i) for i in range(self.torch.cuda.device_count())]

    def emit(self, event, **fields):
        entry = {'event': event, 'family': self.family, 'adapter': self.adapter, **fields}
        print('R18_LIFECYCLE ' + json.dumps(entry, sort_keys=True), flush=True)
        if self.trace:
            with open(self.trace, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, sort_keys=True) + '\n')

    def before_load(self, adapter):
        if self.old is not None and self.old() is not None:
            raise RuntimeError('R18 prior base is still alive')
        self.adapter = Path(adapter).name
        self.started = time.monotonic()
        self.emit('before_load', allocated=self.allocated())

    def loaded(self, model):
        self.old = weakref.ref(model)
        devices = sorted({str(p.device) for p in model.parameters()})
        if os.getenv('R18_REQUIRE_CUDA') == '1' and any(p.device.type != 'cuda' for p in model.parameters()):
            raise RuntimeError('R18 preflight requires GPU-resident parameters: ' + str(devices))
        sample = None
        if self.trace:
            digest = hashlib.sha256()
            for name, param in model.named_parameters():
                digest.update(name.encode())
                flat = param.detach().reshape(-1)
                count = min(64, flat.numel())
                indexes = self.torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
                digest.update(flat[indexes].float().cpu().numpy().tobytes())
            sample = digest.hexdigest()
            if self.fingerprint is not None and sample != self.fingerprint:
                raise RuntimeError('R18 fresh base sample fingerprint changed')
            self.fingerprint = sample
        self.emit('loaded', seconds=time.monotonic() - self.started, sample_sha256=sample,
                  parameter_devices=devices, device_map=getattr(model, 'hf_device_map', None), allocated=self.allocated())

    def released(self):
        if self.old() is not None:
            raise RuntimeError('R18 base object survived release')
        allocated = self.allocated()
        if any(now > before + 16 * 1024 * 1024 for now, before in zip(allocated, self.floor)):
            raise RuntimeError('R18 live GPU allocation survived adapter release: ' + str(allocated))
        self.emit('released', seconds=time.monotonic() - self.started, allocated=allocated)


def validate_comp_log(text):
    if 'COMP STACK DONE ' not in text:
        raise ValueError('R18 complementary stack did not complete')
    skipped = re.findall(r'\bSKIP ([^: ]+): checkpoint not attached', text)
    if set(skipped) - {'lora24_public'}:
        raise ValueError('R18 unexpected missing COMP checkpoint: ' + ', '.join(skipped))
    return {'completed': True, 'missing_checkpoint_members': sorted(set(skipped)),
        'optional_head_skips': re.findall(r'OPTIONAL HEAD SKIPPED ([^: ]+):', text)}


def validate_main_log(text):
    for error in ('COMP SKIPPED:', 'ATLAS SKIPPED:', 'atlas invalid scores'):
        if error in text:
            raise ValueError('R18 runtime contribution failure: ' + error)
    members = ['q14v5', 'q8v5', 'n14b', 'q25_14', 'deb']
    for name in members:
        if not re.search(r'\b' + name + r' done\b', text):
            raise ValueError('R18 main member did not complete: ' + name)
    adapters = {}
    for family in ('q14v5', 'q8v5'):
        actual = re.findall(r'\bl24 ' + family + r': adapter (\S+) done\b', text)
        expected = {family} | {family + '_f' + str(i) for i in range(5)}
        if len(actual) != 6 or set(actual) != expected:
            raise ValueError('R18 incomplete v5 adapter family: ' + family)
        adapters[family] = actual
    if 'comp stack loaded (' not in text:
        raise ValueError('R18 COMP probabilities were not loaded')
    mixes = re.findall(r'\batlas .+? mixed (\d+)/(\d+) rows at (\S+)', text)
    if len(mixes) != 1:
        raise ValueError('R18 requires one explicit ATLAS mixing receipt')
    scored, eligible, weight = int(mixes[0][0]), int(mixes[0][1]), float(mixes[0][2])
    if not (0 <= scored <= eligible) or (eligible > 0 and scored == 0) or weight != 0.1:
        raise ValueError('R18 invalid or empty ATLAS contribution')
    return {'main_members': members, 'v5_adapters': adapters, 'comp_loaded': True,
        'atlas': {'scored_rows': scored, 'eligible_rows': eligible, 'weight': weight}}
