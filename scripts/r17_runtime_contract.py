"""R17 asset identity and execution checks; stdlib only, no model execution."""

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time


def remaining_seconds(start, ceiling=8.3 * 3600):
    remaining = ceiling - (time.monotonic() - start)
    if remaining <= 0:
        raise TimeoutError('R17 complete-execution budget exhausted')
    return remaining


def verify_llama_assets(base, adapters, contract, start=None):
    """Verify base bytes and pinned-dataset adapter identity/presence.

    Kaggle publishes adapter file sizes but no expected payload digests. Their
    observed hashes are recorded for audit, not claimed as independently matched.
    """
    base = Path(base)
    if not base.is_dir():
        raise FileNotFoundError('R17 requires the approved, pinned Llama 8b/1 input: ' + str(base))
    verified = {}
    for name, expected in contract['files'].items():
        if start is not None:
            remaining_seconds(start)
        path = base / name
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError('R17 missing asset or byte-size mismatch: ' + name)
        algorithm = expected['algorithm']
        if algorithm == 'sha256':
            digest = hashlib.sha256()
        elif algorithm == 'git-blob-sha1':
            digest = hashlib.sha1(b'blob ' + str(expected['size']).encode() + b'\0')
        else:
            raise ValueError('R17 unsupported hash algorithm: ' + algorithm)
        with path.open('rb') as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                if start is not None:
                    remaining_seconds(start)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected['digest']:
            raise ValueError('R17 asset content hash mismatch: ' + name)
        verified[name] = {'size': expected['size'], 'algorithm': algorithm, 'digest': actual}
    cfg = json.loads((base / 'config.json').read_bytes())
    expected_config = {'model_type': 'llama', 'hidden_size': 4096,
                       'num_hidden_layers': 32, 'vocab_size': 128256}
    if any(cfg.get(key) != value for key, value in expected_config.items()):
        raise ValueError('R17 Llama base configuration mismatch')
    paths = [Path(p) for p in adapters]
    if ([p.name for p in paths] != contract['adapters']
            or len({p.resolve() for p in paths}) != 6):
        raise ValueError('R17 requires five distinct fold adapters plus the full adapter')
    adapter_info = []
    for path in paths:
        if contract['adapter_dataset_directory'] not in path.parts:
            raise ValueError('R17 adapter is outside the pinned dataset mount')
        file_hashes = {}
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'heads.pt'):
            file = path / name
            expected_size = contract['adapter_file_sizes'][path.name][name]
            if not file.is_file() or file.stat().st_size != expected_size or expected_size <= 0:
                raise ValueError('R17 missing adapter file: ' + str(path / name))
            digest = hashlib.sha256()
            with file.open('rb') as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    if start is not None:
                        remaining_seconds(start)
                    digest.update(chunk)
            file_hashes[name] = {'size': expected_size, 'observed_sha256': digest.hexdigest(),
                                 'independent_expected_digest_available': False}
        acfg = json.loads((path / 'adapter_config.json').read_bytes())
        if any(not math.isfinite(float(acfg[key])) or float(acfg[key]) <= 0
               for key in ('r', 'lora_alpha')):
            raise ValueError('R17 invalid LoRA configuration: ' + path.name)
        if acfg.get('base_model_name_or_path') not in contract['adapter_allowed_base_refs']:
            raise ValueError('R17 unverified adapter base identity: ' + path.name)
        adapter_info.append({'name': path.name,
                             'declared_base': acfg.get('base_model_name_or_path'),
                             'dataset_ref': contract['adapter_dataset_ref'],
                             'dataset_version': contract['adapter_dataset_version'],
                             'files': file_hashes})
    return {'base': str(base), 'kaggle_ref': contract['kaggle_ref'],
            'verified_files': verified, 'adapters': adapter_info,
            'weight_forward_executed': False}


def run_logged(command, log_path, timeout):
    """Preserve all child output and enforce the caller's remaining wall time."""
    if timeout <= 0:
        raise TimeoutError('R17 subprocess has no remaining budget')
    path = Path(log_path)
    print('R17 starting', command[1], '; remaining seconds', round(timeout), flush=True)
    try:
        with path.open('w', encoding='utf-8') as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT,
                           timeout=timeout, text=True)
    finally:
        if path.exists():
            print(path.read_text(encoding='utf-8', errors='replace'), flush=True)
    return path.read_text(encoding='utf-8')


def validate_comp_log(text):
    if 'COMP STACK DONE ' not in text:
        raise ValueError('R17 complementary stack did not complete')
    skipped = re.findall(r'\bSKIP ([^: ]+): checkpoint not attached', text)
    if set(skipped) - {'lora24_public'}:
        raise ValueError('R17 unexpected missing COMP checkpoint: ' + ', '.join(skipped))
    return {'completed': True, 'missing_checkpoint_members': sorted(set(skipped)),
            'optional_head_skips': re.findall(r'OPTIONAL HEAD SKIPPED ([^: ]+):', text)}


def validate_main_log(text):
    for error in ('COMP SKIPPED:', 'ATLAS SKIPPED:', 'atlas invalid scores'):
        if error in text:
            raise ValueError('R17 runtime contribution failure: ' + error)
    members = ['q14v5', 'q8v5', 'll8v5', 'n14b', 'q25_14', 'deb']
    for name in members:
        if not re.search(r'\b' + re.escape(name) + r' done\b', text):
            raise ValueError('R17 main member did not complete: ' + name)
    adapters = {}
    for family in ('q14v5', 'q8v5', 'll8v5'):
        actual = set(re.findall(r'\bl24 ' + family + r': adapter (\S+) done\b', text))
        expected = {family} | {family + '_f' + str(i) for i in range(5)}
        if actual != expected:
            raise ValueError('R17 incomplete v5 adapter family: ' + family)
        adapters[family] = sorted(actual)
    if 'comp stack loaded (' not in text:
        raise ValueError('R17 COMP probabilities were not loaded')
    mixes = re.findall(r'\batlas .+? mixed (\d+)/(\d+) rows at (\S+)', text)
    if len(mixes) != 1:
        raise ValueError('R17 requires one explicit ATLAS mixing receipt')
    scored, eligible, weight = int(mixes[0][0]), int(mixes[0][1]), float(mixes[0][2])
    if not (0 <= scored <= eligible) or (eligible > 0 and scored == 0) or weight != 0.1:
        raise ValueError('R17 invalid or empty ATLAS contribution')
    return {'main_members': members, 'v5_adapters': adapters, 'comp_loaded': True,
            'atlas': {'scored_rows': scored, 'eligible_rows': eligible, 'weight': weight}}
