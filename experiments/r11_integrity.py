"""Integrity checks for the pinned teammate 0.651 ensemble; no score transforms."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5',
    'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3',
])
BASE_WEIGHTS = {
    'stylegram': 1.516, 'stylometry_lgb': 0.063, 'ultrabert': 0.095,
    'debv3l': 0.05, 'electl': 0.032, 'desk': 0.132,
    'lora24_seed7': 0.113, 'lora24_claude': 0.177,
    'lora20_q25_instyle': 0.17, 'tokattn_deb': 0.185,
}
GROUP_WEIGHTS = {
    'q3_8b_twin': 1.5 * 0.219, 'q25_7b_twin': 1.5 * 0.141,
    'q3_4b_twin': 1.5 * 0.083, 'glm9b_twin': 1.5 * 0.110,
    'debv3l_twin': 0.25, 'lora24_claude_twin': 0.15,
    'gauss_q3_8b_L22': 1.5 * 0.219, 'gauss_glm9b_L8': 1.5 * 0.110,
    'gauss_q3_4b_L4': 1.5 * 0.083, 'gauss_debv3l_L24': 0.25,
}


def require(condition, message):
    if not condition:
        raise RuntimeError('R11 integrity: ' + message)


def validate_comp(logits, meta, run_id):
    """Validate without modifying arrays or metadata; reject partial ensembles."""
    require(isinstance(run_id, str) and bool(run_id), 'missing run identifier')
    require(meta.get('r11_run_id') == run_id, 'stale or foreign run identifier')
    require(meta.get('r11_schema') == 1, 'unsupported complementary receipt')
    labels, ids = meta.get('labels'), meta.get('ids')
    require(isinstance(labels, list) and all(isinstance(x, str) for x in labels), 'invalid labels')
    require(len(labels) == 18 and len(set(labels)) == 18 and set(labels) == set(LABELS), 'class set differs')
    require(isinstance(ids, list) and len(ids) > 0 and all(isinstance(x, str) for x in ids), 'invalid IDs')
    require(len(set(ids)) == len(ids), 'duplicate IDs')
    require(isinstance(logits, np.ndarray) and logits.shape == (len(ids), 18), 'logit shape differs')
    require(np.issubdtype(logits.dtype, np.number) and np.isrealobj(logits), 'logits are not real numbers')
    require(np.isfinite(logits).all(), 'nonfinite complementary logits')
    seconds = meta.get('seconds')
    require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0, 'invalid elapsed time')
    grouped = meta.get('grouped_rows')
    require(type(grouped) is int and 0 <= grouped <= len(ids) and grouped != 1, 'invalid grouped-row count')
    members = meta.get('members')
    require(isinstance(members, dict), 'missing member manifest')
    expected = dict(BASE_WEIGHTS)
    if grouped:
        expected.update(GROUP_WEIGHTS)
    require(set(members) == set(expected), 'complementary member set differs: ' + str(sorted(set(members) ^ set(expected))))
    for key, weight in expected.items():
        rec = members[key]
        require(isinstance(rec, dict), key + ': invalid member record')
        rows = len(ids) if key in BASE_WEIGHTS else grouped
        require(type(rec.get('rows')) is int and rec['rows'] == rows, key + ': row coverage differs')
        actual = rec.get('weight')
        require(isinstance(actual, (int, float)) and math.isclose(actual, weight, rel_tol=0, abs_tol=1e-12), key + ': weight differs')


def write_comp(path, logits, labels, ids, seconds, members, grouped_rows, run_id):
    path = Path(path)
    meta_path = path.with_name('comp_meta.json')
    meta = {
        'r11_schema': 1, 'r11_run_id': run_id, 'labels': list(labels),
        'ids': list(ids), 'seconds': float(seconds), 'members': members,
        'grouped_rows': int(grouped_rows),
    }
    validate_comp(logits, meta, run_id)
    tmp = path.with_suffix('.npy.tmp')
    with tmp.open('wb') as handle:
        np.save(handle, logits, allow_pickle=False)
    meta['logit_sha256'] = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp_meta = meta_path.with_suffix('.json.tmp')
    tmp_meta.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)
    os.replace(tmp_meta, meta_path)


def load_comp(path, ids, labels, run_id):
    path = Path(path)
    meta = json.loads(path.with_name('comp_meta.json').read_text(encoding='utf-8'))
    require(hashlib.sha256(path.read_bytes()).hexdigest() == meta.get('logit_sha256'), 'complementary file hash differs')
    logits = np.load(path, allow_pickle=False)
    validate_comp(logits, meta, run_id)
    require(meta['ids'] == list(ids), 'complementary row order differs')
    require(len(labels) == 18 and len(set(labels)) == 18 and set(labels) == set(meta['labels']), 'requested class order differs')
    # This is the exact original indexing and softmax, with no new calibration.
    logits = logits[:, [meta['labels'].index(label) for label in labels]]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    require(np.isfinite(probs).all(), 'nonfinite complementary probabilities')
    return probs, meta


def clear_outputs(outputs):
    paths = [Path(p).resolve() for p in outputs]
    require(all(p.parent == Path.cwd().resolve() for p in paths), 'stage output must be in the working directory')
    for path in paths:
        path.unlink(missing_ok=True)
    return paths


def run_checked(command, outputs, timeout, validator=None):
    """Run a stage with a deadline and remove declared outputs on any failure."""
    paths = clear_outputs(outputs)
    try:
        require(timeout > 0, 'stage has no remaining time budget')
        subprocess.run(command, check=True, timeout=timeout)
        require(all(p.is_file() and p.stat().st_size > 0 for p in paths), 'stage did not produce all outputs')
        if validator is not None:
            validator()
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def validate_submission(path, test_path):
    import pandas as pd
    submission = pd.read_parquet(path)
    test = pd.read_csv(test_path, usecols=['id'])
    require(list(submission.columns) == ['id', 'model'], 'submission columns differ')
    require(len(submission) == len(test) and len(test) > 0, 'submission row count differs')
    require(submission['id'].notna().all() and submission['id'].is_unique, 'submission IDs invalid')
    require(submission['id'].astype(str).tolist() == test['id'].astype(str).tolist(), 'submission ID order differs')
    require(submission['model'].isin(LABELS).all(), 'submission labels invalid')
