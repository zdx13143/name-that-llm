"""Prepared fallback, DISABLED pending explicit permission to read the full CSV.

This is not the preisolated-source intake. If later approved, the operation
reads all official bytes and parses protected rows twice in memory. Only the
historical development rows can be written, to quarantine, with truthful
access counters. It never creates a preisolated-source review or admits training.
The proposal file records approval; it cannot itself establish user consent.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('D:/kaggle/name_that_llm/name-that-llm/data/raw/train.csv')
TRAIN_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
PROPOSAL = ROOT / 'configs/r15_raw_export_proposal_20261029.json'
DESTINATION = ROOT / 'outputs/r15_raw_export_quarantine'
CLAIM = ROOT / 'outputs/r15_raw_export_attempt.json'
MAX_SOURCE_BYTES = 4 * 1024**2
MAX_OUTPUT_BYTES = 8 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(blob):
    with io.TextIOWrapper(io.BytesIO(blob), encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, strict=True)
        require(len(reader.fieldnames or []) == 4
                and set(reader.fieldnames) == {'id', 'prompt_id', 'text', 'model'},
                'Unexpected CSV schema')
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()),
                    'Malformed CSV row')
            yield row


def historical_development_indices(metadata, train_sha256, expected_split_sha256):
    """Copy of the historical split algorithm; never persist its all-row manifest."""
    import numpy as np

    groups = np.asarray([row['prompt_id'] for row in metadata])
    unique = np.unique(groups)
    require(len(metadata) == 2699 and len(unique) == 150, 'Historical population differs')
    shuffled = np.random.default_rng(20261018).permutation(unique)
    heldout_groups = shuffled[:30]
    heldout = np.flatnonzero(np.isin(groups, heldout_groups))
    dev = np.flatnonzero(~np.isin(groups, heldout_groups))
    dev_groups = np.random.default_rng(20261018).permutation(np.unique(groups[dev]))
    fold_of = {int(index): fold for fold, selected in enumerate(np.array_split(dev_groups, 5))
               for index in dev[np.isin(groups[dev], selected)]}
    manifest = {
        'seed': 20261018, 'stability_seeds': [20261019, 20261020],
        'train_sha256': train_sha256,
        'development_indices': dev.tolist(), 'holdout_indices': heldout.tolist(),
        'rows': [{'row_index': i, 'id': row['id'], 'prompt_id': row['prompt_id'],
                  'role': 'development' if i in fold_of else 'holdout',
                  'development_fold': fold_of.get(i)} for i, row in enumerate(metadata)]}
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    require(hashlib.sha256(payload).hexdigest() == expected_split_sha256,
            'Original split identity differs')
    require(len(dev) == 2159 and len(heldout) == 540, 'Historical row counts differ')
    return dev.tolist()


def extract_from_bytes(blob, protocol, expected_train_sha256=TRAIN_SHA256):
    """Pure, bounded extraction core; tests supply only generated artificial CSVs."""
    require(isinstance(blob, bytes) and len(blob) <= MAX_SOURCE_BYTES, 'Source byte limit exceeded')
    require(hashlib.sha256(blob).hexdigest() == expected_train_sha256, 'Official source bytes differ')
    sys.path.insert(0, str(ROOT / 'experiments'))
    from r15_typography_replay import DEVELOPMENT_LABELS, validate_development
    from sklearn.model_selection import GroupKFold
    import numpy as np

    metadata = []
    for row in rows(blob):
        require(row['id'].strip() and row['prompt_id'].strip(), 'Missing row identity')
        metadata.append({'id': row['id'], 'prompt_id': row['prompt_id']})
        require(len(metadata) <= 2699, 'Too many source rows')
    require(len({row['id'] for row in metadata}) == len(metadata), 'Duplicate source ID')
    selected = historical_development_indices(metadata, expected_train_sha256,
                                              protocol['historical_split_sha256'])
    wanted = set(selected)
    data = {key: [] for key in ('row_index', 'ids', 'groups', 'truth', 'labels', 'texts', 'fold_id')}
    data['labels'] = list(DEVELOPMENT_LABELS)
    for index, row in enumerate(rows(blob)):
        if index not in wanted:
            continue
        require(row['text'].strip() and row['model'] in DEVELOPMENT_LABELS,
                'Invalid development record')
        data['row_index'].append(index)
        data['ids'].append(row['id'])
        data['groups'].append(row['prompt_id'])
        data['truth'].append(DEVELOPMENT_LABELS.index(row['model']))
        data['texts'].append(row['text'])
    require(data['row_index'] == selected, 'Development row selection differs')
    fold_id = np.full(len(selected), -1, dtype=np.int64)
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5, shuffle=False).split(
            np.arange(len(selected)), groups=data['groups'])):
        fold_id[valid] = fold
    data['fold_id'] = fold_id.tolist()
    validate_development(data, protocol)
    payload = (json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    require(len(payload) <= MAX_OUTPUT_BYTES, 'Development output byte limit exceeded')
    receipt = {
        'schema': 'ntl-approved-raw-development-extraction-v1',
        'source_kind': 'approved_raw_csv_extraction', 'raw_official_csv_opened': True,
        'source_sha256': expected_train_sha256, 'csv_memory_parse_passes': 2,
        'source_rows': 2699, 'development_rows': 2159, 'development_prompts': 120,
        'protected_rows': 540, 'protected_prompts': 30,
        'protected_record_parse_events': 1080,
        'protected_rows_persisted': 0, 'protected_holdout_evaluations': 0,
        'training_fits': 0, 'training_admitted': False,
        'legacy_preisolated_intake_compatible': False,
        'development_sha256': hashlib.sha256(payload).hexdigest(),
        'row_identity_sha256': protocol['row_identity_sha256'],
        'text_sha256': protocol['text_sha256'], 'fold_sha256': protocol['fold_sha256'],
        'historical_split_sha256': protocol['historical_split_sha256']}
    return payload, receipt


def write_quarantine(destination, payload, receipt):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # A partial directory is retained on failure; no overwrites or silent retries.
    with (destination / 'development.json').open('xb') as stream:
        stream.write(payload)
    with (destination / 'extraction_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run_approved(proposal):
    # This gate precedes source stat/open, frozen-protocol loading and claim writes.
    require(isinstance(proposal, dict) and proposal.get('schema') == 'ntl-one-time-raw-export-proposal-v1',
            'Invalid export proposal')
    require(proposal.get('approval') == 'granted'
            and isinstance(proposal.get('approval_reference'), str)
            and bool(proposal['approval_reference'].strip()),
            'Explicit user permission to read protected-row bytes is pending; raw CSV remains closed')
    require(proposal.get('source') == SOURCE.as_posix()
            and proposal.get('source_sha256') == TRAIN_SHA256
            and proposal.get('scope') == 'one_local_export_without_training_or_scoring'
            and proposal.get('output') == 'outputs/r15_raw_export_quarantine'
            and proposal.get('training_admitted') is False
            and proposal.get('legacy_preisolated_intake_compatible') is False,
            'Approved operation differs from the prepared scope')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from r15_validate_dev_export import load_frozen_protocol
    protocol = load_frozen_protocol()
    require(not DESTINATION.exists(), 'Quarantine already exists; no overwrite allowed')
    start = time.process_time()
    CLAIM.parent.mkdir(parents=True, exist_ok=True)
    with CLAIM.open('x', encoding='utf-8') as stream:
        json.dump({'state': 'attempt_started_no_automatic_retry',
                   'approval_reference': proposal['approval_reference'],
                   'source': SOURCE.as_posix()}, stream)
    require(SOURCE.is_file() and SOURCE.stat().st_size <= MAX_SOURCE_BYTES,
            'Official source missing or oversized')
    with SOURCE.open('rb') as stream:
        blob = stream.read(MAX_SOURCE_BYTES + 1)
    payload, receipt = extract_from_bytes(blob, protocol)
    del blob
    receipt['process_cpu_seconds_before_write'] = time.process_time() - start
    require(receipt['process_cpu_seconds_before_write'] <= 30, 'Export preparation exceeded 30 CPU seconds')
    receipt['approval_reference'] = proposal['approval_reference']
    write_quarantine(DESTINATION, payload, receipt)
    return {'status': 'quarantined_not_training_admitted', 'rows': 2159,
            'development_sha256': receipt['development_sha256'],
            'protected_record_parse_events': 1080, 'protected_rows_persisted': 0,
            'training_admitted': False}


def main():
    try:
        proposal = json.loads(PROPOSAL.read_text(encoding='utf-8'))
        report = run_approved(proposal)
        exit_code = 0
    except (OSError, ValueError, TypeError, KeyError, csv.Error, RecursionError) as error:
        # Controlled errors contain counts/scope only, never row representations.
        report = {'status': 'blocked', 'reason': str(error), 'training_admitted': False}
        exit_code = 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
