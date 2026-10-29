"""Validate a pre-isolated development export, never reconstruct from train.csv.

Metadata/review gates precede every payload read. Review is a human trust
boundary: hashes check consistency, not the truth of a producer's history.
This command does not enable training, scoring or the original R15 replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from r15_typography_replay import (  # noqa: E402
    canonical_hash, is_digest, require, validate_development,
)

TRAIN_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
PROTOCOL_CANONICAL_SHA256 = 'da3538f8566672a7e3bb6a34a5436a7e21118459e7bdd91e507f4d56b39f5c6c'
FIELDS = {'row_index', 'ids', 'groups', 'truth', 'labels', 'texts', 'fold_id'}


def bounded_file(directory, name, maximum):
    """Only fixed basenames within this directory, including after symlinks."""
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    require(path.parent == directory, 'Export file escapes its directory')
    require(path.is_file(), f'Missing {name}; no raw-data fallback exists')
    require(path.stat().st_size <= maximum, f'{name} exceeds byte limit')
    with path.open('rb') as handle:
        blob = handle.read(maximum + 1)
    require(len(blob) <= maximum, f'{name} grew beyond byte limit')
    return blob


def metadata_gate(receipt, review, protocol):
    require(isinstance(receipt, dict) and isinstance(review, dict), 'Sidecars must be JSON objects')
    require(receipt.get('schema') == 'ntl-development-only-export-v1', 'Wrong export schema')
    require(receipt.get('source_kind') == 'preexisting_development_only_artifact',
            'Raw CSV reconstruction and unknown sources are not permitted')
    require(receipt.get('all_input_sources_development_only') is True,
            'Every input must already exclude protected content')
    require(receipt.get('raw_official_csv_opened') is False, 'Raw official CSV must remain closed')
    count = receipt.get('protected_content_reads')
    require(type(count) is int and count == 0, 'Protected content reads must be integer zero')
    require(receipt.get('historical_train_sha256') == TRAIN_SHA256, 'Historical source differs')
    for key in ('row_identity_sha256', 'text_sha256', 'fold_sha256', 'historical_split_sha256'):
        require(receipt.get(key) == protocol[key], f'Export identity mismatch: {key}')
    for key in ('development_rows', 'development_prompts', 'folds'):
        require(type(receipt.get(key)) is int and receipt[key] == protocol[key],
                f'Export population differs: {key}')
    require(is_digest(receipt.get('development_file_sha256')), 'Missing export byte hash')
    require(is_digest(receipt.get('source_evidence_sha256')), 'Missing source evidence hash')
    require(isinstance(receipt.get('producer'), str) and receipt['producer'].strip(),
            'Missing producer identity')
    require(review.get('schema') == 'ntl-development-export-review-v1', 'Wrong review schema')
    require(review.get('receipt_sha256') == canonical_hash(receipt), 'Review does not bind receipt')
    require(review.get('source_evidence_sha256') == receipt['source_evidence_sha256'],
            'Review does not bind source evidence')
    require(review.get('conclusion') == 'preisolated_development_source_verified',
            'Source isolation has not been independently reviewed')
    require(isinstance(review.get('reviewer'), str) and review['reviewer'].strip()
            and review['reviewer'] != receipt['producer'], 'Independent reviewer required')


def load_frozen_protocol():
    protocol = json.loads(bounded_file(ROOT / 'configs', 'r15_protocol_20261029.json', 65536))
    require(isinstance(protocol, dict), 'Protocol must be a JSON object')
    require(canonical_hash(protocol) == PROTOCOL_CANONICAL_SHA256,
            'Frozen R15 protocol changed; re-admission required')
    return protocol


def load_export(directory, protocol):
    # Both sidecars must be safe metadata, never embedded texts or row dumps.
    receipt = json.loads(bounded_file(directory, 'development_export_receipt.json', 65536))
    review = json.loads(bounded_file(directory, 'development_export_review.json', 65536))
    metadata_gate(receipt, review, protocol)
    evidence = bounded_file(directory, 'source_evidence.md', 262144)
    require(hashlib.sha256(evidence).hexdigest() == receipt['source_evidence_sha256'],
            'Reviewed source evidence changed')
    blob = bounded_file(directory, 'development.json', 8 * 1024**2)
    require(hashlib.sha256(blob).hexdigest() == receipt['development_file_sha256'],
            'Development bytes changed')
    data = json.loads(blob)
    require(isinstance(data, dict) and set(data) == FIELDS, 'Unexpected development schema')
    require(all(isinstance(data[k], list) for k in FIELDS), 'Every development field must be a list')
    for key in ('row_index', 'truth', 'fold_id'):
        require(all(type(x) is int for x in data[key]), f'Non-integer {key}')
    for key in ('ids', 'groups', 'texts', 'labels'):
        require(all(isinstance(x, str) and x.strip() for x in data[key]), f'Invalid {key}')
    require(data['row_index'] == sorted(data['row_index']) and min(data['row_index'], default=-1) >= 0,
            'Original row order required')
    validate_development(data, protocol)
    return data, receipt


def nested_plan(data, protocol):
    """Plan metadata only. Never fit, score, tokenize, or extract features."""
    import numpy as np
    from sklearn.model_selection import GroupKFold

    _, outer, groups = validate_development(data, protocol)
    rows = np.arange(len(outer))
    result = []
    for fold in range(protocol['folds']):
        train, valid = rows[outer != fold], rows[outer == fold]
        inner = []
        for inner_fold, (fit, validation) in enumerate(
                GroupKFold(n_splits=5).split(train, groups=groups[train])):
            fit, validation = train[fit], train[validation]
            require(not (set(groups[fit]) & set(groups[validation])), 'Inner prompt leakage')
            require(not (set(groups[valid]) & (set(groups[fit]) | set(groups[validation]))),
                    'Outer validation entered inner calibration')
            inner.append({'inner_fold': inner_fold,
                          'train_rows': fit.tolist(), 'valid_rows': validation.tolist(),
                          'train_groups_sha256': canonical_hash(sorted(set(groups[fit].tolist()))),
                          'valid_groups_sha256': canonical_hash(sorted(set(groups[validation].tolist())))})
        result.append({'outer_fold': fold, 'train_rows': train.tolist(), 'valid_rows': valid.tolist(),
                       'train_groups_sha256': canonical_hash(sorted(set(groups[train].tolist()))),
                       'valid_groups_sha256': canonical_hash(sorted(set(groups[valid].tolist()))),
                       'inner': inner})
    return {'schema': 'ntl-nested-development-plan-v1', 'producer': 'r15p1-nested-dev120-draft',
            'training_enabled': False, 'real_scoring_enabled': False,
            'row_number_semantics': 'positions in development.json, not full train.csv',
            'fold_sha256': protocol['fold_sha256'], 'folds': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'outputs/r15_dev_export')
    args = parser.parse_args()
    start = time.process_time()
    try:
        protocol = load_frozen_protocol()
        data, receipt = load_export(args.directory, protocol)
        plan = nested_plan(data, protocol)
        report = {'status': 'development_identity_verified_not_training_admitted',
                  'rows': len(data['ids']), 'prompts': len(set(data['groups'])),
                  'export_sha256': receipt['development_file_sha256'],
                  'nested_plan_sha256': canonical_hash(plan),
                  'outer_folds': len(plan['folds']), 'inner_folds_total': 25,
                  'training_enabled': False, 'real_scoring_enabled': False}
        # No row-level output and no automatic bundle/provenance admission.
        exit_code = 0
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        report = {'status': 'blocked', 'reason': str(exc),
                  'training_enabled': False, 'real_scoring_enabled': False}
        exit_code = 2
    report['process_cpu_seconds'] = time.process_time() - start
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
