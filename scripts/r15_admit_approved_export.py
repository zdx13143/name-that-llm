"""Admit the reviewed, user-approved extraction; never open the raw source.

This separate entry point preserves the old preisolated-only intake contract.
Only data identity is admitted: training and scoring remain disabled.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from r15_validate_dev_export import (  # noqa: E402
    FIELDS, TRAIN_SHA256, bounded_file, canonical_hash, load_frozen_protocol,
    nested_plan, require, validate_development,
)

APPROVAL = 'user-approved-20261029-one-isolated-export-and-development-admission'
QUARANTINE = ROOT / 'outputs/r15_raw_export_quarantine'
DESTINATION = ROOT / 'outputs/r15_dev_export'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def metadata_gate(receipt, policy, review, protocol):
    require(all(isinstance(x, dict) for x in (receipt, policy, review)), 'Sidecars must be objects')
    require(policy.get('schema') == 'ntl-approved-extraction-admission-policy-v1'
            and policy.get('approval_reference') == APPROVAL
            and policy.get('approval') == 'granted', 'Explicit admission approval is missing')
    require(policy.get('protocol_sha256') == canonical_hash(protocol), 'Protocol binding differs')
    require(policy.get('training_enabled') is False and policy.get('real_scoring_enabled') is False,
            'Data admission cannot enable training or scoring')
    require(type(policy.get('raw_source_rereads_allowed')) is int
            and policy['raw_source_rereads_allowed'] == 0
            and policy.get('single_source_directory') == 'outputs/r15_raw_export_quarantine',
            'Only the existing quarantine is allowed; raw-source rereads are forbidden')
    require(receipt.get('schema') == 'ntl-approved-raw-development-extraction-v1'
            and receipt.get('source_kind') == 'approved_raw_csv_extraction'
            and receipt.get('approval_reference') == APPROVAL, 'Wrong extraction source')
    require(receipt.get('source_sha256') == TRAIN_SHA256
            and receipt.get('raw_official_csv_opened') is True
            and receipt.get('legacy_preisolated_intake_compatible') is False,
            'Extraction history must be preserved')
    counts = {'csv_memory_parse_passes': 2, 'source_rows': 2699, 'protected_rows': 540,
              'protected_prompts': 30, 'protected_record_parse_events': 1080,
              'protected_rows_persisted': 0, 'protected_holdout_evaluations': 0, 'training_fits': 0}
    counts.update({k: protocol[k] for k in ('development_rows', 'development_prompts')})
    require(all(type(receipt.get(k)) is int and receipt[k] == v for k, v in counts.items()),
            'Extraction counters differ')
    require(receipt.get('training_admitted') is False, 'Unexpected training admission')
    for key in ('row_identity_sha256', 'text_sha256', 'fold_sha256', 'historical_split_sha256'):
        require(receipt.get(key) == protocol[key], 'Extraction identity differs')
    require(policy.get('extraction_receipt_sha256') == canonical_hash(receipt)
            and policy.get('development_sha256') == receipt.get('development_sha256'),
            'Policy does not bind this extraction')
    require(review.get('schema') == 'ntl-approved-extraction-review-v1'
            and review.get('conclusion') == 'approved_extraction_identity_verified'
            and review.get('policy_sha256') == canonical_hash(policy)
            and review.get('extraction_receipt_sha256') == canonical_hash(receipt)
            and review.get('development_sha256') == receipt['development_sha256'],
            'Independent review is missing or stale')
    require(isinstance(review.get('reviewer'), str) and bool(review['reviewer'].strip())
            and review['reviewer'] != policy.get('producer'), 'Independent reviewer required')


def load_approved_export(directory, policy, review, protocol):
    receipt = json.loads(bounded_file(directory, 'extraction_receipt.json', 65536))
    metadata_gate(receipt, policy, review, protocol)
    blob = bounded_file(directory, 'development.json', 8 * 1024**2)
    require(sha(blob) == receipt['development_sha256'], 'Development bytes changed')
    data = json.loads(blob)
    require(isinstance(data, dict) and set(data) == FIELDS, 'Unexpected development schema')
    require(all(isinstance(data[k], list) for k in FIELDS), 'Every field must be a list')
    for key in ('row_index', 'truth', 'fold_id'):
        require(all(type(v) is int for v in data[key]), 'Non-integer index')
    for key in ('ids', 'groups', 'texts', 'labels'):
        require(all(isinstance(v, str) and v.strip() for v in data[key]), 'Invalid string field')
    require(data['row_index'] == sorted(data['row_index'])
            and min(data['row_index'], default=-1) >= 0, 'Original row order required')
    validate_development(data, protocol)
    return data, blob, receipt


def write_admitted(destination, blob, extraction, policy, review, plan):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # A failed/partial write is preserved. No automatic overwrite or retry.
    with (destination / 'development.json').open('xb') as stream:
        stream.write(blob)
    admission = {'schema': 'ntl-development-data-admission-v1', 'data_admitted': True,
                 'development_sha256': sha(blob), 'extraction_receipt_sha256': canonical_hash(extraction),
                 'policy_sha256': canonical_hash(policy), 'review_sha256': canonical_hash(review),
                 'nested_plan_sha256': canonical_hash(plan),
                 'training_enabled': False, 'real_scoring_enabled': False,
                 'raw_source_reads_by_admission': 0, 'protected_rows_in_admitted_data': 0}
    for name, value in [('extraction_receipt.json', extraction), ('admission_policy.json', policy),
                        ('admission_review.json', review), ('nested_plan.json', plan),
                        ('admission_receipt.json', admission)]:
        with (destination / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    return admission


def main():
    start = time.process_time()
    try:
        require(not DESTINATION.exists(), 'Admission destination already exists; no overwrite')
        protocol = load_frozen_protocol()
        policy = json.loads(bounded_file(ROOT / 'configs', 'r15_approved_export_policy_20261029.json', 65536))
        review = json.loads(bounded_file(ROOT / 'configs', 'r15_approved_export_review_20261029.json', 65536))
        data, blob, extraction = load_approved_export(QUARANTINE, policy, review, protocol)
        plan = nested_plan(data, protocol)
        admission = write_admitted(DESTINATION, blob, extraction, policy, review, plan)
        report = dict(admission, status='development_data_admitted_not_training_enabled',
                      rows=len(data['ids']), prompts=len(set(data['groups'])),
                      outer_folds=len(plan['folds']), inner_folds_total=25)
        code = 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        report = {'status': 'blocked', 'reason': str(exc), 'training_enabled': False,
                  'real_scoring_enabled': False}
        code = 2
    report['process_cpu_seconds'] = time.process_time() - start
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
