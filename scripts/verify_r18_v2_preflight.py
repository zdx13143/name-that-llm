"""Read-only artifact gate for the actual R18 V2 preflight and mounted inputs."""
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'outputs/r18'
RUN = E / 'v2_remote_evidence_20261102'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dataset_ref(entry):
    state = entry['state']
    ref = re.search(r'Description: View Dataset, Value: kaggle.com/datasets/([^\s]+)', state).group(1)
    version = int(re.search(r'text  \(Version\s+(\d+)\s*\)', state).group(1))
    return f'{ref}/{version}'


def dates(state):
    return re.findall(r'Help: ((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) [^\n]+Singapore Standard Time\))', state)


def main():
    ui = read(E / 'v2_actual_input_ui_20261102.json')
    pins = read(E / 'pinned_input_ui_20261102.json')
    release = read(E / 'cli_release_v2/kernel-metadata.json')
    assert ui['scriptVersionId'] == 364615652
    assert len(ui['datasets']) == 14 and len(ui['kernels']) == 8
    actual_datasets = list(map(dataset_ref, ui['datasets']))
    expected_datasets = release['dataset_sources']
    assert expected_datasets == list(map(dataset_ref, pins['datasets']))
    dataset_mismatches = [{'expected': expected, 'actual': actual}
        for expected, actual in zip(expected_datasets, actual_datasets) if expected != actual]
    model_refs = re.findall(r'link Value: kaggle.com/models/([^,]+),', ui['initial_state'])
    assert sorted(x.lower() for x in model_refs) == sorted(x.lower() for x in release['model_sources'])
    assert 'kaggle.com/competitions/name-that-llm' in ui['initial_state']
    kernel_checks = []
    for actual, pin, sent in zip(ui['kernels'], pins['kernels'], release['kernel_sources']):
        assert sent == f"{pin['ref']}/{pin['version']}"
        assert 'Value: kaggle.com/code/' + pin['ref'] + '\n' in actual['state']
        assert len(dates(pin['state'])) == 1
        expected_date = dates(pin['state'])[0]
        assert dates(actual['state'])[-1] == expected_date
        assert f"button Version {pin['version']} of {pin['version']}" in pin['state']
        assert f"fork-version/{pin['id']}" in pin['state']
        kernel_checks.append({'ref': pin['ref'], 'version': pin['version'],
            'scriptVersionId': pin['id'], 'actual_mount_updated_at': expected_date,
            'method': 'versioned CLI request + returned slug + exact saved-input timestamp matched to fixed published version'})
    p = read(RUN / 'r18_preflight_receipt.json')
    events = [json.loads(line) for line in (RUN / 'r18_lifecycle.jsonl').read_text().splitlines() if line.strip()]
    assert p['passed'] is True and p['artificial_passages'] == 2
    assert p['gpu_names'] == ['Tesla T4', 'Tesla T4'] and 0 < p['seconds'] < 900
    assert len(events) == p['lifecycle_events'] == 36
    summary = []
    for family in ('q14v5', 'q8v5'):
        adapters = [family + '_f' + str(i) for i in range(5)] + [family]
        f = next(x for x in p['families'] if x['family'] == family)
        assert f['shape'] == [2, 18] and f['finite'] and f['adapters'] == adapters
        rows = [e for e in events if e['family'] == family]
        assert [e['event'] for e in rows] == ['before_load', 'loaded', 'released'] * 6
        assert [e['adapter'] for e in rows] == [a for a in adapters for _ in range(3)]
        loaded, released = rows[1::3], rows[2::3]
        fingerprints = {e['sample_sha256'] for e in loaded}
        assert len(fingerprints) == 1 and len(next(iter(fingerprints))) == 64
        assert all(e['parameter_devices'] == ['cuda:0', 'cuda:1'] for e in loaded)
        floor = rows[0]['allocated']
        assert all(all(now <= before + 16 * 1024 * 1024 for now, before in zip(e['allocated'], floor)) for e in released)
        summary.append({'family': family, 'adapters_completed': 6, 'fingerprint': next(iter(fingerprints)),
            'load_seconds': [e['seconds'] for e in loaded], 'release_allocated_bytes': [e['allocated'] for e in released]})
    log = read(RUN / 'ntl-r18-adapter-isolation-20261101.log')
    passes = [x for x in log if x['data'].startswith('R18_PREFLIGHT_PASS ')]
    assert len(passes) == 1 and json.loads(passes[0]['data'].split(' ', 1)[1]) == p
    assert passes[0]['time'] < 900
    assert any('R14 placeholder only: three rows' in x['data'] for x in log)
    placeholder = pd.read_parquet(RUN / 'submission.parquet')
    assert list(placeholder.columns) == ['id', 'model'] and len(placeholder) == 3
    assert placeholder['id'].is_unique and not placeholder.isna().any().any()
    source = read(ROOT / 'configs/r18_v2_source_readback_20261102.json')
    assert source['scriptVersionId'] == 364615652 and source['saved_code_cells_equal_frozen_candidate']
    result = {'release_gate_passed': not dataset_mismatches, 'version': 2, 'scriptVersionId': 364615652,
        'candidate_sha256': read(ROOT / 'configs/r18_v2_repair_20261102.json')['candidate_sha256'],
        'actual_datasets_inspected': 14, 'actual_dataset_versions_matching': 14 - len(dataset_mismatches),
        'dataset_version_mismatches': dataset_mismatches,
        'actual_models_verified': 7, 'actual_competitions_verified': 1,
        'actual_kernel_version_mapping': kernel_checks,
        'kernel_mapping_limit': 'Official SDK and input page do not expose attached kernel version IDs directly; timestamp mapping and exact versioned upload request are recorded explicitly.',
        'preflight': p, 'family_lifecycle': summary, 'save_run_seconds_ui': 252.5,
        'download_provenance': 'V2 is latest and only successful saved version; no V3; run log PASS receipt equals downloaded receipt, and saved source readback matches V2 exactly',
        'placeholder_rows': 3, 'placeholder_is_accuracy_evidence': False,
        'evidence_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in RUN.iterdir() if f.is_file()},
        'formal_submissions_before_release': 0, 'formal_submission_limit': 1,
        'new_automations': 0, 'accuracy_or_score_claim': None}
    (ROOT / 'configs/r18_v2_release_gate_20261102.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'release_gate_passed': not dataset_mismatches, 'version': 2, 'scriptVersionId': 364615652,
        'dataset_version_mismatches': dataset_mismatches,
        'inputs': 30, 'adapters_completed': 12, 'lifecycle_events': 36, 'preflight_seconds': p['seconds'],
        'placeholder_rows': 3, 'formal_submissions': 0}, indent=2))
    if dataset_mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
