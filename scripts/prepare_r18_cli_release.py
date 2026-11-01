"""Package the unchanged R18 candidate with independently pinned CLI inputs."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/r18'
BASE_READBACK = EVIDENCE / 'r16_cli_readback_20261102'


def code_sources(nb):
    return [''.join(c['source']) if isinstance(c['source'], list) else c['source']
            for c in nb['cells'] if c['cell_type'] == 'code']


def prepare():
    recipe = json.loads((ROOT / 'configs/r18_recipe_20261101.json').read_text())
    candidate = ROOT / recipe['candidate']
    raw = candidate.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == recipe['candidate_sha256']
    readback = json.loads((BASE_READBACK / 'ntl-r16-typography-half-20261029.ipynb').read_text(encoding='utf-8'))
    frozen_base = json.loads((ROOT / 'notebooks/team_e654_r16_typography_half_runnable.ipynb').read_text(encoding='utf-8'))
    assert code_sources(readback) == code_sources(frozen_base), 'CLI baseline code does not match scored R16'
    metadata = json.loads((BASE_READBACK / 'kernel-metadata.json').read_text())
    evidence = json.loads((EVIDENCE / 'pinned_input_ui_20261102.json').read_text(encoding='utf-8'))
    assert evidence['baseline_version'] == 363487968
    datasets = []
    for entry in evidence['datasets']:
        ref = re.search(r'Description: View Dataset, Value: kaggle.com/datasets/([^\s]+)', entry['state']).group(1)
        version = int(re.search(r'text  \(Version\s+(\d+)\s*\)', entry['state']).group(1))
        datasets.append(f'{ref}/{version}')
    assert [x.rsplit('/', 1)[0] for x in datasets] == metadata['dataset_sources']
    assert len(datasets) == 14
    kernels = [f"{x['ref']}/{x['version']}" for x in evidence['kernels']]
    assert [x.rsplit('/', 1)[0] for x in kernels] == metadata['kernel_sources']
    assert [x['id'] for x in evidence['kernels']] == [x['sourceId'] for x in recipe['inputs'] if x['sourceType'] == 'kernelVersion']
    assert len(kernels) == 8 and len(metadata['model_sources']) == 7
    assert metadata['competition_sources'] == ['name-that-llm']
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_tpu'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['docker_image'].startswith('gcr.io/kaggle-private-byod/python@sha256:')
    metadata.pop('id_no')
    metadata.update(id='zephyrpong/ntl-r18-adapter-isolation-20261101',
        title='NTL R18 Adapter Isolation 20261101', code_file=candidate.name,
        dataset_sources=datasets, kernel_sources=kernels, docker_image_pinning_type='original')
    target = EVIDENCE / 'cli_release'
    target.mkdir(exist_ok=True)
    shutil.copyfile(candidate, target / candidate.name)
    (target / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    receipt = {'candidate_sha256': recipe['candidate_sha256'], 'baseline_code_parity': True,
        'cli_source_readback': str(BASE_READBACK.relative_to(ROOT)), 'baseline_script_version_id': 363487968,
        'dataset_versions_observed_in_exact_baseline_ui': datasets, 'kernel_versions_and_ids': [
            {k: x[k] for k in ('ref', 'id', 'version')} for x in evidence['kernels']],
        'model_sources': metadata['model_sources'], 'docker_image': metadata['docker_image'],
        'machine_shape': metadata['machine_shape'], 'internet': False, 'private': True,
        'actual_t4x2_required_by_preflight': True, 'prediction_source_changes': 0,
        'cli_pull_version_suffix_attempt': '403; no content; unversioned source subsequently verified against frozen R16 code',
        'authentication': 'existing official OAuth credentials refreshed normally; no secrets copied to project'}
    (ROOT / 'configs/r18_cli_release_20261102.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'candidate_sha256': recipe['candidate_sha256'], 'baseline_code_parity': True,
        'dataset_count': len(datasets), 'kernel_count': len(kernels), 'model_count': 7, 'total_inputs': 30,
        'machine_shape': metadata['machine_shape'], 'release_directory': str(target)}, indent=2))


if __name__ == '__main__':
    prepare()
