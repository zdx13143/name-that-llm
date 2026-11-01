"""Verify a saved R18 source without importing or executing model code."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_r18_cli_release import code_sources, ROOT, EVIDENCE

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=int, choices=(1, 2), required=True)
parser.add_argument('--script-version-id', type=int, required=True)
args = parser.parse_args()
assert args.script_version_id == {1: 364611063, 2: 364615652}[args.version], 'Unobserved saved version identity'
suffix = '' if args.version == 1 else '_v2'
expected = json.loads((ROOT / f'notebooks/team_e659_r18_adapter_isolation{suffix}.ipynb').read_text(encoding='utf-8'))
saved_path = EVIDENCE / f'v{args.version}_cli_readback_20261102/ntl-r18-adapter-isolation-20261101.ipynb'
saved = json.loads(saved_path.read_text(encoding='utf-8'))
actual_code = code_sources(saved)
assert actual_code == code_sources(expected), 'Saved prediction/wrapper code changed'
meta = json.loads((saved_path.parent / 'kernel-metadata.json').read_text())
release = json.loads((EVIDENCE / f'cli_release{suffix}/kernel-metadata.json').read_text())
for key in ('is_private', 'enable_gpu', 'enable_tpu', 'enable_internet', 'machine_shape', 'docker_image'):
    assert meta[key] == release[key], key
for key in ('dataset_sources', 'kernel_sources'):
    assert meta[key] == [ref.rsplit('/', 1)[0] for ref in release[key]], key
assert set(meta['model_sources']) == set(release['model_sources'])
assert meta['competition_sources'] == release['competition_sources']
receipt = {'version_number': args.version, 'scriptVersionId': args.script_version_id, 'kernel_id': meta['id_no'],
    'saved_code_cells_equal_frozen_candidate': True, 'code_cell_count': len(actual_code),
    'code_sha256': [hashlib.sha256(c.encode()).hexdigest() for c in actual_code],
    'readback_file_sha256': hashlib.sha256(saved_path.read_bytes()).hexdigest(),
    'docker_image_exact_match': True, 'private': True, 'internet': False, 'machine_shape': meta['machine_shape'],
    'source_slug_sets_equal': True, 'version_suffixes_not_returned_by_cli_metadata': True,
    'actual_input_version_ui_check': 'required separately', 'formal_submissions': 0}
(ROOT / f'configs/r18_v{args.version}_source_readback_20261102.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
