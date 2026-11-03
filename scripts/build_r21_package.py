"""Prepare a private local Kaggle package; never calls Kaggle or runs CV."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = Path('D:/kaggle/name_that_llm/outputs/worktrees/team-e654-r15-20261029/outputs/r15_dev_export/development.json')
DEV_SHA = '1c9446efa6c8c477bd68448f1de212aadcc8dc90224b76728ec2f70e2ec2334f'
SCRIPTS = ('r19_overlap_audit.py', 'r19_cpu_cv.py', 'r19_svc_component.py',
           'r21_members.py', 'r21_sinkhorn.py', 'r21_scorebank_cv.py')


def main():
    raw = DEV.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DEV_SHA:
        raise ValueError('development data hash mismatch')
    dest = ROOT / 'kaggle/r21_dataset'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'development.json').write_bytes(raw)
    for name in SCRIPTS:
        shutil.copy2(ROOT / 'scripts' / name, dest / name)
    target_source = dest / 'r21_source'
    target_source.mkdir(exist_ok=True)
    for source in (ROOT / 'scripts/r21_source').glob('*.txt'):
        shutil.copy2(source, target_source / source.name)
    metadata = {'title': 'NTL R21 scorebank package', 'id': 'zephyrpong/ntl-r21-scorebank-package',
                'licenses': [{'name': 'unknown'}]}
    (dest / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
