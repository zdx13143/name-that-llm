"""Linux-only allowlisted packaging; no encoder or fitting."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sys
import tarfile


def main():
    if sys.platform!='linux': raise RuntimeError('Preserve Linux binary modes and symlinks')
    module=Path(__file__).resolve().parents[1]
    assets=module/'outputs/kaggle_r6_assets/ntl-r6-assets'
    proof=module/'outputs/kaggle_r6_preflight/verification.json'
    value=json.loads(proof.read_text())
    if value['state']!='verified': raise ValueError('Portability gate has not passed')
    manifest=assets/'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=value['manifest_sha256']:
        raise ValueError('Verified manifest changed')
    runtime=module/'outputs/kaggle_r6_runtime/python/cpython-3.11.17-linux-x86_64-gnu'
    if Path(sys.prefix).resolve()!=runtime.resolve(): raise ValueError('Use the prepared runtime')
    output=module/'outputs/kaggle_r6_package'
    output.mkdir(exist_ok=False)
    # Only aggregate synthetic receipts, no arrays/texts from competition records.
    shutil.copyfile(proof,assets/'portability_verification.json')
    versions={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()}
    (assets/'runtime_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    allowed=set(json.loads(manifest.read_text())['files']) | {'manifest.json','portability_verification.json','runtime_versions.json'}
    actual={p.relative_to(assets).as_posix() for p in assets.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=allowed: raise ValueError('Unexpected files in asset directory')
    def archive_filter(info):
        if '__pycache__' in Path(info.name).parts: return None
        if info.issym() and Path(info.linkname).is_absolute():
            raise ValueError(f'Absolute runtime symlink: {info.name}')
        info.uid=info.gid=0
        info.uname=info.gname=''
        return info
    payload=output/'ntl_r6_payload.tgz.bin'
    with tarfile.open(payload,'x:gz',compresslevel=3) as archive:
        archive.add(assets,arcname=assets.name,filter=archive_filter)
        archive.add(runtime,arcname=runtime.name,filter=archive_filter)
    (output/'README.txt').write_text('Private frozen inference assets for NTL Calibrated R6 20261025.\n'
        'No competition raw data, protected holdout, credentials or development OOF are included.\n'
        'The .bin file is a gzip-compressed tar archive read by the companion notebook.\n'
        'It includes ModernBERT-base fixed public revision 8949b909ec900327062f0ebf497f51aef5e6f0c8,\n'
        'frozen development-only classifier heads, an isolated CPython 3.11 CPU runtime and package licenses.\n')
    print(json.dumps({'payload':str(payload),'bytes':payload.stat().st_size,'runtime_packages':len(versions)}))


if __name__=='__main__': main()
