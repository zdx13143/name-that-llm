"""Build an allowlisted, private Kaggle R6 inference bundle; no data or fitting."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

MODULE = Path(__file__).resolve().parents[1]
FINAL = MODULE/'outputs/windows_calibrated_final_20261025_r6'
ASSETS = MODULE/'outputs/kaggle_r6_assets/ntl-r6-assets'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    from verify_round6_results import check
    if check()['development_gate_passed'] is not True:
        raise ValueError('Development gate must pass')
    claim=json.loads((MODULE/'outputs/continuation_claims/windows-calibrated-fusion-final-20261025-r6.json').read_text())
    report=json.loads((FINAL/'summary.json').read_text())
    if claim['state']!='completed' or claim['summary_sha256']!=digest(FINAL/'summary.json') or claim['supervisor_sha256']!=digest(FINAL/'supervisor.json'):
        raise ValueError('Final export is not reconciled')
    for name,sha in report['model_hashes'].items():
        if digest(FINAL/name)!=sha: raise ValueError(name)
    if (report['classifier_fits'],report['temperature_fits'],report['old_holdout_evaluations'])!=(8,2,0):
        raise ValueError('Final scope changed')
    base=json.loads((MODULE/'config/windows_modernbert_20261023.json').read_text())
    source=MODULE/'models/modernbert-base'/base['revision']
    from probe_modernbert import validate_files
    validate_files(source,base['model_files'])
    ASSETS.mkdir(parents=True,exist_ok=False)
    (ASSETS/'encoder').mkdir()
    for item in base['model_files']:
        shutil.copyfile(source/item['rfilename'],ASSETS/'encoder'/item['rfilename'])
    for name in ('style.joblib','gaussian.npz','calibration.json'):
        shutil.copyfile(FINAL/name,ASSETS/name)
    shutil.copyfile(MODULE/'config/labels.json',ASSETS/'labels.json')
    for name in ('stylegram_round5.py','inference_round6.py'):
        shutil.copyfile(MODULE/'scripts'/name,ASSETS/name)
    files={p.relative_to(ASSETS).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
           for p in sorted(ASSETS.rglob('*')) if p.is_file()}
    manifest={'candidate':'windows-calibrated-fusion-dev-20261025-r6','style_weight':.8,
              'development_ba':.370895580454404,'development_rows':2159,'development_prompts':120,
              'old_holdout_used':False,'final_summary_sha256':digest(FINAL/'summary.json'),
              'encoder_revision':base['revision'],'feature':base['feature'],
              'deployment_dtype':'float32 CPU; requires synthetic compatibility gate against Windows CUDA BF16',
              'files':files}
    (ASSETS/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'assets':str(ASSETS),'files':len(files),'manifest_sha256':digest(ASSETS/'manifest.json')}))


if __name__=='__main__': main()
