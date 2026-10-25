"""At most four one-shot synthetic forward calls across Windows and WSL."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from inference_round6 import digest, write_json

MODULE=Path(__file__).resolve().parents[1]
BASE=MODULE/'outputs/kaggle_r6_preflight'
CONTRACT=MODULE/'config/round6_deployment_contract_20261025.json'
ASSETS=MODULE/'outputs/kaggle_r6_assets/ntl-r6-assets'
LINUXPY='/mnt/d/kaggle/name_that_llm/name-that-llm/outputs/kaggle_r6_runtime/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11'


def linux_path(path):
    return '/mnt/'+path.drive[0].lower()+'/'+path.as_posix()[3:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['windows_cuda_bfloat16','windows_cpu_float32','linux_cpu_float32','linux_cpu_reverse','verify'])
    args=p.parse_args()
    spec=json.loads(CONTRACT.read_text())
    ledger_path=BASE/'ledger.json'
    if args.stage=='verify':
        import numpy as np
        ledger=json.loads(ledger_path.read_text())
        if [s['stage'] for s in ledger['stages']]!=spec['stages'] or any(s['state']!='completed' for s in ledger['stages']):
            raise ValueError('All four stages must complete')
        arrays=[]
        claims=[]
        for stage in spec['stages']:
            claim=json.loads((BASE/stage/'claim.json').read_text())
            if claim['state']!='completed' or claim['attempted_forward_calls']!=1 or claim['completed_forward_calls']!=1:
                raise ValueError('Incomplete synthetic stage')
            claims.append(claim)
            with np.load(BASE/stage/'synthetic.npz',allow_pickle=False) as f:
                arrays.append({k:f[k].copy() for k in f.files})
        arrays[-1]={k:v[::-1] for k,v in arrays[-1].items()}
        reference=arrays[0]
        checks=[]
        for stage,value in zip(spec['stages'],arrays):
            np.testing.assert_array_equal(value['ids'],reference['ids'])
            np.testing.assert_array_equal(value['probabilities'].argmax(1),reference['probabilities'].argmax(1))
            probability_error=float(abs(value['probabilities']-reference['probabilities']).max())
            style_error=float(abs(value['style_scores']-reference['style_scores']).max())
            if probability_error > spec['max_abs_fusion_probability_difference'] or style_error > spec['max_abs_cross_platform_style_score_difference']:
                raise ValueError(f'Compatibility failed: {stage}, {probability_error}, {style_error}')
            checks.append({'stage':stage,'max_abs_probability_difference':probability_error,'max_abs_style_score_difference':style_error})
        reverse_error=float(abs(arrays[2]['probabilities']-arrays[3]['probabilities']).max())
        if reverse_error > spec['linux_batch_reverse_max_abs_probability_difference']:
            raise ValueError('Batch order mismatch')
        result={'state':'verified','scope':'4 artificial texts only; no development or holdout evaluation',
                'encoder_forward_calls':4,'checks':checks,'linux_reverse_max_abs_difference':reverse_error,
                'ledger_sha256':digest(ledger_path),'manifest_sha256':digest(ASSETS/'manifest.json'),
                'deployment_mode':'cpu_float32','synthetic_cpu_seconds':claims[2]['wall_seconds'],
                'limitation':'CPU float32 differs from development CUDA BF16; synthetic compatibility is not proof of identical predictions on all texts.'}
        write_json(BASE/'verification.json',result,exclusive=True)
        print(json.dumps(result))
        return
    BASE.mkdir(exist_ok=True)
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        if ledger['contract_sha256']!=digest(CONTRACT) or ledger['manifest_sha256']!=digest(ASSETS/'manifest.json'):
            raise ValueError('Preflight identity changed')
    else:
        ledger={'contract_sha256':digest(CONTRACT),'manifest_sha256':digest(ASSETS/'manifest.json'),'stages':[]}
        write_json(ledger_path,ledger,exclusive=True)
    if len(ledger['stages'])>=4 or any(s['state']!='completed' for s in ledger['stages']) or args.stage!=spec['stages'][len(ledger['stages'])]:
        raise ValueError('No retry, reorder, or extra stage allowed')
    mode='cuda_bfloat16' if args.stage=='windows_cuda_bfloat16' else 'cpu_float32'
    output=BASE/args.stage
    if args.stage.startswith('linux'):
        command=['wsl.exe','-d','Ubuntu','--',LINUXPY,linux_path(ASSETS/'inference_round6.py'),'synthetic',
                 '--assets',linux_path(ASSETS),'--output',linux_path(output),'--mode',mode]
    else:
        command=[sys.executable,str(ASSETS/'inference_round6.py'),'synthetic','--assets',str(ASSETS),'--output',str(output),'--mode',mode]
    if args.stage=='linux_cpu_reverse': command+=['--reverse']
    entry={'stage':args.stage,'state':'started','maximum_forward_attempts':1}
    ledger['stages'].append(entry)
    write_json(ledger_path,ledger)
    try:
        with (BASE/(args.stage+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=930,check=False)
        entry.update(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode)
        write_json(ledger_path,ledger)
        if result.returncode: raise RuntimeError(f'Stage failed, retain log: {args.stage}')
        print(json.dumps(entry))
    except BaseException:
        entry['state']='failed'
        write_json(ledger_path,ledger)
        raise


if __name__=='__main__': main()
