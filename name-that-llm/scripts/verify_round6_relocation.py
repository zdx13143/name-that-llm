"""Extract the exact payload in Linux temporary storage; no encoder forwards."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile


def main():
    if sys.platform!='linux': raise RuntimeError('Linux runtime verification required')
    module=Path(__file__).resolve().parents[1]
    package=module/'outputs/kaggle_r6_package'
    receipt=json.loads((package/'package_receipt.json').read_text())
    payload=package/receipt['payload_file']
    h=hashlib.sha256()
    with payload.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    if h.hexdigest()!=receipt['payload_sha256']: raise ValueError('Payload bytes changed')
    destination=Path(tempfile.mkdtemp(prefix='ntl-r6-relocate-'))
    with tarfile.open(payload,'r:gz') as archive:
        members=archive.getmembers()
        roots={Path(m.name).parts[0] for m in members}
        if roots!={'ntl-r6-assets','cpython-3.11.17-linux-x86_64-gnu'}:
            raise ValueError('Unexpected package roots')
        archive.extractall(destination,filter='data')
    python=destination/'cpython-3.11.17-linux-x86_64-gnu/bin/python3.11'
    assets=destination/'ntl-r6-assets'
    source=module/'outputs/kaggle_r6_preflight/linux_cpu_float32/synthetic.npz'
    code='''import json,sys
from pathlib import Path
import numpy as np
import torch, transformers, scipy, sklearn, pyarrow
from threadpoolctl import threadpool_limits
from inference_round6 import block_network,verify_assets,verify_runtime,load_heads,predict,TEXTS
block_network()
root=Path(sys.argv[1])
assert Path(sys.prefix).resolve().parent==root.parent
verify_assets(root)
versions=verify_runtime('cpu_float32')
with np.load(sys.argv[2],allow_pickle=False) as handle:
    features=handle['features'].copy()
    reference=handle['probabilities'].copy()
with threadpool_limits(limits=2):
    probability,_,_=predict(TEXTS,features,load_heads(root))
error=float(np.max(np.abs(probability-reference)))
assert error<=1e-12,error
assert np.array_equal(probability.argmax(1),reference.argmax(1))
print(json.dumps({'state':'verified','relocated_prefix':sys.prefix,'versions':versions,
                  'max_abs_head_prediction_difference':error,'encoder_forward_calls':0,
                  'training_fits':0,'scope':'imports and frozen-head predictions on saved synthetic features only'}))
'''
    env=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH'): env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    result=subprocess.run([str(python),'-c',code,str(assets),str(source)],cwd=assets,env=env,
                          text=True,capture_output=True,timeout=300,check=True)
    report=json.loads(result.stdout)
    report.update(payload_sha256=h.hexdigest(),archive_members=len(members),temporary_directory=str(destination))
    target=package/'relocation_verification.json'
    with target.open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__': main()
