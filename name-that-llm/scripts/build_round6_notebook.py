"""Generate one offline Kaggle notebook from the verified R6 payload."""
from pathlib import Path
import hashlib
import json

MODULE=Path(__file__).resolve().parents[1]
PAYLOAD=MODULE/'outputs/kaggle_r6_package/ntl_r6_payload.tgz.bin'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    receipt=MODULE/'outputs/kaggle_r6_preflight/verification.json'
    proof=json.loads(receipt.read_text())
    if proof['state']!='verified' or proof['encoder_forward_calls']!=4:
        raise ValueError('Four-call portability gate has not passed')
    sha=digest(PAYLOAD)
    source='''import hashlib, json, os, pathlib, platform, shutil, subprocess, sys, tarfile, time
started = time.monotonic()
work = pathlib.Path('/kaggle/working')
work.mkdir(exist_ok=True)
print(json.dumps({'host_python':sys.version, 'host_platform':platform.platform(), 'host_glibc':platform.libc_ver(),
                  'inference':'isolated pinned CPython 3.11, Torch 2.8 CPU Float32, no fitting', 'internet_required':False}))
payloads = list(pathlib.Path('/kaggle/input').rglob('ntl_r6_payload.tgz.bin'))
if len(payloads) != 1:
    raise RuntimeError('Attach exactly one private NTL R6 asset dataset')
h = hashlib.sha256()
with payloads[0].open('rb') as f:
    for block in iter(lambda:f.read(1 << 20), b''):
        h.update(block)
if h.hexdigest() != 'PAYLOAD_HASH':
    raise RuntimeError('Frozen payload SHA256 mismatch')
destination = work/'ntl_r6_runtime'
if destination.exists() or (work/'submission.parquet').exists():
    raise FileExistsError('Preserve prior execution; do not rerun in this working directory')
destination.mkdir()
with tarfile.open(payloads[0], 'r:gz') as archive:
    archive.extractall(destination, filter='data')
assets = destination/'ntl-r6-assets'
python = destination/'cpython-3.11.17-linux-x86_64-gnu/bin/python3.11'
python.chmod(0o755)
proof = json.loads((assets/'portability_verification.json').read_text())
if proof['state'] != 'verified' or proof['encoder_forward_calls'] != 4:
    raise RuntimeError('Local synthetic portability proof missing')
test_paths = [pathlib.Path('/kaggle/input/competitions/name-that-llm/test.csv'),
              pathlib.Path('/kaggle/input/name-that-llm/test.csv')]
test_paths = [p for p in test_paths if p.is_file()]
if len(test_paths) != 1:
    raise RuntimeError('Attach official Name That LLM competition input')
env = os.environ.copy()
for key in ('PYTHONHOME','PYTHONPATH'):
    env.pop(key,None)
env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           HF_HOME=str(work/'hf_offline'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
subprocess.run([str(python),str(assets/'inference_round6.py'),'test','--assets',str(assets),
                '--output',str(work/'r6_inference'),'--test',str(test_paths[0]),'--mode','cpu_float32'],
               env=env,check=True,timeout=max(1,32400-(time.monotonic()-started)))
claim=json.loads((work/'r6_inference/claim.json').read_text())
if claim['state']!='completed' or claim['model_fits']!=0 or claim['temperature_fits']!=0:
    raise RuntimeError('Frozen inference did not complete')
shutil.copyfile(work/'r6_inference/submission.parquet',work/'submission.parquet')
print(json.dumps({'state':'submission_ready','test_rows':claim['rows'],'output':'/kaggle/working/submission.parquet',
                  'encoder_forward_calls':claim['completed_forward_calls'],'total_wall_seconds':time.monotonic()-started,
                  'output_sha256':claim['output_sha256'],'payload_sha256':'PAYLOAD_HASH'}))
'''.replace('PAYLOAD_HASH',sha)
    markdown='''# NTL Calibrated R6 — frozen offline inference

Local development BA **0.370895580454404** on 120 prompt groups / 2159 rows. This is repeated development evaluation, not a Kaggle score.

One frozen candidate: 80% four-view text SVC + 20% frozen ModernBERT/Gaussian head, each temperature calibrated using three grouped inner folds. Final assets use only the fixed development subset. No fitting or protected holdout access occurs here. A self-contained CPython 3.11 runtime keeps dependencies independent of Kaggle's current image.

Use **CPU**, **Internet Off**, the official **Name That LLM** competition input and private **NTL R6 Assets 20261025** dataset. Output is `submission.parquet` with one `id,model` row per test ID, limited to 4096 test rows and 9 hours. CPU Float32 deployment passed four synthetic forward checks against Windows CUDA BF16; this does not establish identical predictions for every possible input.
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':markdown.splitlines(True)},
           {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}]
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                      'language_info':{'name':'python','version':'3.11.17'}},'nbformat':4,'nbformat_minor':5}
    target=MODULE/'notebooks/calibrated_round6_offline.ipynb'
    if target.exists(): raise FileExistsError(target)
    target.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    record={'payload_file':PAYLOAD.name,'payload_bytes':PAYLOAD.stat().st_size,'payload_sha256':sha,
            'notebook_sha256':digest(target),'portability_receipt_sha256':digest(receipt),
            'private_assets_only':True,'competition_raw_data_included':False}
    (PAYLOAD.parent/'package_receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record))


if __name__=='__main__': main()
