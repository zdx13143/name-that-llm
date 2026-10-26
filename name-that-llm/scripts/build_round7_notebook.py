"""Generate the single R7 Kaggle notebook from the exact Linux-verified payload."""
import json
from pathlib import Path

from inference_round7 import digest, write_json

MODULE = Path(__file__).resolve().parents[1]


def main():
    package = MODULE / "outputs/kaggle_r7_package"
    receipt = json.loads((package / "package_receipt.json").read_bytes())
    proof_path = package / "relocation_verification.json"
    proof = json.loads(proof_path.read_bytes())
    payload = package / receipt["payload_file"]
    if (proof["state"] != "verified" or proof["payload_sha256"] != receipt["payload_sha256"]
            or digest(payload) != receipt["payload_sha256"] or proof["manifest_sha256"] != receipt["manifest_sha256"]
            or proof["rows"] != 4 or proof["synthetic_max_abs_error"] > 1e-7):
        raise ValueError("Exact payload has not passed Linux compatibility verification")
    source = '''import hashlib, json, os, pathlib, platform, shutil, subprocess, sys, tarfile, time
started = time.monotonic()
work = pathlib.Path('/kaggle/working')
work.mkdir(exist_ok=True)
print(json.dumps({'host_python':sys.version,'host_platform':platform.platform(),
                  'inference':'isolated Python 3.11; frozen four-view weighted SVC; CPU; no fitting',
                  'development_gate_passed':False,'submission_reason':'explicit user request to observe public score'}))
payloads = list(pathlib.Path('/kaggle/input').rglob('ntl_r7_payload.tgz.bin'))
if len(payloads) != 1:
    raise RuntimeError('Attach exactly one private NTL R7 asset dataset')
h = hashlib.sha256()
with payloads[0].open('rb') as handle:
    for block in iter(lambda:handle.read(1<<20),b''):
        h.update(block)
if h.hexdigest() != 'PAYLOAD_HASH':
    raise ValueError('Payload hash mismatch')
destination = pathlib.Path('/tmp/ntl_r7_runtime')
if destination.exists() or (work/'submission.parquet').exists():
    raise FileExistsError('Preserve prior execution; do not rerun')
destination.mkdir()
with tarfile.open(payloads[0],'r:gz') as archive:
    if {pathlib.Path(member.name).parts[0] for member in archive.getmembers()} != {'ntl-r7-assets','cpython-3.11.17-linux-x86_64-gnu'}:
        raise ValueError('Unexpected archive roots')
    archive.extractall(destination,filter='data')
assets = destination/'ntl-r7-assets'
python = destination/'cpython-3.11.17-linux-x86_64-gnu/bin/python3.11'
test_paths = [pathlib.Path('/kaggle/input/competitions/name-that-llm/test.csv'),pathlib.Path('/kaggle/input/name-that-llm/test.csv')]
test_paths = [p for p in test_paths if p.is_file()]
if len(test_paths) != 1:
    raise RuntimeError('Attach official Name That LLM competition input')
env = os.environ.copy()
for key in ('PYTHONHOME','PYTHONPATH'):
    env.pop(key,None)
env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([str(python),str(assets/'inference_round7.py'),'test','--assets',str(assets),
                '--output',str(work/'r7_inference'),'--test',str(test_paths[0]),'--manifest-sha256','MANIFEST_HASH'],
               env=env,check=True,timeout=max(1,32400-(time.monotonic()-started)))
claim = json.loads((work/'r7_inference/claim.json').read_bytes())
if (claim['state'] != 'completed' or claim['model_fits'] != 0 or claim['old_holdout_evaluations'] != 0
        or claim['encoder_forward_calls'] != 0 or claim['development_gate_passed'] is not False):
    raise RuntimeError('Frozen inference did not complete under the requested scope')
shutil.copyfile(work/'r7_inference/submission.parquet',work/'submission.parquet')
print(json.dumps({'state':'submission_ready','test_rows':claim['rows'],'output':'/kaggle/working/submission.parquet',
                  'output_sha256':claim['output_sha256'],'payload_sha256':'PAYLOAD_HASH',
                  'model_fits':0,'encoder_forward_calls':0,'total_wall_seconds':time.monotonic()-started}))
'''.replace("PAYLOAD_HASH", receipt["payload_sha256"]).replace("MANIFEST_HASH", receipt["manifest_sha256"])
    compile(source, "r7-kaggle-notebook", "exec")
    markdown = '''# NTL Weighted External SVC R7 — one requested score observation

Frozen four-view text SVC trained on the fixed 2,159-row / 120-prompt development subset plus public external passages. Official sample weight 4, external weight 1. No training, calibration, fusion, encoder, group assignment or test-time fitting occurs in this notebook.

Development BA 0.44593837535014; equal-weight control 0.4454831932773109. The preregistered incremental gate failed. The user explicitly requested one Kaggle submission to observe the public score; this does not change the failed gate or prove improvement over the full teammate 0.601 ensemble.

Use **CPU**, **Internet Off**, the official **Name That LLM** input and private **NTL R7 Assets 20261026**. The payload contains a fixed Linux CPython 3.11 runtime and six pinned inference dependencies, with no competition raw records, protected holdout, OOF or credentials. Four artificial-text Windows/Linux checks passed for this exact payload; they test compatibility, not model accuracy. Maximum test rows: 4096; maximum time: 9 hours. Output: `submission.parquet`, columns `id,model`.

Method reference: [fixed teammate 0.601 version](https://www.kaggle.com/code/zdx13143/ntl-ensemble-600?scriptVersionId=360258830). Public external source versions and file hashes are recorded in the payload's `external_source_manifest.json`. The Hailey and Michael sources were excluded. This standalone SVC is not the complete referenced ensemble.
'''
    notebook = {"cells": [
        {"cell_type": "markdown", "id": "r7-context", "metadata": {}, "source": markdown.splitlines(True)},
        {"cell_type": "code", "id": "r7-inference", "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11.17"}}, "nbformat": 4, "nbformat_minor": 5}
    target = MODULE / "notebooks/weighted_round7_offline.ipynb"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    result = {"state": "ready_for_private_kaggle_upload", "payload_sha256": receipt["payload_sha256"],
              "payload_bytes": receipt["bytes"], "manifest_sha256": receipt["manifest_sha256"],
              "notebook_sha256": digest(target), "relocation_verification_sha256": digest(proof_path),
              "development_gate_passed": False, "kaggle_saved_runs": 0, "kaggle_formal_submissions": 0}
    write_json(package / "deployment_ready.json", result, True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
