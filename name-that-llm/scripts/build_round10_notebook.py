"""Generate the single R10 Kaggle notebook from the exact Linux-verified payload."""
import json
from pathlib import Path

from inference_round10 import digest, write_json

MODULE = Path(__file__).resolve().parents[1]


def main():
    package = MODULE / "outputs/kaggle_r10_package"
    receipt = json.loads((package / "package_receipt.json").read_bytes())
    proof_path = package / "relocation_verification.json"
    proof = json.loads(proof_path.read_bytes())
    payload = package / receipt["payload_file"]
    if (proof["state"] != "verified" or proof["payload_sha256"] != receipt["payload_sha256"]
            or digest(payload) != receipt["payload_sha256"] or proof["manifest_sha256"] != receipt["manifest_sha256"]
            or proof["rows"] != 80 or proof["synthetic_max_abs_error"] > 1e-7):
        raise ValueError("Exact payload has not passed Linux compatibility verification")
    stress = json.loads((package / "grouping_stress_verification.json").read_bytes())
    if stress["state"] != "verified" or stress["rows"] != 4096 or stress["manifest_sha256"] != receipt["manifest_sha256"]:
        raise ValueError("Maximum-row grouping resource check is incomplete")
    source = '''import hashlib, json, os, pathlib, platform, shutil, subprocess, sys, tarfile, time
started = time.monotonic()
work = pathlib.Path('/kaggle/working')
work.mkdir(exist_ok=True)
print(json.dumps({'host_python':sys.version,'host_platform':platform.platform(),
                  'inference':'isolated Python 3.11; frozen topic-word SVC; CPU; whole-batch text-only grouping; no supervised fitting',
                  'development_gate_passed':True,'submission_reason':'explicit user request to observe public score'}))
payloads = list(pathlib.Path('/kaggle/input').rglob('ntl_r10_payload.tgz.bin'))
if len(payloads) != 1:
    raise RuntimeError('Attach exactly one private NTL R10 asset dataset')
h = hashlib.sha256()
with payloads[0].open('rb') as handle:
    for block in iter(lambda:handle.read(1<<20),b''):
        h.update(block)
if h.hexdigest() != 'PAYLOAD_HASH':
    raise ValueError('Payload hash mismatch')
destination = pathlib.Path('/tmp/ntl_r10_runtime')
if destination.exists() or (work/'submission.parquet').exists():
    raise FileExistsError('Preserve prior execution; do not rerun')
destination.mkdir()
with tarfile.open(payloads[0],'r:gz') as archive:
    if {pathlib.Path(member.name).parts[0] for member in archive.getmembers()} != {'ntl-r10-assets','cpython-3.11.17-linux-x86_64-gnu'}:
        raise ValueError('Unexpected archive roots')
    archive.extractall(destination,filter='data')
assets = destination/'ntl-r10-assets'
python = destination/'cpython-3.11.17-linux-x86_64-gnu/bin/python3.11'
test_paths = [pathlib.Path('/kaggle/input/competitions/name-that-llm/test.csv'),pathlib.Path('/kaggle/input/name-that-llm/test.csv')]
test_paths = [p for p in test_paths if p.is_file()]
if len(test_paths) != 1:
    raise RuntimeError('Attach official Name That LLM competition input')
env = os.environ.copy()
for key in ('PYTHONHOME','PYTHONPATH'):
    env.pop(key,None)
env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([str(python),str(assets/'inference_round10.py'),'test','--assets',str(assets),
                '--output',str(work/'r10_inference'),'--test',str(test_paths[0]),'--manifest-sha256','MANIFEST_HASH'],
               env=env,check=True,timeout=max(1,32400-(time.monotonic()-started)))
claim = json.loads((work/'r10_inference/claim.json').read_bytes())
if (claim['state'] != 'completed' or claim['model_fits'] != 0 or claim['old_holdout_evaluations'] != 0
        or claim['encoder_forward_calls'] != 0 or claim['development_gate_passed'] is not True
        or claim['completed_prediction_calls'] != 1 or claim['margin_correction'] is not True):
    raise RuntimeError('Frozen inference did not complete under the requested scope')
shutil.copyfile(work/'r10_inference/submission.parquet',work/'submission.parquet')
print(json.dumps({'state':'submission_ready','test_rows':claim['rows'],'output':'/kaggle/working/submission.parquet',
                  'output_sha256':claim['output_sha256'],'payload_sha256':'PAYLOAD_HASH',
                  'model_fits':0,'encoder_forward_calls':0,'total_wall_seconds':time.monotonic()-started}))
'''.replace("PAYLOAD_HASH", receipt["payload_sha256"]).replace("MANIFEST_HASH", receipt["manifest_sha256"])
    compile(source, "r10-kaggle-notebook", "exec")
    markdown = '''# NTL Topic-Word SVC R10 — one requested score observation

Frozen four-view SVC trained on the original 2,159-row / 120-prompt development subset plus 38,724 public external passages. Official/external weights 4/1. Word features subtract half the group mean; the fixed new correction subtracts half the group mean of class margins. Only groups of size 2..18 are corrected.

The entire unlabeled test batch is grouped once using word TF-IDF, min_df=2/max_df=.05, capped average linkage, cap18/threshold.15, stable IDs. This is transductive inference: predictions depend on batch composition. No supervised model fitting, encoder or label-based test adaptation. Fewer than 40 rows use singleton fallback.

Development BA **0.47740429505135384**, R9 **0.4542794895736073**, R7 **0.44593837535014**. Five of five folds improved over R9; the fixed R10 gate passed. The original 120 prompts have been used repeatedly for development; this is not a blind holdout or proof of improvement to the full teammate 0.651 ensemble. R9's historic gain gate remains failed.

Use **CPU**, **Internet Off**, official **Name That LLM** input and private **NTL R10 Assets 20261028**. The exact payload passed Windows/Linux checks on 80 artificial texts, reverse order and singleton fallback, plus a 4096-row synthetic grouping resource check. Those are compatibility/resource checks, not accuracy evidence. Fixed Linux Python3.11 and six pinned dependencies; no raw competition records, protected holdout, OOF or credentials. Maximum 4096 test rows and 9 hours. Output `submission.parquet`, columns `id,model`.

Grouping reference: [fixed teammate e651](https://www.kaggle.com/code/zdx13143/ntl-e651?scriptVersionId=362577633). `external_source_manifest.json` is the original 14-source audit catalog, including two reviewed but excluded Hailey/Michael sources; it is not a list of only the sources used for training. This is a standalone SVC component.
'''
    notebook = {"cells": [
        {"cell_type": "markdown", "id": "r10-context", "metadata": {}, "source": markdown.splitlines(True)},
        {"cell_type": "code", "id": "r10-inference", "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11.17"}}, "nbformat": 4, "nbformat_minor": 5}
    target = MODULE / "notebooks/topic_margin_round10_offline.ipynb"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    result = {"state": "ready_for_private_kaggle_upload", "payload_sha256": receipt["payload_sha256"],
              "payload_bytes": receipt["bytes"], "manifest_sha256": receipt["manifest_sha256"],
              "notebook_sha256": digest(target), "relocation_verification_sha256": digest(proof_path),
              "grouping_stress_verification_sha256": digest(package / "grouping_stress_verification.json"),
              "development_gate_passed": True, "kaggle_saved_runs": 0, "kaggle_formal_submissions": 0}
    write_json(package / "deployment_ready.json", result, True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
