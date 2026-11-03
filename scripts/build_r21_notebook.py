"""Create only local, unexecuted notebook and kernel metadata."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = '''import json, os, shutil, subprocess, sys
from pathlib import Path

def find_one(name):
    hits = list(Path('/kaggle/input').rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'input {name}: expected one match, found {len(hits)}')
    return hits[0]

working = Path('/kaggle/working')
scripts = ['r19_overlap_audit.py', 'r19_cpu_cv.py', 'r19_svc_component.py',
           'r21_members.py', 'r21_sinkhorn.py', 'r21_scorebank_cv.py']
sources = [f'{version}_{member}.py.txt' for version in ('r16', 'areeb')
           for member in ('extra', 'tfidf', 'svc4', 'distort', 'blend', 'sinkhorn')]
for name in scripts:
    shutil.copy2(find_one(name), working / name)
source_dir = working / 'r21_source'
source_dir.mkdir(exist_ok=True)
for name in sources:
    shutil.copy2(find_one(name), source_dir / name)
dev = find_one('development.json')
baseline = find_one('nlm_pool_v5.parquet')
extra = find_one('nlm_matched_clean.parquet')
opened = find_one('ntl_open_pool.parquet')
if baseline.parent != extra.parent:
    raise RuntimeError('matched-clean input files have different parents')
os.environ.update({'NTL_DEV': str(dev), 'NTL_BASE_DIR': str(baseline.parent),
                   'NTL_OPEN_DIR': str(opened.parent), 'NTL_R21_OUT': str(working / 'r21_run'),
                   'NTL_ALLOW_ENV_VERSIONS': '1', 'NTL_R21_LIMIT_SECONDS': '32100',
                   'NTL_R21_MIN_AVAILABLE_GIB': '24', 'NTL_R21_MAX_RSS_GIB': '28'})
run_dir = working / 'r21_run'
output = working / 'r21_out'
output.mkdir(exist_ok=False)
try:
    completed = subprocess.run([sys.executable, str(working / 'r21_scorebank_cv.py'), '--run'],
                               cwd=working, capture_output=True, text=True)
finally:
    for name in ('result.json', 'scores.npz', 'receipt.json', 'claim.json', 'progress.json', 'stages.jsonl', 'worker.log'):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, output / name)
    if (run_dir / 'parts').is_dir():
        shutil.copytree(run_dir / 'parts', output / 'parts')
receipt = output / 'receipt.json'
print(receipt.read_text(encoding='utf-8') if receipt.is_file() else
      json.dumps({'status': 'failed', 'reason': 'runner stopped before receipt'}))
if completed.returncode:
    raise RuntimeError('R21 runner failed; inspect worker.log and receipt')
'''


def main():
    notebook = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {},
                           'outputs': [], 'source': CODE.splitlines(keepends=True)}],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    folder = ROOT / 'notebooks/r21'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'r21_scorebank_kaggle.ipynb').write_text(text, encoding='utf-8')
    metadata = {'id': 'zephyrpong/ntl-r21-scorebank', 'title': 'NTL R21 scorebank',
                'code_file': 'r21_scorebank_kaggle.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': False, 'enable_internet': False,
                'dataset_sources': ['zephyrpong/ntl-r21-scorebank-package',
                                    'crtproduction/name-that-llm-open-pool/11',
                                    'dhruvtibarewal/nlm-matched-clean/3'],
                'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
