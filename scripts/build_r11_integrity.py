"""Build a separate guarded notebook from the exact scored 0.651 source cells."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'external_notebooks/e651/name-that-llm-0-651-verified-reproduction.ipynb'
CELL_HASHES = [
    '4fc27dca93a92adfb2759ca65260e5285898d73fc8ccd1acd401982e5f98ed37',
    'b53e305fd7a755d138f931c55ac5f3762ed3978d73a65594b5ce2523fdf165ff',
    '219900bdc7d087f734a6a152b65b3ed515249afab1d82cad7915c0a41fd02adf',
    'f1bb8c65e926f0fde39e384cd500aa98162527492251dd63651e3f78b29232bf',
    '16f3076335718c090271e42a36fcd4efdf3cefdb5197f83a4be75947dc75ace3',
    '03a93a666c99d8d2129083be5383cc6c3b532356a67556c9c6fbe5e20e6d9f0e',
]


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Pinned source target is missing or ambiguous: ' + old[:80])
    return text.replace(old, new, 1)


def code_cell(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(keepends=True)}


def build(baseline=BASELINE):
    original = json.loads(Path(baseline).read_bytes())
    cells = [source(c) for c in original['cells'] if c['cell_type'] == 'code']
    hashes = [hashlib.sha256(c.encode()).hexdigest() for c in cells]
    if hashes != CELL_HASHES:
        raise RuntimeError('Baseline source differs from teammate V1 362577633; refusing to patch')
    patched = list(cells)
    infer = patched[2]
    begin = infer.index('COMP_P = None\n')
    end = infer.index('_bundles = ', begin)
    infer = infer[:begin] + '''COMP_P = None
if a.comp_w > 0:
    from r11_integrity import load_comp
    COMP_P, _cm = load_comp(a.comp_path, test['id'].astype(str).tolist(), LABELS, os.environ.get('R11_RUN_ID'))
    log(f"comp stack loaded ({_cm['seconds']:.0f}s); R11 member/hash/run checks passed")
''' + infer[end:]
    infer = replace_once(infer, "                log('ATLAS SKIPPED:', _bd, repr(_e))", "                raise RuntimeError('R11 required ATLAS failed: ' + str(_bd)) from _e")
    infer = replace_once(infer, "pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)", "if not np.isfinite(tot).all():\n    raise RuntimeError('R11 nonfinite final ensemble scores')\npred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)")
    patched[2] = infer

    comp = replace_once(patched[3], 'LOGIT = np.zeros((len(TE), K))', 'from r11_integrity import require, write_comp\nMEMBER_AUDIT = {}\nLOGIT = np.zeros((len(TE), K))')
    comp = replace_once(comp, '    w = W[key] if weight is None else weight\n    lp = ', '''    w = W[key] if weight is None else weight
    count = len(TE) if rows is None else int(np.sum(rows))
    require(key not in MEMBER_AUDIT, key + ': duplicate member')
    require(prob.shape == (count, K) and np.isfinite(prob).all(), key + ': invalid probability shape/values')
    require((prob >= 0).all() and (prob <= 1 + 1e-6).all() and np.allclose(prob.sum(1), 1, atol=1e-4), key + ': invalid probabilities')
    MEMBER_AUDIT[key] = {'weight': float(w), 'rows': count}
    lp = ''')
    comp = replace_once(comp, "np.save('/kaggle/working/comp_logit.npy', LOGIT)\njson.dump({'labels': LABELS, 'ids': test['id'].astype(str).tolist(), 'seconds': time.time() - T0}, open('/kaggle/working/comp_meta.json', 'w'))", "write_comp('/kaggle/working/comp_logit.npy', LOGIT, LABELS, test['id'].astype(str).tolist(),\n           time.time() - T0, MEMBER_AUDIT, int(GROUP_ROWS.sum()), os.environ.get('R11_RUN_ID'))")
    patched[3] = comp

    patched[4] = '''# Required complementary stage: fresh outputs, checked exit status, bounded run.
import time as _t
import os, sys, uuid
from r11_integrity import run_checked
os.environ['R11_RUN_ID'] = uuid.uuid4().hex
os.environ['NTL_DROP_Q35'] = '1'
_c0 = _t.time()
run_checked([sys.executable, 'comp_stack.py'], ['comp_logit.npy', 'comp_meta.json'], timeout=int(8.3 * 3600 - 600))
COMP_SEC = _t.time() - _c0
print('comp stack wall time', round(COMP_SEC))
'''
    # Parse the literal launch into argv; retain every flag, value and dynamic expression.
    line = next(line for line in cells[5].splitlines() if line.startswith('!python infer.py '))
    tokens = re.findall(r"'[^']*'|\S+", line)[2:]
    args = []
    for token in tokens:
        token = token.strip("'")
        if token.startswith('{') and token.endswith('}'):
            expression = token[1:-1]
            ast.parse(expression, mode='eval')
            args.append('str(' + expression + ')')
        else:
            args.append(repr(token))
    launch = "from r11_integrity import run_checked, validate_submission\n" + "run_checked([sys.executable, 'infer.py',\n    " + ',\n    '.join(args) + "], ['submission.parquet'],\n    timeout=int(8.3 * 3600 - COMP_SEC),\n    validator=lambda: validate_submission('submission.parquet', os.path.join(DATA, 'test.csv')))"
    patched[5] = replace_once(patched[5], line, launch)

    helper = (ROOT / 'experiments/r11_integrity.py').read_text(encoding='utf-8')
    result_cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# R11: complete-ensemble integrity\n',
        'Based on teammate ntl-e651 V1 (362577633), Public Score 0.651.\n',
        'This candidate preserves numerical operations and fails on missing required stages.\n',
        '**No new accuracy result, full GPU validation, or Kaggle submission is claimed.**\n',
        'Copy the exact baseline version and preserve its pinned inputs before any future authorized run.\n',
        'Original source attribution and authors are recorded in README.md and the unchanged baseline.\n',
    ]}, code_cell('%%writefile r11_integrity.py\n' + helper), code_cell(
        "# Invalidate previous generated outputs before any baseline setup can fail.\n"
        "from r11_integrity import clear_outputs\n"
        "clear_outputs(['submission.parquet', 'comp_logit.npy', 'comp_meta.json'])\n"
    )]
    result_cells.extend(code_cell(c) for c in patched)
    for cell in result_cells:
        if cell['cell_type'] == 'code':
            code = source(cell)
            if code.startswith('%%writefile '):
                code = code.split('\n', 1)[1]
            compile(code, '<r11-notebook-cell>', 'exec')
    return {'cells': result_cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13.15'},
        'r11': {'baseline_script_version': 362577633, 'baseline_code_sha256': CELL_HASHES,
                'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(), 'accuracy_change': 'not_evaluated'},
    }, 'nbformat': 4, 'nbformat_minor': 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--output', type=Path, default=ROOT / 'notebooks/team_e651_r11_integrity.ipynb')
    args = parser.parse_args()
    result = build(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result, ensure_ascii=False, indent=1) + '\n'
    args.output.write_text(content, encoding='utf-8', newline='\n')
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(content.encode()).hexdigest(),
                      'code_cells': sum(c['cell_type'] == 'code' for c in result['cells'])}, indent=2))


if __name__ == '__main__':
    main()
