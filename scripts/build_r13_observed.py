"""Build a local diagnostic candidate while preserving all R12 score source."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/team_e651_r12_public654.ipynb'
SOURCE_SHA256 = 'f17e8bc03c2cf1e6a93448c0481b4953c6dd43391b376f7c96468019620e6c51'
OBSERVER = ROOT / 'experiments/r13_runtime.py'
OUTPUT = ROOT / 'notebooks/team_e651_r13_observed.ipynb'


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def digest(content):
    return hashlib.sha256(content.encode() if isinstance(content, str) else content).hexdigest()


def substitute(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Pinned observer insertion point differs')
    return text.replace(old, new, 1)


def code_cell(text):
    body = text.split('\n', 1)[1] if text.startswith('%%writefile ') else text
    compile(body, '<r13-generated-cell>', 'exec')
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.splitlines(keepends=True)}


def prediction_labels(code):
    infer = next(text.split('\n', 1)[1] for text in code if text.startswith('%%writefile infer.py\n'))
    assignment = next(node for node in ast.parse(infer).body
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == 'LABELS' for target in node.targets))
    labels = ast.literal_eval(assignment.value)
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError('Frozen prediction label set differs')
    return labels


def build():
    raw = SOURCE.read_bytes()
    if digest(raw) != SOURCE_SHA256:
        raise ValueError('Frozen R12 notebook hash differs')
    nb = json.loads(raw)
    code = [source(c) for c in nb['cells'] if c['cell_type'] == 'code']
    original = list(code)
    labels = prediction_labels(original)
    helper = OBSERVER.read_text(encoding='utf-8')
    bootstrap = (
        'from ntl_runtime_observer import RuntimeTrace as _R13Trace\n'
        "_r13_trace = _R13Trace('/kaggle/working/runtime_traces',\n"
        "    mode='competition_rerun' if RERUN else 'placeholder',\n"
        f"    source_sha256='{SOURCE_SHA256}')\n"
        f'_R13_EXPECTED_LABELS = {labels!r}\n'
    )
    code[6] = substitute(code[6],
        "    subprocess.run([sys.executable, 'comp_stack.py'], check=True)",
        "    _r13_trace.run([sys.executable, 'comp_stack.py'], stage='comp_stack')")
    validation = code[6][code[6].index('    import numpy as _r12np'):code[6].index('    COMP_SEC =')]
    code[6] = substitute(code[6], validation,
        "    with _r13_trace.stage('comp_output_validation'):\n" +
        ''.join('    ' + line for line in validation.splitlines(keepends=True)))
    code[7] = substitute(code[7], '    subprocess.run(cmd, check=True)',
        "    _r13_trace.run(cmd, stage='main_inference')")
    code[7] = substitute(code[7], '    print(\'RUNNING MAIN INFERENCE\', flush=True)',
        "    _r13_trace.budget(comp_seconds=COMP_SEC, atlas_budget_seconds=max(180, ATLAS_BUDGET))\n"
        "    print('RUNNING MAIN INFERENCE', flush=True)")
    validation = code[7][code[7].index("    submission = pd.read_parquet"):code[7].index('else:\n')]
    code[7] = substitute(code[7], validation,
        "    with _r13_trace.stage('submission_validation'):\n" +
        ''.join('    ' + line for line in validation.splitlines(keepends=True)) +
        "        _r13_trace.validate_submission(ids=submission['id'].tolist(), models=submission['model'].tolist(),\n"
        "            expected_ids=pd.read_csv(os.path.join(DATA, 'test.csv'), usecols=['id'])['id'].tolist(),\n"
        "            labels=_R13_EXPECTED_LABELS)\n" +
        "    _r13_trace.finish('pipeline_validated')\n")
    code[7] += "    _r13_trace.finish('placeholder_only')\n"
    header = (
        '# R13: local runtime diagnostics candidate\n\n'
        'Based on frozen R12 V1 / 363188587, derived from Alex Chilton V4 / 363071115 '
        'and Roman Tamrazov / Dhruv Tibarewal. All original upstream credits remain in the frozen reference.\n'
        'No new accuracy or speed result. No R13 Kaggle save or submission is authorized by this build. '
        'All generated prediction scripts, input IDs, models, weights and arguments are unchanged.\n'
        'Structured runtime traces observe stages and existing progress messages. They do not imply a '
        'Kaggle score, implement a global timeout, or expose logs that Kaggle withholds. Logging overhead '
        'could affect the number of ATLAS rows completed near its time boundary.\n'
    )
    nb['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': header.splitlines(keepends=True)}]
    for index, text in enumerate(code):
        nb['cells'].append(code_cell(text))
        if index == 1:
            nb['cells'].append(code_cell('%%writefile ntl_runtime_observer.py\n' + helper))
            nb['cells'].append(code_cell(bootstrap))
    nb['metadata']['r13'] = {'source_version': 363188587, 'source_notebook_sha256': SOURCE_SHA256,
                             'observer_sha256': digest(helper), 'local_cv': None,
                             'saved_runs_budget': 0, 'formal_submissions_budget': 0}
    manifest = {
        'source_version': 363188587, 'source_notebook_sha256': SOURCE_SHA256,
        'observer_sha256': digest(helper),
        'prediction_scripts_unchanged': {
            text.splitlines()[0].removeprefix('%%writefile '): digest(text.split('\n', 1)[1])
            for text in original if text.startswith('%%writefile ')
        },
        'pinned_inputs': nb['metadata']['kaggle']['dataSources'],
        'local_training_fits': 0, 'encoder_forwards': 0, 'official_label_evaluations': 0,
        'protected_holdout_accesses': 0, 'kaggle_saves': 0, 'formal_submissions': 0,
        'accuracy_or_speed_improvement_claimed': False,
    }
    return nb, manifest


def main():
    nb, manifest = build()
    content = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
    OUTPUT.write_text(content, encoding='utf-8', newline='\n')
    manifest['candidate_sha256'] = digest(content)
    target = ROOT / 'configs/r13_observed_recipe_20261029.json'
    target.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'candidate': str(OUTPUT), 'sha256': manifest['candidate_sha256'],
                      'unchanged_prediction_scripts': list(manifest['prediction_scripts_unchanged'])}, indent=2))


if __name__ == '__main__':
    main()
