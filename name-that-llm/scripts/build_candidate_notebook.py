#!/usr/bin/env python3
"""Bundle the frozen 75/25 development candidate; building does not train or submit."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from build_cpu_notebook import code, markdown

ROOT = Path(__file__).resolve().parents[1]
PINNED = {
    "scripts/model_cpu.py": "fcfe8df6f52a077cdb5c8e0cddc7b917ef3354e2280854bb075059f0749cbcdc",
    "scripts/model_cpu_round2.py": "5c1b2d1da22082d92137d25bf6a399428b9c0b98b944ad96fd3903a24ee509d1",
    "scripts/compare_team_cpu.py": "bac5a5f1a282a0a5d0a04cd5aaa92a5bced0cd17b71bb3e511ae0084baf8637e",
    "scripts/compare_prompt_centering.py": "ab41aebb1db7a3ad3501c9a7a45bcd47e190aa3cfd33277d5a0c61bc1fe7ec92",
    "config/team_dev_20261021.json": "c9f541de654da331f927bf951674ed6675859c2220d6d038ce9c40541f63c688",
    "config/prompt_centering_20261021.json": "3e5e5e597e596482e38148bba89503bbef71a561fe4d590675926fa13ebcbcd3",
    "config/labels.json": "d8c9dcad5979236fb0e5f5504dd2e53c676123bf75cbf06faa13bc1e86524b9d",
}
SUMMARY_SHA256 = "b788bda9970a3a329d4111c75600eb8edd035caa6914df7dcd5e8353262675a5"
CONTRACT_SHA256 = "9b0899a59dc21a0baed7e90fe5b1bd1a1f1afd791e07ad4ebd81644f83417e2b"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_functions(path, names):
    """Extract exact function source without importing an experiment entry point."""
    source = Path(path).read_text()
    tree = ast.parse(source)
    found = {node.name: ast.get_source_segment(source, node)
             for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    if set(found) != set(names):
        raise ValueError("Missing frozen helper functions")
    return "\n\n".join(found[name] for name in names) + "\n"


RUNTIME = r'''
def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_inputs():
    train_path, test_path = INPUT / 'train.csv', INPUT / 'test.csv'
    if file_hash(train_path) != EXPECTED_TRAIN_SHA256:
        raise ValueError('Training hash differs from the frozen official snapshot')
    train, test = read_csv(train_path), read_csv(test_path)
    if not train or not test:
        raise ValueError('Train and test must both contain rows')
    # Conservative allowance for several dense float64 distance/similarity copies.
    # Refuse oversized batches rather than silently changing the frozen clustering.
    if 64 * len(test) ** 2 > 1073741824:
        raise MemoryError('Test clustering admission budget exceeded (64*n*n > 1 GiB)')
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError('Training row count differs from the audited snapshot')
    if any(set(row) != {'id', 'prompt_id', 'text', 'model'} for row in train):
        raise ValueError('Training schema differs from the audited snapshot')
    if any(set(row) != {'id', 'text'} for row in test):
        raise ValueError('Test schema must be exactly id and text')
    for name, rows in (('train', train), ('test', test)):
        if any(not isinstance(row['id'], str) or not row['id'].strip()
               or not isinstance(row['text'], str) or not row['text'].strip() for row in rows):
            raise ValueError(name + ' IDs and texts must be nonblank strings')
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError(name + ' IDs must be unique')
    if len(LABELS) != 18 or len(set(LABELS)) != 18 or set(row['model'] for row in train) != set(LABELS):
        raise ValueError('Training classes differ from the frozen 18-label reference')
    if any(not isinstance(row['prompt_id'], str) or not row['prompt_id'].strip() for row in train):
        raise ValueError('Training prompts must be nonblank')
    if len({row['prompt_id'] for row in train}) != EXPECTED_TRAIN_GROUPS:
        raise ValueError('Training prompt count differs from audited snapshot')
    return train, test, {'train_sha256': file_hash(train_path), 'test_sha256': file_hash(test_path)}


def assert_classes(model):
    if not np.array_equal(model.classes_, np.arange(len(LABELS))):
        raise ValueError('Classifier class order differs from canonical label order')


def run_candidate():
    train, test, input_hashes = validate_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifacts = ('submission.parquet', 'probabilities.npz', 'runtime.json')
    if any((OUTPUT / name).exists() for name in artifacts):
        raise FileExistsError('Candidate output already exists; refusing overwrite or silent rerun')
    claim_path = OUTPUT / 'candidate_attempt.json'
    claim = {'state': 'running', 'candidate': 'blend_partial_0.25', 'max_classifier_fits': 2,
             'attempted_fits': 0, 'completed_fits': 0, 'formal_submission': False,
             'input_hashes': input_hashes, 'provenance': PROVENANCE}
    # Exclusive creation is the persistent admission guard; never delete/reset a claim.
    with claim_path.open('x', encoding='utf-8') as handle:
        json.dump(claim, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    started, fit_records = time.monotonic(), []
    try:
        texts = np.asarray([row['text'] for row in train])
        test_texts = np.asarray([row['text'] for row in test])
        ids = np.asarray([row['id'] for row in test])
        truth = np.asarray([LABELS.index(row['model']) for row in train])
        groups = np.asarray([row['prompt_id'] for row in train])
        def guarded_fit(model, x, name):
            if claim['attempted_fits'] >= 2:
                raise RuntimeError('Exactly two classifier fits are authorized')
            claim['attempted_fits'] += 1
            claim['current_model'] = name
            atomic_json(claim_path, claim)
            fit_started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x, truth)
            claim['completed_fits'] += 1
            atomic_json(claim_path, claim)
            convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            fit_records.append({'model': name, 'seconds': time.monotonic() - fit_started,
                                'convergence_warnings': convergence})
            if convergence:
                raise RuntimeError('Convergence warning; do not retune or retry implicitly')
            assert_classes(model)
            return model
        with threadpool_limits(limits=2):
            original_model = guarded_fit(make_pipeline(ORIGINAL_CONFIG, seed=20261018), texts, 'original_style')
            original_proba = canonical_probabilities(original_model.predict_proba(test_texts), len(test), len(LABELS))
            del original_model
            gc.collect()
            train_x, test_x = feature_pair(texts, test_texts, CENTER_CONFIG)
            train_x = soft_center(train_x, groups, 0.5, CENTER_CONFIG['max_centered_csr_bytes'])
            partial_model = guarded_fit(LogisticRegression(C=16.0, class_weight='balanced',
                            max_iter=3000, random_state=20261021), train_x, 'partial')
            del train_x
            gc.collect()
            test_groups = cluster_batch(test_texts, ids, 0.65, CENTER_CONFIG['cluster_max_features'])
            test_x = soft_center(test_x, test_groups, 0.5, CENTER_CONFIG['max_centered_csr_bytes'])
            partial_proba = canonical_probabilities(partial_model.predict_proba(test_x), len(test), len(LABELS))
            blend = canonical_probabilities(0.75 * original_proba + 0.25 * partial_proba, len(test), len(LABELS))
            del partial_model, test_x
        predictions = [LABELS[i] for i in blend.argmax(axis=1)]
        checks = checked_submission(test, predictions, LABELS, OUTPUT / 'submission.parquet')
        np.savez_compressed(OUTPUT / 'probabilities.npz', ids=ids, labels=np.asarray(LABELS),
                            original_style=original_proba, partial=partial_proba, blend=blend)
        runtime = {'candidate': 'blend_partial_0.25', 'weights': {'original_style': 0.75, 'partial': 0.25},
            'original_config': ORIGINAL_CONFIG, 'centering_config': CENTER_CONFIG,
            'train_rows': len(train), 'train_prompts': len(set(groups)), 'test_rows': len(test),
            'public_test_example_rows': 3, 'is_three_row_interface_example': len(test) == 3,
            'test_size_is_not_assumed_hidden_size': True,
            'test_cluster_sizes': sorted(np.unique(test_groups, return_counts=True)[1].tolist()),
            'elapsed_seconds': time.monotonic() - started, 'fit_records': fit_records,
            'full_training_classifier_fits': claim['completed_fits'], 'attempted_fits': claim['attempted_fits'],
            'input_hashes': input_hashes, 'provenance': PROVENANCE,
            'python': platform.python_version(), 'versions': ACTUAL_VERSIONS,
            'version_mismatches': VERSION_MISMATCH, 'exact_environment_match': not VERSION_MISMATCH,
            'clustering_keyword': CLUSTERING_KEYWORD, 'output_validation': checks,
            'clustering_admission_bytes': 64 * len(test) ** 2,
            'probabilities_sha256': file_hash(OUTPUT / 'probabilities.npz'),
            'final_holdout_evaluations': 0, 'formal_submission': False,
            'interpretation': 'Full-data fit and interface prediction; no new validation or leaderboard score'}
        atomic_json(OUTPUT / 'runtime.json', runtime)
        claim.update(state='completed', elapsed_seconds=runtime['elapsed_seconds'],
                     runtime_sha256=file_hash(OUTPUT / 'runtime.json'), output_validation=checks)
        atomic_json(claim_path, claim)
        print(json.dumps({key: runtime[key] for key in ('candidate', 'train_rows', 'test_rows',
              'full_training_classifier_fits', 'elapsed_seconds', 'version_mismatches',
              'output_validation', 'formal_submission')}, indent=2))
        return runtime
    except BaseException as error:
        claim.update(state='failed', error=type(error).__name__ + ': ' + str(error),
                     elapsed_seconds=time.monotonic() - started, fit_records=fit_records)
        atomic_json(claim_path, claim)
        raise
'''


def build(output=ROOT / 'notebooks/candidate_blend_offline.ipynb', run_dir=None):
    output = Path(output)
    for relative, expected in PINNED.items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Frozen source/config drift: ' + relative)
    # Clean clones lack local experiment outputs. Fixed provenance below permits
    # rebuilding from the tracked source; any available evidence must still match.
    explicit_run_dir = run_dir is not None
    run_dir = Path(run_dir) if explicit_run_dir else ROOT / 'outputs/prompt_centering_20261021'
    for filename, expected in (('summary.json', SUMMARY_SHA256), ('contract.json', CONTRACT_SHA256)):
        path = run_dir / filename
        if (explicit_run_dir or path.exists()) and digest(path) != expected:
            raise ValueError('Frozen experiment artifact drift: ' + filename)
    contract = {'python': '3.13.2', 'packages': {'numpy': '2.5.4', 'scipy': '1.18.1',
        'scikit-learn': '1.9.1', 'pandas': '3.0.6', 'joblib': '1.6.0', 'threadpoolctl': '3.7.0'}}
    team = json.loads((ROOT / 'config/team_dev_20261021.json').read_text())
    center = json.loads((ROOT / 'config/prompt_centering_20261021.json').read_text())
    labels = json.loads((ROOT / 'config/labels.json').read_text())['labels']
    provenance = {'candidate': 'blend_partial_0.25', 'summary_sha256': SUMMARY_SHA256,
        'contract_file_sha256': CONTRACT_SHA256, 'source_sha256': PINNED,
        'generator_sha256': digest(__file__), 'training_hash': center['train_sha256'],
        'selection_scope': 'adaptive 120-prompt development; no confirmed improvement over previous best'}
    setup = '''import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import gc
import importlib.metadata
import inspect
import json
import platform
import sys
import time
import types
import warnings
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

if 'NTL_INPUT_DIR' in os.environ:
    INPUT = Path(os.environ['NTL_INPUT_DIR'])
else:
    INPUT = Path('/kaggle/input/competitions/name-that-llm')
    fallback = Path('/kaggle/input/name-that-llm')
    if not INPUT.exists() and fallback.exists():
        INPUT = fallback
OUTPUT = Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working'))
CLUSTERING_KEYWORD = ('metric' if 'metric' in inspect.signature(_AgglomerativeClustering).parameters else 'affinity')
def AgglomerativeClustering(**kwargs):
    # Older sklearn renamed affinity to metric; the precomputed distances are identical.
    if CLUSTERING_KEYWORD == 'affinity':
        kwargs['affinity'] = kwargs.pop('metric')
    return _AgglomerativeClustering(**kwargs)
'''
    setup += f'EXPECTED_VERSIONS = {dict(contract["packages"], pyarrow="26.0.0")!r}\n'
    setup += f'EXPECTED_PYTHON = {contract["python"]!r}\n'
    setup += '''ACTUAL_VERSIONS = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
VERSION_MISMATCH = {name: {'local': value, 'runtime': ACTUAL_VERSIONS[name]}
                    for name, value in EXPECTED_VERSIONS.items() if ACTUAL_VERSIONS[name] != value}
if platform.python_version() != EXPECTED_PYTHON:
    VERSION_MISMATCH['python'] = {'local': EXPECTED_PYTHON, 'runtime': platform.python_version()}
print({'python': platform.python_version(), 'versions': ACTUAL_VERSIONS, 'version_mismatches': VERSION_MISMATCH})
'''
    setup += f'ORIGINAL_CONFIG = {team["original_config"]!r}\nCENTER_CONFIG = {center!r}\nLABELS = {labels!r}\n'
    setup += f'EXPECTED_TRAIN_SHA256 = {center["train_sha256"]!r}\nEXPECTED_TRAIN_ROWS = 2699\nEXPECTED_TRAIN_GROUPS = 150\nPROVENANCE = {provenance!r}\n'
    embedded = 'MODEL_SOURCES = ' + repr({name: (ROOT / 'scripts' / name).read_text()
                    for name in ('model_cpu.py', 'model_cpu_round2.py')}) + '\n'
    embedded += '''original_cpu = types.ModuleType('model_cpu')
exec(compile(MODEL_SOURCES['model_cpu.py'], '<frozen model_cpu.py>', 'exec'), original_cpu.__dict__)
_prior_model_cpu = sys.modules.get('model_cpu')
sys.modules['model_cpu'] = original_cpu
try:
    style_cpu = types.ModuleType('model_cpu_round2')
    exec(compile(MODEL_SOURCES['model_cpu_round2.py'], '<frozen model_cpu_round2.py>', 'exec'), style_cpu.__dict__)
finally:
    if _prior_model_cpu is None:
        del sys.modules['model_cpu']
    else:
        sys.modules['model_cpu'] = _prior_model_cpu
make_pipeline = style_cpu.make_pipeline
read_csv = original_cpu.read_csv
file_hash = original_cpu.file_hash
checked_submission = original_cpu.checked_submission
'''
    helpers = extract_functions(ROOT / 'scripts/compare_prompt_centering.py', ('soft_center', 'cluster_batch', 'feature_pair'))
    helpers += extract_functions(ROOT / 'scripts/compare_team_cpu.py', ('validate_probabilities', 'canonical_probabilities'))
    cells = [markdown('''# Name That LLM — frozen 75/25 CPU candidate

Use **CPU / Accelerator None**, **Internet Off**, and attach official competition data.
The notebook fits two classifiers once on all 2,699 official training rows and
writes `/kaggle/working/submission.parquet` with `id, model` for every actual test row.
It downloads nothing. The three-row public test is only an interface example.
Full-data fitting is for submission inference; the previously used holdout is
never evaluated. Selection is frozen: 75% original style, 25% partial centering.
The development gain over the previous best was uncertain; there is no new
leaderboard result. Saving/running a notebook is not a competition submission.

Runtime versions, two-fit cost, probabilities, output hashes, and a durable
exclusive attempt claim are recorded. An existing attempt or output is never
silently overwritten; a failed attempt requires review before any new run.
The full test batch is clustered using text alone; no test prompt IDs, known
class counts, or per-class assignment rules are used. This step uses quadratic
memory in test row count: admission requires `64 * test_rows**2 <= 1 GiB`
(4,000 rows pass). Training centering retains its fixed 700 MiB allocation
admission bound. These admission estimates are not total-process RAM limits.
Package mismatches are explicit, not hidden behind
an exact-reproduction claim.
'''), code(setup), code(embedded), code(helpers), code(RUNTIME), code('runtime = run_candidate()\n')]
    for index, cell in enumerate(cells):
        cell['id'] = f'ntl-candidate-{index}'
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': contract['python']},
        'ntl_provenance': provenance}, 'nbformat': 4, 'nbformat_minor': 5}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + '\n')
    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'notebooks/candidate_blend_offline.ipynb')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    build(args.output, args.run_dir)
    print(args.output)
