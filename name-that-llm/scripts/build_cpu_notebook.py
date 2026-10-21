#!/usr/bin/env python3
"""Build a self-contained, offline CPU Notebook from a frozen search result."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    ast.parse(source)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


def build(run_dir, output):
    summary_path = run_dir / "search_summary.json"
    summary = json.loads(summary_path.read_text())
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    config = summary["selected"]["config"]
    selected_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    if summary["selected"]["id"] != selected_id:
        raise ValueError("Selected configuration identifier differs from frozen configuration")
    for filename in ("artifact_contract.json", "frozen_selection.json"):
        path = run_dir / filename
        if path.exists() and json.loads(path.read_text())["search_summary_sha256"] != summary_hash:
            raise ValueError(f"Frozen selection summary hash differs: {filename}")
    label_bytes = (ROOT / "config" / "labels.json").read_bytes()
    if hashlib.sha256(label_bytes).hexdigest() != summary["label_reference_sha256"]:
        raise ValueError("Label reference differs from frozen experiment")
    # The complete experiment entry is checked; old experiments use an exact
    # archived snapshot when a later reviewed guardrail fix changed that entry.
    for name, digest in summary["code_sha256"].items():
        current = ROOT / "scripts" / name
        snapshot = run_dir / "code_snapshot" / name
        if hashlib.sha256(current.read_bytes()).hexdigest() != digest:
            if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen experiment code is unavailable: {name}")
    model_sources = {}
    for name in ("model_cpu.py", "model_cpu_round2.py"):
        if name not in summary["code_sha256"]:
            continue
        source = ROOT / "scripts" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != summary["code_sha256"][name]:
            source = run_dir / "code_snapshot" / name
        model_sources[name] = source.read_text()
    selected = summary["selected"]["config"]
    labels = json.loads((ROOT / "config" / "labels.json").read_text())["labels"]
    expected = {"scikit-learn": summary["environment"]["scikit_learn"],
                "numpy": summary["environment"]["numpy"], "pyarrow": "26.0.0",
                **{name: summary["environment"][name] for name in ("scipy", "joblib")
                   if name in summary["environment"]}}
    cells = [markdown("""# Name That LLM — offline CPU inference entry

This Notebook fits the frozen TF-IDF classifier on the official training data,
then writes one `id, model` prediction per actual test row to
`/kaggle/working/submission.parquet`. It does not download external models,
packages, or data. Use Accelerator **None** and Internet **Off**, and attach the
official Name That LLM competition data. Saving or running this Notebook does
not constitute a competition submission.

The configuration was chosen using local prompt-grouped development validation.
The public test example only checks the interface. This Notebook has not been
executed on Kaggle. Runtime package versions are logged; exact reruns require
the same versions as the local experiment (see the version cell). If Kaggle's
preinstalled packages differ, attach compatible offline wheels or treat the
run as a separately versioned reproduction and record its results.
"""), code("""import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from threadpoolctl import threadpool_limits

INPUT = Path(os.environ.get('NTL_INPUT_DIR', '/kaggle/input/competitions/name-that-llm'))
OUTPUT = Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working'))
OUTPUT.mkdir(parents=True, exist_ok=True)
""" + f"EXPECTED_VERSIONS = {expected!r}\nEXPECTED_PYTHON = {summary['environment']['python']!r}\n" + """ACTUAL_VERSIONS = {key: importlib.metadata.version(key) for key in EXPECTED_VERSIONS}
VERSION_MISMATCH = {key: {'local': value, 'runtime': ACTUAL_VERSIONS[key]}
                    for key, value in EXPECTED_VERSIONS.items()
                    if ACTUAL_VERSIONS[key] != value}
if platform.python_version() != EXPECTED_PYTHON:
    VERSION_MISMATCH['python'] = {'local': EXPECTED_PYTHON, 'runtime': platform.python_version()}
print({'python': platform.python_version(), 'packages': ACTUAL_VERSIONS,
       'version_mismatches': VERSION_MISMATCH})
"""), code(model_sources["model_cpu.py"])]
    if "model_cpu_round2.py" in model_sources:
        extension = model_sources["model_cpu_round2.py"].replace("import model_cpu as original",
            "import types\noriginal = types.SimpleNamespace(make_features=make_features, make_classifier=make_classifier)")
        cells.append(code(extension))
    cells.append(code(f"CONFIG = {selected!r}\nLABELS = {labels!r}\nEXPECTED_TRAIN_SHA256 = {summary['train_sha256']!r}\n" + """
train_path = INPUT / 'train.csv'
test_path = INPUT / 'test.csv'
if file_hash(train_path) != EXPECTED_TRAIN_SHA256:
    raise ValueError('Training data differs from the audited snapshot; re-audit before fitting.')
train = read_csv(train_path)
test = read_csv(test_path)
if not train or not test:
    raise ValueError('Official train and test files must contain rows.')
if set(row['model'] for row in train) != set(LABELS):
    raise ValueError('Actual training labels differ from the frozen reference.')
if any(not row['text'].strip() for row in train + test):
    raise ValueError('Blank text requires a new data audit.')
started = time.monotonic()
with threadpool_limits(limits=2):
    pipeline = make_pipeline(CONFIG)
    pipeline.fit([row['text'] for row in train], [LABELS.index(row['model']) for row in train])
    predictions = [LABELS[index] for index in pipeline.predict([row['text'] for row in test])]
checks = checked_submission(test, predictions, LABELS, OUTPUT / 'submission.parquet')
runtime = {'config': CONFIG, 'train_sha256': EXPECTED_TRAIN_SHA256,
           'train_rows': len(train), 'test_rows': len(test),
           'elapsed_seconds': time.monotonic() - started,
           'python': platform.python_version(),
           'full_training_classifier_fits': 2 if CONFIG.get('classifier') == 'score_fusion' else 1,
           'versions': ACTUAL_VERSIONS, 'version_mismatches': VERSION_MISMATCH,
           'output_validation': checks, 'formal_submission': False}
(OUTPUT / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\\n')
print(runtime)
"""))
    for index, cell in enumerate(cells):
        cell["id"] = f"ntl-cpu-{index}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": summary["environment"]["python"]},
                    "ntl_provenance": {"selected_config_id": selected_id,
                        "search_summary_sha256": summary_hash,
                        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "model_source_sha256": {name: summary["code_sha256"][name] for name in model_sources}}},
                "nbformat": 4, "nbformat_minor": 5}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "cpu_round1")
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks" / "cpu_offline.ipynb")
    args = parser.parse_args()
    build(args.run_dir, args.output)
