"""CPU-only reproduction and unapplied fix for the frozen R18 V1 failure.

No real models, inputs or GPU forward calls. Large logical tensor shapes use
one float32 storage element. The original runtime and notebook stay unchanged.
"""
import contextlib
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import time
import types
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/r18_runtime_contract.py"
OUT = ROOT / "outputs/r18/fingerprint_fix_local_20261102"
OLD = "                indexes = self.torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()"
NEW = """                if count == 0:
                    continue
                indexes = self.torch.tensor(
                    [i * (flat.numel() - 1) // max(count - 1, 1) for i in range(count)],
                    dtype=self.torch.long, device=flat.device)"""


class LogicalParameter:
    """A parameter-like fixture backed by exactly one CPU float32 element."""
    def __init__(self, n):
        self.weight = torch.zeros(1, device="cpu", dtype=torch.float32).expand(n)

    def parameters(self):
        yield self.weight

    def named_parameters(self):
        yield "weight", self.weight


def load_module(source, name):
    module = types.ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def exercise(module, n, trace):
    model = LogicalParameter(n)
    assert model.weight.untyped_storage().nbytes() == 4
    with patch.dict(os.environ, {"R18_TRACE_PATH": str(trace), "R18_REQUIRE_CUDA": "0"}), \
            patch.object(torch.cuda, "device_count", return_value=0):
        lifecycle = module.Lifecycle("artificial")
        lifecycle.before_load("artificial_parameter")
        lifecycle.loaded(model)
        first = lifecycle.fingerprint
        lifecycle.loaded(model)
        assert lifecycle.fingerprint == first
        assert first and len(first) == 64
    return {"numel": n, "backing_bytes": 4, "fingerprint_repeat_equal": True}


def main():
    started = time.monotonic()
    assert torch.get_default_dtype() == torch.float32
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count(OLD) == 1, "Frozen runtime changed; review before applying diagnosis"
    fixed_source = source.replace(OLD, NEW)
    OUT.mkdir(parents=True, exist_ok=True)
    original = load_module(source, "r18_original_diagnostic")
    fixed = load_module(fixed_source, "r18_fixed_diagnostic")
    stream = io.StringIO()
    baseline_failure = None
    with contextlib.redirect_stdout(stream):
        try:
            exercise(original, 26_214_400, OUT / "original_cpu_lifecycle.jsonl")
        except IndexError as exc:
            baseline_failure = str(exc)
    assert baseline_failure and "26214400" in baseline_failure, "Expected original bounds failure absent"
    sizes = [0, 1, 2, 63, 64, 65, 16_777_217, 26_214_400, 777_912_320]
    cases = []
    with contextlib.redirect_stdout(stream):
        for n in sizes:
            cases.append(exercise(fixed, n, OUT / "fixed_cpu_lifecycle.jsonl"))
    (OUT / "r18_runtime_contract.py").write_text(fixed_source, encoding="utf-8")
    diff = "".join(difflib.unified_diff(source.splitlines(True), fixed_source.splitlines(True),
        fromfile="a/scripts/r18_runtime_contract.py", tofile="b/scripts/r18_runtime_contract.py"))
    (OUT / "integer_sampling_fix.patch").write_text(diff, encoding="utf-8")
    (OUT / "cpu_diagnostic.log").write_text(stream.getvalue(), encoding="utf-8")
    receipt = {
        "status": "original_failed_fixed_9_cases_passed_local_only",
        "torch_version": torch.__version__, "device": "cpu", "default_dtype": "float32",
        "original_error": baseline_failure, "cases": cases,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "fixed_source_text_sha256": hashlib.sha256(fixed_source.encode()).hexdigest(),
        "wall_seconds": time.monotonic() - started,
        "real_models_loaded": 0, "model_forwards": 0, "gpu_operations": 0,
        "official_data_accesses": 0, "remote_retry": False,
        "fix_applied_to_frozen_runtime": False, "notebook_regenerated": False,
        "limit": "CPU index reproduction only; corrected real GPU preflight remains untested"
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
