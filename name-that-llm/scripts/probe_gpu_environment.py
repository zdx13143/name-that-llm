#!/usr/bin/env python3
"""Read host/CUDA capabilities; optionally run a tiny, explicit CUDA smoke test.

Uses only the Python standard library unless Torch is already installed. The
default mode creates no tensors, changes no packages and makes no network calls.
This report describes the machine running this script, never another computer.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any


def error_record(context: str, error: Exception) -> dict[str, str]:
    """Do not serialize exception messages, which may contain private paths."""
    return {"context": context, "exception_type": type(error).__name__}


def physical_memory_bytes() -> int:
    """Return physical host RAM, not free RAM or a process/container allowance."""
    system = platform.system()
    if system == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        function.argtypes = [ctypes.POINTER(MemoryStatus)]
        function.restype = ctypes.c_int
        if not function(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        value = int(status.total_physical)
    elif system == "Darwin":
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], check=True,
            capture_output=True, text=True, timeout=5,
        )
        value = int(result.stdout.strip())
    else:
        value = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    if value <= 0:
        raise ValueError("Physical RAM was not positive")
    return value


def driver_metadata() -> dict[str, Any]:
    """Read driver versions only; do not collect device UUIDs or process lists."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "not_found", "driver_versions": None}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not versions or any(re.fullmatch(r"\d+(?:\.\d+)+", version) is None for version in versions):
            raise ValueError("Unexpected driver version format")
        return {"status": "reported", "driver_versions": sorted(set(versions))}
    except Exception as error:
        return {"status": "unknown", "driver_versions": None, "error": error_record("nvidia_smi_driver", error)}


def cuda_smoke(torch: Any) -> dict[str, Any]:
    """Small integer-valued float32 matmul; reference is calculated on CPU."""
    result: dict[str, Any] = {
        "status": "error", "device_index": 0, "dtype": "float32",
        "matrix_shape": [32, 32], "absolute_tolerance": 1e-5,
        "max_absolute_error": None, "matmul_seconds": None,
        "allocated_before_bytes": None, "allocated_after_bytes": None,
        "peak_allocated_bytes": None,
    }
    try:
        with torch.inference_mode():
            values = torch.arange(32 * 32, dtype=torch.float32, device="cpu").reshape(32, 32)
            a = values.remainder(7) - 3
            b = (values.remainder(5) - 2).T.contiguous()
            expected = a @ b
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            result["allocated_before_bytes"] = int(torch.cuda.memory_allocated(0))
            gpu_a, gpu_b = a.to("cuda:0"), b.to("cuda:0")
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            actual = gpu_a @ gpu_b
            torch.cuda.synchronize(0)
            result["matmul_seconds"] = time.perf_counter() - started
            difference = float((actual.cpu() - expected).abs().max().item())
            result["max_absolute_error"] = difference if math.isfinite(difference) else None
            result["allocated_after_bytes"] = int(torch.cuda.memory_allocated(0))
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            result["status"] = "passed" if math.isfinite(difference) and difference <= 1e-5 else "failed"
    except Exception as error:
        result["error"] = error_record("cuda_smoke", error)
    return result


def collect_report(*, run_cuda_smoke: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "current_host_only",
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "host": {
            "system": platform.system(), "release": platform.release(),
            "architecture": platform.machine(), "cpu_description": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(), "physical_ram_bytes": None,
        },
        "torch": {"distribution_version": None, "import_status": "unknown", "runtime_version": None},
        "optional_package_versions": {},
        "nvidia_driver": driver_metadata(),
        "cuda": {"build_version": None, "available": None, "device_count": None, "devices": []},
        "smoke": {"status": "not_requested"},
        "errors": [],
        "limitations": [
            "No model training, inference, download, package installation or network access.",
            "Device enumeration and a tiny optional matmul do not establish model fit, speed or accuracy.",
            "Reported host RAM is physical capacity, not available RAM or a container limit.",
        ],
    }
    try:
        report["host"]["physical_ram_bytes"] = physical_memory_bytes()
    except Exception as error:
        report["errors"].append(error_record("physical_ram", error))
    try:
        report["torch"]["distribution_version"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as error:
        report["errors"].append(error_record("torch_distribution_version", error))
    for package in ("transformers", "accelerate", "bitsandbytes", "peft"):
        try:
            report["optional_package_versions"][package] = {"status": "installed", "version": importlib.metadata.version(package)}
        except importlib.metadata.PackageNotFoundError:
            report["optional_package_versions"][package] = {"status": "not_installed", "version": None}
        except Exception as error:
            report["optional_package_versions"][package] = {"status": "unknown", "version": None}
            report["errors"].append(error_record("optional_package_version_" + package, error))
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        report["torch"]["import_status"] = "not_installed" if error.name == "torch" else "error"
        report["errors"].append(error_record("torch_import", error))
        if run_cuda_smoke:
            report["smoke"] = {"status": "unavailable", "reason": "torch_not_importable"}
        return report
    except Exception as error:
        report["torch"]["import_status"] = "error"
        report["errors"].append(error_record("torch_import", error))
        if run_cuda_smoke:
            report["smoke"] = {"status": "unavailable", "reason": "torch_not_importable"}
        return report
    report["torch"]["import_status"] = "imported"
    try:
        report["torch"]["runtime_version"] = str(torch.__version__)
    except Exception as error:
        report["errors"].append(error_record("torch_runtime_version", error))
    try:
        report["cuda"]["build_version"] = torch.version.cuda
    except Exception as error:
        report["errors"].append(error_record("torch_cuda_build_version", error))
    try:
        report["cuda"]["available"] = bool(torch.cuda.is_available())
    except Exception as error:
        report["errors"].append(error_record("torch_cuda_availability", error))
    try:
        count = int(torch.cuda.device_count())
        if count < 0:
            raise ValueError("Negative CUDA device count")
        report["cuda"]["device_count"] = count
    except Exception as error:
        report["errors"].append(error_record("torch_cuda_device_count", error))
    if report["cuda"]["available"] is True and report["cuda"]["device_count"] is not None:
        for index in range(report["cuda"]["device_count"]):
            device: dict[str, Any] = {
                "index": index, "status": "unknown", "name": None,
                "compute_capability": None, "total_vram_bytes": None,
            }
            try:
                properties = torch.cuda.get_device_properties(index)
                device.update({
                    "status": "reported", "name": str(properties.name),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                    "total_vram_bytes": int(properties.total_memory),
                })
            except Exception as error:
                device["error"] = error_record("torch_cuda_device_properties", error)
            report["cuda"]["devices"].append(device)
    if run_cuda_smoke:
        if report["cuda"]["available"] is True and (report["cuda"]["device_count"] or 0) > 0:
            report["smoke"] = cuda_smoke(torch)
        else:
            report["smoke"] = {"status": "unavailable", "reason": "cuda_not_confirmed_available"}
    return report


def write_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(payload, end="")
    else:
        # Exclusive creation intentionally refuses to overwrite an earlier probe.
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-smoke", action="store_true", help="explicitly allocate tiny CUDA tensors and check a float32 matmul")
    parser.add_argument("--output", type=Path, help="new JSON file; parent must exist and existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        print("Output already exists; no probe or overwrite performed.", file=sys.stderr)
        return 2
    report = collect_report(run_cuda_smoke=args.cuda_smoke)
    try:
        write_report(report, args.output)
    except (OSError, ValueError) as error:
        print(json.dumps(error_record("write_report", error)), file=sys.stderr)
        return 2
    return 2 if args.cuda_smoke and report["smoke"]["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
