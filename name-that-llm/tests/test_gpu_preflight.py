"""Safety and evidence tests using synthetic runtime objects, never a real GPU."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_gpu_environment as probe


class GpuPreflightTests(unittest.TestCase):
    def setUp(self):
        self.ram = patch.object(probe, "physical_memory_bytes", return_value=16 * 1024 ** 3)
        self.version = patch.object(probe.importlib.metadata, "version", return_value="synthetic-test-version")
        self.driver = patch.object(probe, "driver_metadata", return_value={"status": "not_found", "driver_versions": None})
        self.ram.start()
        self.version.start()
        self.driver.start()
        self.addCleanup(self.ram.stop)
        self.addCleanup(self.version.stop)
        self.addCleanup(self.driver.stop)

    def runtime(self, available=True):
        return SimpleNamespace(
            __version__="synthetic-test-version", version=SimpleNamespace(cuda="synthetic-build"),
            cuda=SimpleNamespace(
                is_available=Mock(return_value=available), device_count=Mock(return_value=1 if available else 0),
                get_device_properties=Mock(return_value=SimpleNamespace(
                    name="SYNTHETIC DEVICE, NOT A HOST OBSERVATION", major=8, minor=9,
                    total_memory=8 * 1024 ** 3,
                )),
            ),
        )

    def test_missing_torch_is_not_a_claim_of_no_physical_gpu(self):
        failure = ModuleNotFoundError("private/path/token", name="torch")
        with patch.object(probe.importlib, "import_module", side_effect=failure):
            report = probe.collect_report(run_cuda_smoke=True)
        self.assertEqual(report["torch"]["import_status"], "not_installed")
        self.assertIsNone(report["cuda"]["available"])
        self.assertIsNone(report["cuda"]["device_count"])
        self.assertEqual(report["smoke"]["status"], "unavailable")
        self.assertNotIn("private/path/token", json.dumps(report))

    def test_missing_torch_dependency_is_import_error_not_not_installed(self):
        failure = ModuleNotFoundError("private content", name="torch_dependency")
        with patch.object(probe.importlib, "import_module", side_effect=failure):
            report = probe.collect_report()
        self.assertEqual(report["torch"]["import_status"], "error")
        self.assertIsNone(report["cuda"]["available"])

    def test_default_only_enumerates_metadata_and_never_runs_smoke(self):
        runtime = self.runtime()
        with patch.object(probe.importlib, "import_module", return_value=runtime), patch.object(probe, "cuda_smoke") as smoke:
            report = probe.collect_report()
        smoke.assert_not_called()
        self.assertEqual(report["scope"], "current_host_only")
        self.assertEqual(report["smoke"]["status"], "not_requested")
        self.assertEqual(report["cuda"]["devices"][0]["total_vram_bytes"], 8 * 1024 ** 3)
        self.assertEqual(report["cuda"]["devices"][0]["compute_capability"], [8, 9])

    def test_runtime_error_is_unknown_not_false_and_is_sanitized(self):
        runtime = self.runtime()
        runtime.cuda.is_available.side_effect = RuntimeError("PRIVATE TOKEN AND PATH")
        runtime.cuda.device_count.side_effect = RuntimeError("PRIVATE TOKEN AND PATH")
        with patch.object(probe.importlib, "import_module", return_value=runtime):
            report = probe.collect_report(run_cuda_smoke=True)
        self.assertIsNone(report["cuda"]["available"])
        self.assertIsNone(report["cuda"]["device_count"])
        self.assertEqual(report["cuda"]["devices"], [])
        self.assertEqual(report["smoke"]["status"], "unavailable")
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_property_failure_does_not_fabricate_vram(self):
        runtime = self.runtime()
        runtime.cuda.get_device_properties.side_effect = RuntimeError("private")
        with patch.object(probe.importlib, "import_module", return_value=runtime):
            report = probe.collect_report()
        self.assertEqual(report["cuda"]["devices"][0]["status"], "unknown")
        self.assertIsNone(report["cuda"]["devices"][0]["total_vram_bytes"])

    def test_smoke_is_explicit_and_propagates_failure(self):
        runtime = self.runtime()
        with patch.object(probe.importlib, "import_module", return_value=runtime), patch.object(probe, "cuda_smoke", return_value={"status": "failed"}) as smoke:
            report = probe.collect_report(run_cuda_smoke=True)
        smoke.assert_called_once_with(runtime)
        self.assertEqual(report["smoke"]["status"], "failed")
        with patch.object(probe, "collect_report", return_value=report), redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["--cuda-smoke"]), 2)

    def test_smoke_exception_is_reported_without_private_error_text(self):
        runtime = SimpleNamespace(inference_mode=Mock(side_effect=RuntimeError("private driver path")))
        report = probe.cuda_smoke(runtime)
        self.assertEqual(report["status"], "error")
        self.assertIsNone(report["max_absolute_error"])
        self.assertEqual(report["error"]["context"], "cuda_smoke")
        self.assertNotIn("private driver path", json.dumps(report))

    def test_known_cpu_torch_does_not_execute_requested_cuda_smoke(self):
        runtime = self.runtime(available=False)
        runtime.version.cuda = None
        with patch.object(probe.importlib, "import_module", return_value=runtime), patch.object(probe, "cuda_smoke") as smoke:
            report = probe.collect_report(run_cuda_smoke=True)
        smoke.assert_not_called()
        self.assertFalse(report["cuda"]["available"])
        self.assertEqual(report["cuda"]["device_count"], 0)
        self.assertEqual(report["smoke"]["status"], "unavailable")

    def test_output_is_exclusive_and_cli_refuses_before_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "report.json"
            probe.write_report({"scope": "synthetic-test"}, output)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                probe.write_report({"scope": "would-overwrite"}, output)
            with patch.object(probe, "collect_report") as collect, redirect_stderr(io.StringIO()):
                self.assertEqual(probe.main(["--output", str(output)]), 2)
            collect.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_ram_failure_remains_unknown_and_json_is_strict(self):
        with patch.object(probe, "physical_memory_bytes", side_effect=OSError("private path")), patch.object(probe.importlib, "import_module", return_value=self.runtime(False)):
            report = probe.collect_report()
        self.assertIsNone(report["host"]["physical_ram_bytes"])
        self.assertIn({"context": "physical_ram", "exception_type": "OSError"}, report["errors"])
        with redirect_stdout(io.StringIO()) as stdout:
            probe.write_report(report, None)
        self.assertEqual(json.loads(stdout.getvalue())["scope"], "current_host_only")
        self.assertNotIn("private path", stdout.getvalue())

    def test_driver_command_reads_only_versions_and_rejects_unexpected_output(self):
        self.driver.stop()
        with patch.object(probe.shutil, "which", return_value="synthetic-nvidia-smi"), patch.object(probe.subprocess, "run", return_value=SimpleNamespace(stdout="999.10\n999.10\n")) as run:
            report = probe.driver_metadata()
        self.assertEqual(report, {"status": "reported", "driver_versions": ["999.10"]})
        self.assertEqual(run.call_args.args[0], ["synthetic-nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        with patch.object(probe.shutil, "which", return_value="synthetic-nvidia-smi"), patch.object(probe.subprocess, "run", return_value=SimpleNamespace(stdout="PRIVATE PATH /token\n")):
            report = probe.driver_metadata()
        self.assertEqual(report["status"], "unknown")
        self.assertIsNone(report["driver_versions"])
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_missing_driver_tool_is_not_evidence_of_no_gpu(self):
        self.driver.stop()
        with patch.object(probe.shutil, "which", return_value=None):
            report = probe.driver_metadata()
        self.assertEqual(report, {"status": "not_found", "driver_versions": None})


if __name__ == "__main__":
    unittest.main()
