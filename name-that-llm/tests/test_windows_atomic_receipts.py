"""Receipt replacement under real Windows reader locks; no model/data work."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_modernbert as probe


@unittest.skipUnless(sys.platform == "win32", "Windows file sharing regression")
class AtomicReceiptTests(unittest.TestCase):
    def test_transient_reader_lock_does_not_fail_receipt_update(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claim.json"
            probe.write_json(path, {"attempted": 0})
            handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            release = threading.Timer(0.15, lambda: kernel.CloseHandle(handle))
            release.start()
            try:
                probe.write_json(path, {"attempted": 1}, exclusive=False)
            finally:
                release.join()
            self.assertEqual(json.loads(path.read_bytes()), {"attempted": 1})

    def test_permanent_denial_is_bounded_and_preserves_old_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claim.json"
            probe.write_json(path, {"attempted": 0})
            before = path.read_bytes()
            error = PermissionError("synthetic permanent denial")
            error.winerror = 5
            with patch.object(probe.os, "replace", side_effect=error) as replacement:
                with patch.object(probe.time, "sleep") as sleep:
                    with self.assertRaises(PermissionError):
                        probe.write_json(path, {"attempted": 1}, exclusive=False)
            self.assertEqual(replacement.call_count, 8)
            self.assertEqual(sleep.call_count, 7)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.with_suffix(".tmp").read_bytes()), {"attempted": 1})

    def test_unrelated_io_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claim.json"
            probe.write_json(path, {"attempted": 0})
            with patch.object(probe.os, "replace", side_effect=FileNotFoundError("synthetic missing target")) as replacement:
                with self.assertRaises(FileNotFoundError):
                    probe.write_json(path, {"attempted": 1}, exclusive=False)
            self.assertEqual(replacement.call_count, 1)


if __name__ == "__main__":
    unittest.main()
