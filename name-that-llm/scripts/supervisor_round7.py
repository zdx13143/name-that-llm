"""Windows executable identity without psutil's drive-letter translation."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys

import psutil


def executable_path(pid):
    if sys.platform != "win32":
        raise RuntimeError("Native Windows executable query required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                 wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not Path(buffer.value).is_absolute():
            raise RuntimeError("Executable query returned a non-absolute path")
        return os.path.normcase(str(Path(buffer.value).resolve()))
    finally:
        kernel.CloseHandle(handle)


def require_live_supervisor(claim):
    failure = "Execution requires the live supervisor and an unused claim"
    if claim.get("state") != "started" or any(claim.get(k) != 0 for k in
            ("attempted_forward_calls", "completed_forward_calls", "attempted_fits", "completed_fits",
             "attempted_vectorizer_fits", "completed_vectorizer_fits")):
        raise RuntimeError(failure)
    try:
        current = psutil.Process()
        parent = current.parent()
        if parent is None:
            raise RuntimeError(failure)
        if parent.pid != claim.get("supervisor_pid"):
            if (sys.platform != "win32" or sys.prefix == sys.base_prefix
                    or Path(sys.executable).resolve() == Path(sys._base_executable).resolve()
                    or executable_path(current.pid) != os.path.normcase(str(Path(sys._base_executable).resolve()))
                    or executable_path(parent.pid) != os.path.normcase(str(Path(sys.executable).resolve()))):
                raise RuntimeError(failure)
            parent = parent.parent()
        if (parent is None or parent.pid != claim.get("supervisor_pid")
                or parent.create_time() != claim.get("supervisor_create_time")):
            raise RuntimeError(failure)
    except (psutil.Error, OSError) as error:
        raise RuntimeError(failure) from error
