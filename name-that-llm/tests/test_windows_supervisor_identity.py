"""Real Windows venv process topology, without data, CUDA, or classifier fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import psutil


CHILD = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import windows_modernbert_dev as dev
dev.CLAIM = Path(sys.argv[2])
class GuardAccepted(Exception):
    pass
def no_data(*args):
    raise GuardAccepted()
dev.load_development = no_data
try:
    dev.execute(Path("NO_DATA_MUST_BE_READ"), {}, {"input": {}})
except GuardAccepted:
    print(json.dumps({"accepted": True, "pid": os.getpid(), "ppid": os.getppid()}))
except RuntimeError as error:
    print(json.dumps({"accepted": False, "error": str(error), "pid": os.getpid(), "ppid": os.getppid()}))
    raise SystemExit(3)
'''


@unittest.skipUnless(sys.platform == "win32", "Registered native Windows launcher regression")
class SupervisorIdentityTests(unittest.TestCase):
    def invoke(self, changes=None):
        claim = {"state": "started", "supervisor_pid": os.getpid(),
                 "supervisor_create_time": psutil.Process().create_time(),
                 "attempted_forward_calls": 0, "completed_forward_calls": 0,
                 "attempted_fits": 0, "completed_fits": 0}
        claim.update(changes or {})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claim.json"
            path.write_text(json.dumps(claim), encoding="utf-8")
            return subprocess.run([sys.executable, "-c", CHILD,
                                   str(Path(__file__).resolve().parents[1] / "scripts"), str(path)],
                                  capture_output=True, text=True, timeout=20)

    def test_real_venv_child_accepts_its_live_supervisor(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["accepted"])
        if Path(sys.executable).resolve() != Path(sys._base_executable).resolve():
            self.assertNotEqual(json.loads(result.stdout)["ppid"], os.getpid(), "Expected the real venv redirector layer")

    def test_stale_identity_and_consumed_claim_are_rejected(self):
        for changes in ({"supervisor_pid": 0}, {"supervisor_create_time": 0},
                        {"state": "failed"}, {"attempted_forward_calls": 1}, {"attempted_fits": 1}):
            with self.subTest(changes=changes):
                result = self.invoke(changes)
                self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
                self.assertFalse(json.loads(result.stdout)["accepted"])


if __name__ == "__main__":
    unittest.main()
