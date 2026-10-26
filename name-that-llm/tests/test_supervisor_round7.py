"""Run the actual guard through the Windows venv redirector; no data or fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from supervisor_round7 import require_live_supervisor


@unittest.skipUnless(sys.platform == "win32", "Native Windows process test")
class SupervisorTests(unittest.TestCase):
    def test_venv_chain_accepts_only_the_live_unused_parent(self):
        claim = {"state": "started", "supervisor_pid": os.getpid(),
                 "supervisor_create_time": psutil.Process().create_time(),
                 **{key: 0 for key in ("attempted_forward_calls", "completed_forward_calls", "attempted_fits",
                                     "completed_fits", "attempted_vectorizer_fits", "completed_vectorizer_fits")}}
        for flags in ([], ["-u"]):
            completed = subprocess.run([sys.executable, *flags, str(Path(__file__).resolve()), "--child", json.dumps(claim)],
                                       text=True, capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), {"accepted": 1, "rejected": 3})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        value = json.loads(sys.argv[2])
        require_live_supervisor(value)
        rejected = 0
        for change in ({"supervisor_create_time": 0}, {"supervisor_pid": -1}, {"attempted_vectorizer_fits": 1}):
            try:
                require_live_supervisor({**value, **change})
            except RuntimeError:
                rejected += 1
        if rejected != 3:
            raise AssertionError("Invalid supervisor claim accepted")
        print(json.dumps({"accepted": 1, "rejected": rejected}))
    else:
        unittest.main()
