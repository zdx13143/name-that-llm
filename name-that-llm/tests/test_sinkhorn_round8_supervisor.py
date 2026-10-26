"""Exercise the actual venv ancestry guard without reading experiment inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from windows_sinkhorn_round8_startup import require_supervisor


@unittest.skipUnless(sys.platform == "win32", "Native Windows process test")
class SupervisorTests(unittest.TestCase):
    def test_only_live_unused_supervisor_passes_redirector(self):
        claim = {"state": "started", "supervisor_pid": os.getpid(),
                 "supervisor_create_time": psutil.Process().create_time(),
                 **{key: 0 for key in ("attempted_transforms", "completed_transforms",
                    "model_fit_calls", "model_prediction_calls", "new_encoder_forward_calls", "old_holdout_evaluations")}}
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--child", json.dumps(claim)],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"accepted": 1, "rejected": 5})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        claim = json.loads(sys.argv[2])
        require_supervisor(claim)
        rejected = 0
        for change in ({"supervisor_pid": -1}, {"supervisor_create_time": 0},
                       {"state": "completed"}, {"attempted_transforms": 1}, {"model_fit_calls": 1}):
            try:
                require_supervisor({**claim, **change})
            except RuntimeError:
                rejected += 1
        print(json.dumps({"accepted": 1, "rejected": rejected}))
    else:
        unittest.main()
