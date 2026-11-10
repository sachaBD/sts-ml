import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runs import run

WRITE_SUMMARY = """
import json, os
open(os.path.join(os.environ["RUN_OUT"], "summary.json"), "w").write(json.dumps({"ok": 1}))
"""


class LauncherTest(unittest.TestCase):
    def test_run_records_status_and_summary(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(run, "RUNS", root / "runs"), mock.patch.object(run, "git_state", dict):
            code = run.main(["gumbel_nn_combat_v1", "t", "--", sys.executable, "-c", WRITE_SUMMARY])
        record = json.loads(next((root / "runs").glob("schema=gumbel_nn_combat_v1/*/id=t/run.json")).read_text())
        self.assertEqual(code, 0)
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["summary"], {"ok": 1})


if __name__ == "__main__":
    unittest.main()
