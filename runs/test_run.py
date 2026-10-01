import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from runs import compact, run

WRITE_PARTS = """
import os, pyarrow as pa, pyarrow.parquet as pq
for k in range(5):
    pq.write_table(pa.table({"k": [k] * 10, "v": list(range(10))}), os.path.join(os.environ["RUN_OUT"], f"part-{k:03d}.parquet"))
"""


def parquet(out):
    return sorted(p.name for p in Path(out).glob("*.parquet"))


class LauncherCompactsTest(unittest.TestCase):
    def launch(self, *flags):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(run, "RUNS", root / "runs"), mock.patch.object(run, "git_state", dict):
            code = run.main(["combat_v3", "t", *flags, "--", sys.executable, "-c", WRITE_PARTS])
        run_dir = next((root / "runs").glob("schema=combat_v3/*/id=t"))
        return code, run_dir

    def test_parts_are_compacted_before_done(self):
        code, run_dir = self.launch()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["status"], "done")
        [name] = parquet(run_dir / "out")
        self.assertTrue(name.startswith("compact-"))
        table = pq.read_table(run_dir / "out" / name)
        self.assertEqual(table.num_rows, 50)
        self.assertEqual(pq.ParquetFile(run_dir / "out" / name).metadata.num_row_groups, 1)

    def test_no_compact(self):
        _, run_dir = self.launch("--no-compact")
        self.assertEqual(len(parquet(run_dir / "out")), 5)


class CompactTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_healthy_file_is_left_alone(self):
        pq.write_table(pa.table({"v": list(range(100))}), self.dir / "big.parquet")
        self.assertIsNone(compact.compact(self.dir, target=1))  # already over target, one row group
        self.assertEqual(parquet(self.dir), ["big.parquet"])

    def test_file_with_tiny_row_groups_is_rewritten(self):
        pq.write_table(pa.table({"v": list(range(1000))}), self.dir / "old.parquet", row_group_size=10)
        self.assertEqual(compact.compact(self.dir, target=1), (1, 1, 1000))
        [name] = parquet(self.dir)
        self.assertEqual(pq.ParquetFile(self.dir / name).metadata.num_row_groups, 1)
        self.assertEqual(pq.read_table(self.dir / name)["v"].to_pylist(), list(range(1000)))
        self.assertIsNone(compact.compact(self.dir, target=1))  # second pass: nothing to do


if __name__ == "__main__":
    unittest.main()
