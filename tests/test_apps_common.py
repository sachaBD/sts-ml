"""apps/common: worker calls, the parallel loop, parquet parts, snapshots, replay requests."""
import hashlib
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from apps.common import app, replay, worker

# A stand-in worker: REQUEST.json OUTPUT_DIR [WEIGHTS] -> result.msgpack = {request, weights}.
FAKE_WORKER = """#!{python}
import json, sys, msgpack
request = json.load(open(sys.argv[1]))
if request.get("fail"):
    sys.exit("worker failed on purpose")
result = {{"request": request, "weights": sys.argv[3] if len(sys.argv) > 3 else None}}
open(sys.argv[2] + "/result.msgpack", "wb").write(msgpack.packb(result))
"""


def decision(seed, fight, index, action, **columns):
    row = {"run_seed": seed, "fight_index": fight, "episode_id": seed * 100 + fight, "decision_index": index,
           "chosen_action": action, "ascension": 20}
    return {**row, **dict.fromkeys(replay.START, 0), **columns}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.binary = Path(self.tmp.name) / "fake_worker"
        self.binary.write_text(FAKE_WORKER.format(python=sys.executable))
        self.binary.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def test_request_and_weights(self):
        self.assertEqual(worker.run_worker(self.binary, {"a": 1}), {"request": {"a": 1}, "weights": None})
        self.assertEqual(worker.run_worker(self.binary, {}, Path("w.bin"))["weights"], "w.bin")

    def test_failure_carries_stderr(self):
        with self.assertRaisesRegex(RuntimeError, "on purpose"):
            worker.run_worker(self.binary, {"fail": True})


class ParallelTests(unittest.TestCase):
    def test_all_results_endless_items_bounded(self):
        seen = []
        worker.run_parallel(lambda x: x * 2, range(10), 3, lambda item, result: seen.append((item, result)))
        self.assertEqual(sorted(seen), [(i, 2 * i) for i in range(10)])
        # Endless items: only `workers` are taken at a time, so an exception ends the loop.
        taken = itertools.count()

        def fn(x):
            if x == 5:
                raise ValueError("stop")
            return x

        with self.assertRaisesRegex(ValueError, "stop"):
            worker.run_parallel(fn, taken, 2)
        self.assertLess(next(taken), 20)


class PartTests(unittest.TestCase):
    def test_atomic_part_with_metadata(self):
        with tempfile.TemporaryDirectory() as out:
            out = Path(out)
            worker.write_part(out, 7, [], {"collection_method": "dagger"})
            self.assertEqual([p.name for p in out.iterdir()], ["part-00000007.parquet"])
            metadata = pq.read_schema(out / "part-00000007.parquet").metadata
            self.assertEqual(metadata[b"schema"], b"combat_v3")
            self.assertEqual(metadata[b"collection_method"], b"dagger")


class AppTests(unittest.TestCase):
    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            built, out = Path(tmp) / "built", Path(tmp) / "out"
            out.mkdir()
            built.write_bytes(b"worker v1")
            built.chmod(0o755)
            copy, sha = app.snapshot(built, out, "worker")
            built.write_bytes(b"worker v2")  # a later rebuild doesn't change the run's copy
            self.assertEqual(copy, (out / "worker").resolve())
            self.assertEqual(copy.read_bytes(), b"worker v1")
            self.assertEqual(sha, hashlib.sha256(b"worker v1").hexdigest())
            self.assertFalse(copy.stat().st_mode & 0o222)  # read-only

    def test_config_checks(self):
        app.check_keys({"a": 1}, {"a", "b"}, "run")
        with self.assertRaises(SystemExit):
            app.check_keys({"c": 1}, {"a"}, "run")
        self.assertFalse(app.flag({}, "oracle"))
        self.assertTrue(app.flag({"oracle": True}, "oracle"))
        with self.assertRaises(SystemExit):
            app.flag({"oracle": 1}, "oracle")


class ReplayTests(unittest.TestCase):
    ROWS = [decision(3, 1, 0, 9, encounter="slime_boss"), decision(3, 0, 1, 2), decision(3, 0, 0, 1),
            decision(3, 1, 1, 8), decision(4, 0, 0, 5)]

    def test_requests(self):
        requests = replay.replay_requests(self.ROWS, [301, 400])
        request, start = requests[301]
        self.assertEqual(request, {"run_seed": 3, "ascension": 20, "fight_index": 1, "actions": [[1, 2]]})
        self.assertEqual(start["encounter"], "slime_boss")
        self.assertEqual(requests[400][0]["actions"], [])

    def test_missing_and_duplicate(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            replay.replay_requests(self.ROWS, [302])
        with self.assertRaisesRegex(RuntimeError, "earlier fight 0"):
            replay.replay_requests([r for r in self.ROWS if (r["run_seed"], r["fight_index"], r["decision_index"]) != (3, 0, 0)], [301])
        with self.assertRaisesRegex(RuntimeError, "stored twice"):
            replay.replay_requests(self.ROWS + self.ROWS[:1], [301])

    def test_check_start(self):
        stored = decision(3, 1, 0, 9)
        replay.check_start(301, dict(stored), stored)
        with self.assertRaisesRegex(RuntimeError, "floor"):
            replay.check_start(301, {**stored, "floor": 2}, stored)


if __name__ == "__main__":
    unittest.main()
