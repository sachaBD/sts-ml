"""History-dependent Liquid Memories replay regression (local r10 model artifact).

Run on one core:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 11 \
    .venv/bin/python -m unittest agents.combat.pv.test_oracle_replay -v
Override PV_REPLAY_WORKER/PV_REPLAY_MODEL to test another frozen executable/model.
"""
import json
import os
from pathlib import Path
import subprocess
import unittest

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(os.environ.get("PV_REPLAY_WORKER", ROOT / "build/pv/agents/combat/pv/pv_worker"))
MODEL = Path(os.environ.get("PV_REPLAY_MODEL", ROOT / "runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r10/model/model.onnx"))
FIXTURE = Path(__file__).with_name("testdata") / "oracle-replay-history.parquet"


class OracleReplayTest(unittest.TestCase):
    @unittest.skipUnless(WORKER.is_file() and MODEL.is_file(), "requires built worker and local r10 ONNX artifact")
    def test_liquid_memories_history(self):
        # The immediate predecessor leaves a Liquid Memories selection task behind.
        # In the original frozen worker this exact pair fails on target action 4:
        # the returned Ghostly Armor's cost depended on stale/uninitialized task state.
        starts = pq.ParquetFile(FIXTURE).read().to_pylist()
        self.assertEqual([f["fight_id"] for f in starts], ["gen:5650011003205", "gen:5650011003206"])
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        played = subprocess.run(
            [str(WORKER), "play", str(MODEL), "800", "--oracle", "--explore", "--sample-turns"],
            input="".join(json.dumps(f) + "\n" for f in starts), text=True,
            capture_output=True, env=env, timeout=600,
        )
        self.assertEqual(played.returncode, 0, played.stderr)
        results = [json.loads(line) for line in played.stdout.splitlines()]
        self.assertEqual(len(results), len(starts))
        for start, result in zip(starts, results):
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["fight"]["fight_id"], start["fight_id"])
            self.assertTrue(result["fight"]["actions"])
            # Successful play already verifies the entire canonical replay, HP and outcome.
            self.assertIsInstance(result["fight"]["won"], bool)


if __name__ == "__main__":
    unittest.main()
