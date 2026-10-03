"""combat_v4 -> combat_v4_full on four recorded fights (testdata/): replay, exact schema, step / chosen / search join.

Runs build/<STSRL_BUILD_DIR, default main>/combat_v4_decompress (built when pyarrow is in the project venv).
"""
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "testdata"
BINARY = ROOT / "build" / os.environ.get("STSRL_BUILD_DIR", "main") / "combat_v4_decompress"


def load_schema():
    spec = importlib.util.spec_from_file_location("combat_v4_full", ROOT / "runs/schema=combat_v4_full/schema.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(BINARY.exists(), f"{BINARY} not built")
class DecompressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        subprocess.run([str(BINARY), "--out", str(out), "--fights", str(DATA / "fights-test.parquet"),
                        "--search", str(DATA / "search-test.parquet")], check=True)
        cls.source = {r["fight_id"]: r for r in pq.read_table(DATA / "fights-test.parquet").to_pylist()}
        cls.decisions = pq.read_table(out / "decisions-0.parquet")
        cls.fights = pq.read_table(out / "fights-0.parquet")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_replays_match(self):
        rows = {r["fight_id"]: r for r in self.fights.to_pylist()}
        self.assertEqual(set(rows), set(self.source))
        for fid, src in self.source.items():
            self.assertEqual((rows[fid]["won"], rows[fid]["final_hp"]), (src["won"], src["final_hp"]))
            self.assertEqual(rows[fid]["decisions"], len(src["actions"]))

    def test_schema_exact(self):
        schema = load_schema()
        self.assertTrue(self.decisions.schema.equals(schema.DECISIONS, check_metadata=False))
        self.assertTrue(self.fights.schema.equals(schema.FIGHTS, check_metadata=False))

    def test_steps_chosen_search(self):
        by_fight = {}
        for d in self.decisions.to_pylist():
            by_fight.setdefault(d["fight_id"], []).append(d)
        searched = 0
        for fid, src in self.source.items():
            rows = sorted(by_fight[fid], key=lambda d: d["step"])
            self.assertEqual([d["step"] for d in rows], list(range(len(src["actions"]))))
            for d in rows:
                self.assertEqual(d["legal"][d["chosen"]]["action"], src["actions"][d["step"]])
                self.assertEqual(d["explored"], src["explored"][d["step"]])
                if d["simulations"] is not None:
                    searched += 1
                    self.assertTrue(any(m["visits"] for m in d["legal"]))
        self.assertGreater(searched, 0)

    def test_champ_phase_is_monotone(self):
        champ = [f["fight_id"] for f in self.fights.to_pylist() if f["encounter"] == "champ"]
        self.assertTrue(champ)
        for fid in champ:
            phase = [next(c["amount"] for m in d["public"]["monsters"] for c in m["counters"] if c["status"] == "phase2")
                     for d in sorted((d for d in self.decisions.to_pylist() if d["fight_id"] == fid), key=lambda d: d["step"])]
            self.assertEqual(phase, sorted(phase))


if __name__ == "__main__":
    unittest.main()
