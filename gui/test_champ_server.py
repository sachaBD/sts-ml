"""Champ viewer server: a recorded Champ fight replays through champ_session to its recorded outcome.

Needs build/champ-viewer/champ_session and the champ-ox-c-r01 run; skipped when either is missing.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGHT = dict(date="2026-10-04", id="champ-ox-c-r01", part="bench-oracle", fight="ah-fresh-inc:981000000068:18")
DATA = ROOT / f"runs/schema=combat_v4/date={FIGHT['date']}/id={FIGHT['id']}/{FIGHT['part']}"


@unittest.skipUnless((ROOT / "build/champ-viewer/champ_session").exists() and DATA.exists(), "needs champ_session and run data")
class ChampServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "gui"))
        import champ_server
        cls.s = champ_server

    @classmethod
    def tearDownClass(cls):
        p = cls.s.FAST.proc
        p.stdin.close(); p.wait(); p.stdout.close()

    def test_lists_datasets_and_champ_fights(self):
        self.assertIn(dict(date=FIGHT["date"], id=FIGHT["id"], part=FIGHT["part"]), self.s.meta({})["datasets"])
        ids = [f["fight_id"] for f in self.s.fights(FIGHT)["fights"]]
        self.assertIn(FIGHT["fight"], ids)

    def test_replay_matches_recording(self):
        rec = self.s.fight(FIGHT)
        ops = []
        for a in rec["actions"]:
            view = self.s.query(dict(base=rec["base"], ops=ops, query="view"))["view"]
            self.assertIn(a, [l["bits"] for l in view["legal"]])
            ops.append({"act": a})
        view = self.s.query(dict(base=rec["base"], ops=ops, query="view"))["view"]
        self.assertEqual((view["kind"], view["won"], view["player"]["hp"]), ("done", rec["won"], rec["final_hp"]))

    def test_rejects_paths_outside_runs(self):
        with self.assertRaises(ValueError):
            self.s.fights(dict(date=FIGHT["date"], id="../../../etc"))


if __name__ == "__main__":
    unittest.main()
