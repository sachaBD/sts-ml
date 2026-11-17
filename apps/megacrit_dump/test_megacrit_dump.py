"""Offline tests with synthetic .json.gz data: filtering, Champ indexing, dedupe, resume, listing parse, file choice."""
import datetime as dt
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from apps.megacrit_dump import drive, list as lister, pull
from apps.megacrit_dump.schema import load


def run_rec(pid, char="IRONCLAD", asc=20, champ_floor=None, **kw):
    n = 20
    ev = {"play_id": pid, "character_chosen": char, "ascension_level": asc, "victory": False, "floor_reached": 20,
          "seed_played": "123", "timestamp": 1540000000, "build_version": "2018-10-25",
          "current_hp_per_floor": [100 + i for i in range(n)], "max_hp_per_floor": [200 + i for i in range(n)],
          "damage_taken": [{"enemies": "Jaw Worm", "damage": 3, "turns": 2, "floor": 1}]}
    if champ_floor:
        ev["damage_taken"].append({"enemies": "Champ", "damage": 17, "turns": 9, "floor": champ_floor})
    ev.update(kw)
    return {"event": ev}


def gz(items):
    return gzip.compress(json.dumps(items).encode())


FILE_A = gz([
    run_rec("a1", champ_floor=14),
    run_rec("a2", asc=19),
    run_rec("a3", char="SILENT"),
    run_rec("a4", is_daily=True),
    run_rec("a5", is_endless=True),
    run_rec("a6", is_trial=True),
    run_rec("a7", chose_seed=True),
    run_rec("a8"),
    run_rec("a9", champ_floor=14, victory=True, floor_reached=55),
])
FILE_B = gz([run_rec("a1", champ_floor=14), run_rec("b1", champ_floor=15, floor_reached=15)])  # a1 overlaps FILE_A


class T(unittest.TestCase):
    def test_filters_and_champ_indexing(self):
        rows, seen = pull.process(FILE_A, "fa", "a.json.gz", "IRONCLAD", 20)
        self.assertEqual(seen, 9)
        self.assertEqual([r["play_id"] for r in rows], ["a1", "a8", "a9"])
        r = rows[0]
        # Champ on floor 14 -> state entering it is the end of floor 13 = index 12
        self.assertEqual((r["champ_floor"], r["hp_before_champ"], r["max_hp_before_champ"]), (14, 112, 212))
        self.assertEqual((r["champ_damage"], r["champ_turns"], r["reached_champ"]), (17, 9, True))
        self.assertTrue(r["champ_won"])  # floor_reached 20 > 14
        self.assertIsNone(rows[1]["champ_floor"])
        self.assertFalse(rows[1]["reached_champ"])
        self.assertEqual(json.loads(r["raw"])["play_id"], "a1")

    def test_champ_loss_and_short_lists(self):
        ev = run_rec("x", champ_floor=15, floor_reached=15, current_hp_per_floor=[5])["event"]
        c = pull.champ_fields(ev)
        self.assertFalse(c["champ_won"])
        self.assertIsNone(c["hp_before_champ"])  # list too short -> null, not a wrong value

    def test_run_dedupe_resume(self):
        S = load()
        data = {"fa": FILE_A, "fb": FILE_B}
        calls = []

        def dl(fid):
            calls.append(fid)
            return data[fid]

        files = [{"file_id": "fa", "name": "a"}, {"file_id": "fb", "name": "b"}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            s = pull.run(files[:1], out, "IRONCLAD", 20, 2, download=dl, log=lambda *_: None)
            self.assertEqual((s["files_processed"], s["runs_kept"]), (1, 3))
            s = pull.run(files, out, "IRONCLAD", 20, 2, download=dl, log=lambda *_: None)
            self.assertEqual(calls, ["fa", "fb"])  # fa skipped on restart
            self.assertEqual((s["files_processed"], s["runs_seen"], s["runs_kept"]), (2, 11, 4))  # a1 deduped
            self.assertEqual((s["kept_reached_champ"], s["kept_champ_wins"]), (3, 3 - 1 + 0))
            t = pa.concat_tables([pq.read_table(p) for p in sorted((out / "runs").glob("*.parquet"))])
            self.assertEqual(sorted(t["play_id"].to_pylist()), ["a1", "a8", "a9", "b1"])
            self.assertEqual(t.schema, S.RUNS)

    def test_throttle_stops(self):
        def dl(fid):
            raise drive.Throttled("429")

        with tempfile.TemporaryDirectory() as d:
            s = pull.run([{"file_id": "f", "name": "n"}], Path(d), "IRONCLAD", 20, 1, download=dl, log=lambda *_: None)
            self.assertTrue(s["throttled"])
            self.assertEqual(s["files_processed"], 0)

    def test_choose_files_even_and_deterministic(self):
        base = dt.datetime(2018, 10, 25)
        rows = [{"file_id": str(i), "name": f"{i:05d}", "folder_path": "",
                 "timestamp": base + dt.timedelta(days=i if i < 50 else i + 400)} for i in range(100)]
        t = pa.Table.from_pylist(rows, schema=load().FILES)
        a = pull.choose_files(t, 10)
        self.assertEqual(a, pull.choose_files(t, 10))
        self.assertEqual(len(a), 10)
        self.assertEqual((a[0]["file_id"], a[-1]["file_id"]), ("0", "99"))
        self.assertEqual(len({r["file_id"] for r in a}), 10)

    def test_exclude_done(self):
        t = pa.Table.from_pylist([{"file_id": str(i), "name": str(i), "folder_path": "",
                                   "timestamp": dt.datetime(2020, 1, 1 + i)} for i in range(5)], schema=load().FILES)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "out" / "done").mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{"file_id": "1", "runs_seen": 0, "runs_kept": 0, "kept_reached_champ": 0,
                                                  "kept_champ_won": 0, "bytes": 0}], schema=load().DONE),
                           Path(d) / "out" / "done" / "part-00000.parquet")
            self.assertEqual(pull.exclude_done(t, [d])["file_id"].to_pylist(), ["0", "2", "3", "4"])
            self.assertEqual(pull.exclude_done(t, [])["file_id"].to_pylist(), ["0", "1", "2", "3", "4"])

    def test_listing_parse_and_crawl(self):
        def entry(i, name, folder):
            href = f"https://drive.google.com/drive/folders/{i}" if folder else f"https://drive.google.com/file/d/{i}/view"
            return (f'<div class="flip-entry" id="entry-{i}" tabindex="0"><div class="flip-entry-info"><a href="{href}">'
                    f'<div class="flip-entry-title">{name}</div></a></div></div>')

        pages = {"root": entry("sub", "Monthly_2020_10", True) + entry("f1", "2018-10-25-02-34#1352.json.gz", False),
                 "sub": entry("f2", "2020-10-01-00-06#870.json.gz", False)}
        rows, folders = lister.crawl("root", fetch_html=pages.__getitem__)
        self.assertEqual({(r["file_id"], r["folder_path"]) for r in rows}, {("f1", ""), ("f2", "Monthly_2020_10")})
        self.assertEqual(rows[0]["timestamp"] if rows[0]["file_id"] == "f1" else rows[1]["timestamp"],
                         dt.datetime(2018, 10, 25, 2, 34))
        self.assertFalse(any(f["capped"] for f in folders.values()))

    def test_gzip_guard(self):
        orig = drive.fetch
        try:
            drive.fetch = lambda *a, **k: b"<html>Quota exceeded</html>"
            with self.assertRaises(drive.Throttled):
                drive.download("x")
        finally:
            drive.fetch = orig


if __name__ == "__main__":
    unittest.main()
