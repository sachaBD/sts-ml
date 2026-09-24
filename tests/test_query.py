"""sts_combat_rl.query over a tiny runs/ tree."""
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sts_combat_rl import query
from sts_combat_rl.schemas.combat_v3 import COMBAT_V3


def make_run(root, id_, rows, inputs=()):
    run = root / "schema=combat_v3" / "date=2026-01-01" / f"id={id_}"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": f"combat_v3/2026-01-01/{id_}", "schema": "combat_v3",
                                              "status": "done", "inputs": list(inputs), "summary": None}))
    pq.write_table(pa.Table.from_pylist(rows, schema=COMBAT_V3), run / "out" / "part-1.parquet")


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_run(self.root, "boot-a", [{"episode_id": 100, "encounter": "slime_boss", "row_kind": "decision"},
                                       {"episode_id": 101, "encounter": "jaw_worm", "row_kind": "decision"}])
        make_run(self.root, "boot-b", [{"episode_id": 200, "encounter": "slime_boss", "row_kind": "child"}])
        make_run(self.root, "oracle", [{"episode_id": 300, "encounter": "slime_boss", "oracle": True}])

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_columns_where_and_run_ids(self):
        sql = "select * from combat_v3 where id like 'boot%' and encounter = 'slime_boss'"
        rows = query.rows(sql, ["episode_id"], root=self.root)
        self.assertEqual(sorted(rows, key=lambda r: r["episode_id"]),
                         [{"episode_id": 100, "run_id": "combat_v3/2026-01-01/boot-a"},
                          {"episode_id": 200, "run_id": "combat_v3/2026-01-01/boot-b"}])
        self.assertEqual([r["episode_id"] for r in query.rows(sql, ["episode_id"], "row_kind = 'decision'", root=self.root)],
                         [100])
        self.assertEqual(query.run_ids(sql, root=self.root), ["combat_v3/2026-01-01/boot-a", "combat_v3/2026-01-01/boot-b"])
        self.assertEqual(len(query.rows(sql, root=self.root)[0]), len(COMBAT_V3) + 4)  # + run_id, schema, date, id

    def test_oracle_refused_unless_allowed(self):
        sql = "select * from combat_v3 where encounter = 'slime_boss'"
        with self.assertRaisesRegex(ValueError, "oracle"):
            query.rows(sql, ["episode_id"], root=self.root)
        self.assertEqual(len(query.rows(sql, ["episode_id"], oracle=True, root=self.root)), 3)

    def test_runs_view(self):
        db = query.connect(self.root)
        self.assertEqual(db.sql("select count(*) from runs where status = 'done'").fetchone()[0], 3)

    def test_sql_list(self):
        self.assertEqual(query.sql_list(["a", "b'c"]), "('a', 'b''c')")
        self.assertEqual(query.sql_list([1, 2]), "(1, 2)")
        self.assertEqual(query.sql_list([]), "(null)")


if __name__ == "__main__":
    unittest.main()
