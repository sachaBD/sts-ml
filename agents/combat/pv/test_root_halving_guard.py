"""Root-halving records are evaluation-only: worker encode and data.collect must refuse them; legacy rows unchanged.

Zero gameplay: uses the recorded dev:0 baseline fight. PV_WORKER selects the worker binary under test.
Run from the repo root: PV_WORKER=build/expert-champ-halving/agents/combat/pv/pv_worker .venv/bin/python -m unittest agents.combat.pv.test_root_halving_guard
"""
import json, os, subprocess, tempfile, unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import data

REPO = Path(__file__).resolve().parents[3]
WORKER = Path(os.environ.get('PV_WORKER', REPO / 'build/expert-champ-halving/agents/combat/pv/pv_worker'))
DEV = REPO / 'runs/schema=combat_v4/date=2026-10-06/id=demon-form-decoupled-policy-v1/out/main/baseline'
HIVE = ('schema', 'date', 'id')


def line():
    r = json.loads((DEV / 'results.jsonl').open().readline()); f = r['fight']
    return {'fight_id': f['fight_id'], 'start': f['start'], 'actions': f['actions'], 'won': f['won'], 'final_hp': f['final_hp'],
            'agent': f['agent'], 'search': [{k: s[k] for k in ('step', 'root_value', 'children', 'agent')} for s in r['search']]}


def encode(lines, out):
    return subprocess.run([str(WORKER), 'encode', str(out)], input=''.join(json.dumps(x) + '\n' for x in lines),
                          text=True, capture_output=True)


def rows(p): return [{k: v for k, v in r.items() if k not in HIVE} for r in pq.read_table(p).to_pylist()]


@unittest.skipUnless(WORKER.exists() and DEV.exists(), 'worker or recorded fixture unavailable')
class RootHalvingGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.tmp = Path(self._tmp.name)

    def tearDown(self): self._tmp.cleanup()

    def test_legacy_unchanged(self):
        x = line(); p = encode([x], self.tmp / 'a.parquet')
        self.assertEqual(p.returncode, 0, p.stderr)
        ref = [r for r in rows(DEV / 'encoded/rows.parquet') if r['fight_id'] == x['fight_id']]
        self.assertEqual(rows(self.tmp / 'a.parquet'), ref)

    def test_worker_rejects_marked(self):
        for where in ('top_field', 'top_agent', 'search_field', 'search_agent'):
            with self.subTest(where=where):
                x = line()
                if where == 'top_field': x['root_halving'] = True
                if where == 'top_agent': x['agent'] += ' root_halving=m16,g0.000000'
                if where == 'search_field': x['search'][0]['root_halving'] = True
                if where == 'search_agent': x['search'][3]['agent'] += ' root_halving=m16,g0.000000'
                out = self.tmp / f'{where}.parquet'; p = encode([x], out)
                self.assertNotEqual(p.returncode, 0); self.assertIn('root_halving', p.stderr)
                # A failed `encode` leaves its OUT file partial (pre-existing behaviour); data.collect writes to .tmp and removes it.

    def test_collect_propagates_and_rejects(self):
        def strip(t): return t.drop_columns([c for c in HIVE if c in t.schema.names])
        fights = strip(pq.read_table(DEV / 'fights-0.parquet')).slice(0, 2)
        search = strip(pq.read_table(DEV / 'search-0.parquet'))
        fp, sp = self.tmp / 'fights.parquet', self.tmp / 'search.parquet'
        pq.write_table(fights, fp); pq.write_table(search, sp)
        self.assertEqual(data.collect([fp], [sp], [39], self.tmp / 'ok', worker=WORKER), 2)
        target = fights['fight_id'][0].as_py()
        agents = [a + (' root_halving=m16,g0.000000' if f == target else '') for a, f in zip(search['agent'].to_pylist(), search['fight_id'].to_pylist())]
        marked = search.set_column(search.schema.get_field_index('agent'), 'agent', pa.array(agents))
        mp = self.tmp / 'marked.parquet'; pq.write_table(marked, mp)
        with self.assertRaisesRegex(RuntimeError, 'root_halving'):
            data.collect([fp], [mp], [39], self.tmp / 'bad', worker=WORKER)
        self.assertFalse((self.tmp / 'bad' / 'rows.parquet').exists())


if __name__ == '__main__':
    unittest.main()
