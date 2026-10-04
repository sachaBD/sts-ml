"""PV plumbing on a recorded Champ fight start: teacher plays → encode shard → train 1 epoch + ONNX → PV plays.

Checks plumbing and contracts only (tables readable, targets sane, telemetry present), not playing strength."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from agents.combat.pv import data

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / 'build' / os.environ.get('STSRL_BUILD_DIR', 'pv') / 'agents/combat/pv/pv_worker'
FIGHTS = ROOT / 'apps/combat_record/testdata/fights-test.parquet'
PLAY = ROOT / 'apps/pv/play.py'


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))


@unittest.skipUnless(WORKER.exists(), f'{WORKER} not built (PV needs ONNX Runtime; apps/pv/setup_runtime.sh)')
class Pipeline(unittest.TestCase):
    def test_play_encode_train_play(self):
        champ = [r for r in pq.read_table(FIGHTS, columns=['fight_id', 'start']).to_pylist() if r['start']['encounter'] == 39][0]
        rows = []
        for seed in range(1, 41):  # distinct battle seeds so that some fights fall in the validation split
            start = copy.deepcopy(champ['start']); start['seed'] = seed
            rows.append({'fight_id': f'test:{seed}', 'start': start})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pq.write_table(pa.Table.from_pylist(rows, pq.read_schema(FIGHTS).empty_table().select(['fight_id', 'start']).schema),
                           tmp / 'starts.parquet')
            run(PLAY, '--starts', tmp / 'starts.parquet', '--agent', 'teacher', '--sims', 20, '--workers', 1,
                '--out', tmp / 'teacher', '--worker', WORKER)
            n = data.collect([tmp / 'teacher/fights-0.parquet'], [tmp / 'teacher/search-0.parquet'], [39], tmp / 'rows', WORKER)
            self.assertEqual(n, 40)
            shard = duckdb.sql(f"select count(*), count(distinct fight_id), sum(has_policy::int), min(n_actions) "
                               f"from '{tmp}/rows/rows.parquet'").fetchone()
            self.assertEqual(shard[1], 40); self.assertGreater(shard[2], 0); self.assertGreaterEqual(shard[3], 1)
            run('-m', 'agents.combat.pv.train', '--data', tmp / 'rows/rows.parquet', '--out', tmp / 'model', '--epochs', 1)
            run(PLAY, '--starts', tmp / 'starts.parquet', '--agent', 'pv', '--model', tmp / 'model/model.onnx', '--sims', 16,
                '--workers', 1, '--out', tmp / 'pv', '--worker', WORKER)
            run(PLAY, '--starts', tmp / 'starts.parquet', '--agent', 'pv', '--model', tmp / 'model/model.onnx', '--sims', 16,
                '--rollout-mix', 1, '--workers', 1, '--out', tmp / 'mixed', '--worker', WORKER)
            self.assertIn('rollout_mix=1', duckdb.sql(f"select any_value(agent) from '{tmp}/mixed/fights-0.parquet'").fetchone()[0])
            summary = duckdb.sql(f"select count(*), count(mean_depth), min(max_depth), min(nodes) from '{tmp}/pv/decisions_stats-0.parquet'").fetchone()
            self.assertGreater(summary[0], 0); self.assertEqual(summary[0], summary[1]); self.assertGreaterEqual(summary[2], 1)
            # PV-played fights are valid combat_v4 rows: the encoder replays them.
            n = data.collect([tmp / 'pv/fights-0.parquet'], [tmp / 'pv/search-0.parquet'], [39], tmp / 'rows2', WORKER)
            self.assertGreater(n, 0)
            # Paired comparison of the two arms on the common fights.
            report = subprocess.run([sys.executable, ROOT / 'apps/pv/compare.py', tmp / 'teacher', tmp / 'pv', '--json'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(report.returncode, 0, report.stderr)
            report = report.stdout
            splits = json.loads(report)[str(tmp / 'pv')]['vs_baseline']
            self.assertEqual(splits[0]['split'], 'all'); self.assertGreater(splits[0]['n'], 0)
            self.assertEqual(sum(s['n'] for s in splits if s['split'].startswith('hp')), splits[0]['n'])


if __name__ == '__main__':
    unittest.main()
