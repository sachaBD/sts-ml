"""Focused worker checks; local r09 model and immutable pre-change worker required."""
import json
import os
from pathlib import Path
import subprocess
import unittest
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(os.environ.get('PV_TURN_WORKER', ROOT / 'build/pv/agents/combat/pv/pv_worker'))
MODEL = ROOT / 'runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/model/model.onnx'
BASELINES = sorted((ROOT / 'build/frozen').glob('pv_worker.liquid-fix-*'))
BASELINE = Path(os.environ.get('PV_TURN_BASELINE', str(BASELINES[0]) if BASELINES else '/missing'))
FIXTURE = Path(__file__).with_name('testdata') / 'oracle-replay-history.parquet'


def stable(result):
    if isinstance(result, dict):
        return {k: stable(v) for k, v in result.items() if k != 'seconds'}
    if isinstance(result, list):
        return [stable(v) for v in result]
    return result


@unittest.skipUnless(WORKER.exists() and MODEL.exists(), 'needs built worker/local r09 model')
class TurnWorkerTest(unittest.TestCase):
    def run_worker(self, worker, *flags):
        fight = pq.ParquetFile(FIXTURE).read().to_pylist()[1]
        done = subprocess.run([str(worker), 'play', str(MODEL), '64', *flags],
                              input=json.dumps(fight) + '\n', text=True, capture_output=True,
                              env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), timeout=600)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    @unittest.skipUnless(BASELINE.exists(), 'needs immutable pre-change worker')
    def test_no_flag_stable_line(self):
        before = self.run_worker(BASELINE, '--oracle')
        after = self.run_worker(WORKER, '--oracle')
        self.assertEqual(json.dumps(stable(before), separators=(',', ':')),
                         json.dumps(stable(after), separators=(',', ':')))
        self.assertNotIn('turn_stats', after)

    def test_requires_oracle(self):
        done = subprocess.run([str(WORKER), 'play', '/not/a/model', '1', '--turn-search'],
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('requires --oracle', done.stderr)

    def test_explicit_fallback_and_no_macro_policy(self):
        out = self.run_worker(WORKER, '--oracle', '--turn-search', '--turn-max-sequences', '1')
        self.assertEqual(out['status'], 'completed')  # worker verifies the entire canonical replay
        rows = out['turn_stats']
        self.assertTrue(any(row['fallback'] for row in rows))
        searched = {row['step'] for row in out['search']}
        for i, row in enumerate(rows):
            end = rows[i + 1]['step'] if i + 1 < len(rows) else len(out['fight']['actions'])
            if row['fallback']:
                self.assertTrue(row['reason'])
                self.assertEqual(row['root_children'], 0)
            else:
                self.assertFalse(searched.intersection(range(row['step'], end)), 'invented macro policy rows')


if __name__ == '__main__':
    unittest.main()
