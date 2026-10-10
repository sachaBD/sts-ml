"""Local-artifact real-turn CLI/replay regression; no fabricated visit targets."""
import json
import os
from pathlib import Path
import subprocess
import unittest
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3]
WORKER=Path(os.environ.get('PV_REAL_TURN_WORKER',ROOT/'build/pv-real-turn/agents/combat/pv/pv_worker'))
MODEL=ROOT/'runs/schema=combat_v4/date=2026-10-05/id=champ-diag-D5/model/model.onnx'
@unittest.skipUnless(WORKER.exists() and MODEL.exists(),'requires local worker/D5 model')
class RealTurnTest(unittest.TestCase):
    def test_fallback_is_real2000_and_replay(self):
        fight=pq.ParquetFile(Path(__file__).with_name('testdata')/'oracle-replay-history.parquet').read().to_pylist()[1]
        p=subprocess.run([str(WORKER),'play',str(MODEL),'2000','--real-turn','--particles','8',
                          '--real-turn-max-leaves','1','--real-turn-max-sequences','1'],
            input=json.dumps(fight)+'\n',text=True,capture_output=True,timeout=600,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        self.assertEqual(p.returncode,0,p.stderr);out=json.loads(p.stdout)
        self.assertEqual(out['status'],'completed')
        failed={row['step'] for row in out['real_turn_stats'] if row['fallback']}
        self.assertTrue(failed)
        self.assertEqual(failed,{row['step'] for row in out['search']})
        for row in out['search']:self.assertEqual(row['simulations'],2000)
        for row in out['real_turn_stats']:
            if row['fallback']:self.assertIsNone(row['reveal_fraction']);self.assertIsNone(row['mean_leaf_depth'])
    def test_incompatible_flags(self):
        for flag in ('--oracle','--turn-search','--explore','--policy-only','--sample-turns'):
            with self.subTest(flag=flag):
                p=subprocess.run([str(WORKER),'play','/missing','2000','--real-turn',flag],capture_output=True,text=True)
                self.assertNotEqual(p.returncode,0);self.assertIn('--real-turn requires',p.stderr)
if __name__=='__main__':unittest.main()
