"""Oracle turn-target rows must pass the unchanged canonical encode path."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import pyarrow.parquet as pq
from agents.combat.pv.test_turn_search import stable
ROOT=Path(__file__).resolve().parents[3]
WORKER=Path(os.environ.get('PV_TURN_WORKER',ROOT/'build/pv-phase2/agents/combat/pv/pv_worker'))
MODEL=ROOT/'runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r13/model/model.onnx'
BASELINE=ROOT/'build/frozen/pv_worker.pimc-136e4ebe28f7'
FIXTURE=Path(__file__).with_name('testdata')/'oracle-replay-history.parquet'
@unittest.skipUnless(WORKER.exists() and MODEL.exists(),'needs worker/local r13 artifact')
class TurnTargetsTest(unittest.TestCase):
    def play(self,worker,*flags):
        fight=pq.ParquetFile(FIXTURE).read().to_pylist()[1]
        p=subprocess.run([str(worker),'play',str(MODEL),'16','--oracle','--turn-search',*flags],
            input=json.dumps(fight)+'\n',text=True,capture_output=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=600)
        self.assertEqual(p.returncode,0,p.stderr)
        return json.loads(p.stdout)
    def test_targets_encode(self):
        out=self.play(WORKER,'--turn-targets','--explore','--sample-turns')
        self.assertEqual(out['status'],'completed')
        searched={row['step'] for row in out['search']}
        for i,row in enumerate(out['turn_stats']):
            end=out['turn_stats'][i+1]['step'] if i+1<len(out['turn_stats']) else len(out['fight']['actions'])
            if not row['fallback']: self.assertTrue(set(range(row['step'],end))<=searched)
        self.assertTrue(out['search'])
        for row in out['search']:
            self.assertGreater(sum(child['visits'] for child in row['children']),0)
            self.assertTrue(0<=row['root_value']<=100)
        fight=dict(out['fight'],search=out['search'])
        with tempfile.TemporaryDirectory() as tmp:
            shard=Path(tmp)/'rows.parquet'
            p=subprocess.run([str(WORKER),'encode',str(shard)],input=json.dumps(fight)+'\n',text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            table=pq.ParquetFile(shard).read()
            self.assertEqual(table.num_rows,len(fight['actions']))
            self.assertTrue(any(table['has_policy'].to_pylist()))
    @unittest.skipUnless(BASELINE.exists(),'needs immutable pre-target worker')
    def test_eval_turn_output_unchanged(self):
        self.assertEqual(json.dumps(stable(self.play(WORKER))),json.dumps(stable(self.play(BASELINE))))
    def test_e1_rejected(self):
        p=subprocess.run([str(WORKER),'play','/missing','1','--oracle','--turn-search','--turn-targets'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('E >= 2',p.stderr)
if __name__=='__main__':unittest.main()
