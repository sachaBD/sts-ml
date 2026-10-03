"""Managed combat collection with a separately managed linked overworld trace."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from runs.run import RUNS
import pyarrow.parquet as pq

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--first-seed', type=int, required=True)
    p.add_argument('--seeds', type=int, required=True)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--worker', required=True)
    p.add_argument('--combat-weights', required=True)
    p.add_argument('--ckpt', required=True)
    a=p.parse_args()
    cmd=[sys.executable,'-m','runs.run','overworld_v1',a.id,'--no-compact',
         '--input',a.ckpt,'--input',a.combat_weights,'--',sys.executable,'apps/run_rl/play.py',
         '--out','{out}','--first-seed',str(a.first_seed),'--seeds',str(a.seeds),
         '--workers',str(a.workers),'--worker',a.worker,'--policy','net','--ckpt',a.ckpt,
         '--eps','.1','--decide','rest','path','shop','event','neow','--sims','500,5000,10000,10000,20000',
         '--combat-leaf','value_net','--combat-weights',a.combat_weights,'--combat-explore',
         '--combat-out',str(a.out),'--overworld-record','--collection-id',a.id]
    matches=list(RUNS.glob(f'schema=overworld_v1/date=*/id={a.id}/run.json'))
    if matches and json.loads(matches[0].read_text())['status']=='done':
        child=matches[0].parent
    else:
        subprocess.run(cmd,check=True)
        child=next(RUNS.glob(f'schema=overworld_v1/date=*/id={a.id}'))
    summary=json.loads((child/'out/summary.json').read_text())
    fights=[r for f in a.out.glob('fights-*.parquet')
            for r in pq.ParquetFile(f).read(columns=['actions','explored']).to_pylist()]
    search=[r for f in a.out.glob('search-*.parquet') for r in pq.ParquetFile(f).read(columns=['agent']).to_pylist()]
    summary['actual_fights']=len(fights)
    summary['decisions']=sum(len(r['actions']) for r in fights)
    summary['recorded_random_actions']=sum(sum(r['explored']) for r in fights)
    summary['search_rows']=len(search)
    summary['rollout_rescue_decisions']=sum(r['agent'].endswith(' rescue') for r in search)
    summary['overworld_run_id']=json.loads((child/'run.json').read_text())['run_id']
    summary['overworld_out']=str(child/'out')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
