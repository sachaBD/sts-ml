"""Native coverage checks without touching final-test outcomes. Artifacts belong under --out."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from environments.combat.starts import fights
from apps.gumbel_nn_combat_expert_iteration.make_fights import load_bases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',action='store_true');a=p.parse_args()
    c=tomllib.loads(a.config.read_text());out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    bases=load_bases(c['deck']);ev=c['evaluation'];worker=Path(c['run']['worker']).resolve()
    train=[dict(fight_id=name,start=row['start']) for name,row in bases.items()]
    dev_all=[json.loads(s) for s in Path(ev['starts']).read_text().splitlines()]
    final_all=[json.loads(s) for s in Path(ev['final_starts']).read_text().splitlines()]
    def first_by_deck(rows):
        return list({r['fight_id'].split(':')[1]:dict(fight_id=r['fight_id'].split(':')[1],start=r['start']) for r in reversed(rows)}.values())
    check=train+first_by_deck(dev_all)+first_by_deck(final_all)
    (out/'check-starts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in check))
    env=dict(os.environ)
    packages=Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    env['LD_LIBRARY_PATH']=':'.join(map(str,[packages/'tensorrt_libs',*sorted((packages/'nvidia').glob('*/lib'))]))+':'+env.get('LD_LIBRARY_PATH','')
    with (out/'check-starts.jsonl').open() as source,(out/'check-results.jsonl').open('w') as target,(out/'check.log').open('w') as errors:
        subprocess.run([str(worker),'check'],stdin=source,stdout=target,stderr=errors,env=env,check=True)
    assert len((out/'check-results.jsonl').read_text().splitlines())==1400
    # Final test: import/encode only above. Never search, play, or inspect outcomes.
    probe=[]
    for base in train+first_by_deck(dev_all):
        probe+=fights(base,f'preflight:{0}:{base["fight_id"]}',1)
    (out/'starts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in probe))
    old=Path('runs/schema=gumbel_nn_combat_v1/date=2026-10-28/id=gumbel-nn-champ-fifty-20261028/out/training/iter040/model.onnx').resolve()
    command=[str(worker),'play',str(old),str(out/'samples-0.bin'),'--threads','9','--parallel','32','--groups','2',
             '--scheduler','local','--sims','8','--gumbel-scale','1','--rollout-mix','0.5','--timeout-ms','60000']
    if a.gpu:
        from agents.combat.nn.model import Net,export,export_static
        import torch
        pt=torch.load(old.with_suffix('.pt'),map_location='cpu',weights_only=True)
        net=Net(**pt['spec']);net.load_state_dict(pt['net']);net.eval()
        export(net,out/'model.onnx');export_static(net,out/'model-gpu.onnx',512,64)
        for b in [256,32]:export_static(net,out/f'model-gpu-b{b}.onnx',b,64)
        command[2]=str(out/'model.onnx')
        command+=['--gpu-model',str(out/'model-gpu.onnx'),'--gpu-batch','512','--gpu-tokens','64','--gpu-wait-us','100',
                  '--gpu-bank',','.join(str(out/f'model-gpu-b{b}.onnx') for b in [256,32])]
    start=time.time();print('native probe:',len(probe),'games; GPU',a.gpu,flush=True)
    with (out/'starts.jsonl').open() as source,(out/'results.jsonl').open('w') as target,(out/'worker.log').open('w') as errors:
        subprocess.run(command,stdin=source,stdout=target,stderr=errors,env=env,check=True)
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    assert len(rows)==len(probe)==1200 and {r['fight_id'] for r in rows}=={r['fight_id'] for r in probe}
    assert all('capped' in r and 'finished_unix' in r for r in rows)
    report=dict(games=len(rows),seconds=time.time()-start,capped=sum(r['capped'] for r in rows),
                no_decisions=[r['fight_id'] for r in rows if not r['decisions']],final_outcomes_played=0)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
