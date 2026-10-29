"""Real two-process learner recovery with a fake failing native worker; compare uninterrupted weights."""
import argparse
import json
import os
from pathlib import Path
import sys

from apps.gumbel_nn_combat_expert_iteration.supervisor import launch, jsonl

FAKE = '''#!/usr/bin/env python3
import json,struct,sys,time
from pathlib import Path
sample=Path(sys.argv[3]); games=[json.loads(s) for s in sys.stdin if s.strip()]
flag=sample.parents[1]/'injected-once'
if '/iter002/' in str(sample) and '/test/' not in str(sample) and not flag.exists():
    flag.touch(); sample.write_bytes(b'interrupted')
    print('workers: failed: CUDA: unknown error',file=sys.stderr,flush=True)
    sys.exit(1)
with sample.open('wb') as f:
    for g in games:
        won=':a:' in g['fight_id']; x=[0.0]*(3*68)
        x[0]=1; x[68]=x[136]=6
        f.write(struct.pack('<iif',3,2,float(won)))
        f.write(struct.pack('<204f',*x));f.write(struct.pack('<2f',.5,.5))
        print(json.dumps(dict(fight_id=g['fight_id'],won=won,decisions=1,capped=False,finished_unix=time.time(),seconds=0)),flush=True)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    worker=out/'fake_worker';worker.write_text(FAKE);worker.chmod(0o755)
    (out/'starts.json').write_text(json.dumps([dict(fight_id=d,start=dict(seed=1)) for d in ['a','b']]))
    for filename,deck in [('dev.jsonl','held'),('monitor.jsonl','a')]:
        (out/filename).write_text(json.dumps(dict(fight_id=f'fixture:{deck}:validation:0',start=dict(seed=1)))+'\n')
    checkpoints=[]
    for mode in ['recovered','control']:
        root=out/mode;root.mkdir();training=root/'training'
        if mode=='control':
            control=out/'control_worker';control.write_text(FAKE.replace("and not flag.exists():", "and False:"));control.chmod(0o755)
        command=[sys.executable,'-m','agents.combat.nn.loop','--start',str(out/'starts.json'),'--out',str(training),
            '--play',str(control if mode=='control' else worker),'--iterations','2','--games','2','--balanced-replay',
            '--steps','2','--batch','2','--dim','8','--layers','1','--heads','1','--device','cpu',
            '--workers','1','--parallel','1','--scheduler','local','--gpu-batch','2','--gpu-tokens','3',
            '--test',str(out/'dev.jsonl'),'--test-in',str(out/'monitor.jsonl'),'--test-at','0','--test-every','1','--prune-replay']
        (root/'command.json').write_text(json.dumps(command));launch(command,root,retries=2)
        assert [r['iteration'] for r in jsonl(training/'log.jsonl') if 'iteration' in r]==[0,1,2]
        assert len(list(training.glob('.interrupted-*')))==(1 if mode=='recovered' else 0)
        assert len(jsonl(training/'test/iter002/results.jsonl'))==1
        assert len(jsonl(training/'test/iter002/in-sample/results.jsonl'))==1
        assert not list((training/'test').rglob('samples-*.bin'))
        import torch
        checkpoints.append(torch.load(training/'iter002/model.pt',weights_only=True,map_location='cpu'))
    import torch
    for k,v in checkpoints[0]['net'].items():torch.testing.assert_close(v,checkpoints[1]['net'][k],rtol=0,atol=0)
    torch.testing.assert_close(checkpoints[0]['rng'],checkpoints[1]['rng'],rtol=0,atol=0)
    assert checkpoints[0]['optimizer']['state'] and checkpoints[1]['optimizer']['state']
    report=dict(passed=True,recovered_processes=2,retries=1,uninterrupted_weights='bitwise identical',rng='bitwise identical',
                primary_and_in_sample_separate=True,partial_preserved=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
