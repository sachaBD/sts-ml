"""One full 500-step GPU fit on training-deck probe rows; weights are discarded, not warm-started."""
import argparse
import json
from pathlib import Path
import resource
import time
import tomllib

import torch

from agents.combat.nn.model import Net
from agents.combat.nn.replay import Replay
from agents.combat.nn.train import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    c=tomllib.loads(a.config.read_text());names={d['name'] for d in c['deck']}
    replay=Replay([a.probe/'samples-0.bin'])
    # Probe contains dev coverage too; those rows MUST NOT enter even this discarded fit.
    replay.groups={k:v for k,v in replay.groups.items() if k in names}
    assert set(replay.groups)==names and len(names)==1000
    replay.available={k:len(v) for k,v in replay.groups.items()};replay.selected=dict(replay.available);replay.drawn={k:0 for k in names}
    pt=Path('runs/schema=gumbel_nn_combat_v1/date=2026-10-28/id=gumbel-nn-champ-fifty-20261028/out/training/iter040/model.pt')
    old=torch.load(pt,map_location='cpu',weights_only=True)
    torch.set_num_threads(2);torch.manual_seed(0)
    net=Net(**old['spec']).cuda();net.load_state_dict(old['net'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4);optimizer.load_state_dict(old['optimizer'])
    original=replay.batches;micro=[]
    def tracked(size):
        for x,p,w in original(size):
            micro.append((len(w),x.shape[1]));yield x,p,w
    replay.batches=tracked
    torch.cuda.reset_peak_memory_stats();start=time.time()
    train(net,optimizer,replay,None,None,500,2048,'cuda')
    torch.cuda.synchronize();seconds=time.time()-start
    assert sum(replay.drawn.values())==500*2048
    report=dict(steps=500,effective_batch=2048,seconds=seconds,groups=len(replay.groups),eligible_rows=len(replay),
        draws=sum(replay.drawn.values()),draws_per_deck=[min(replay.drawn.values()),max(replay.drawn.values())],
        microbatches=len(micro),largest_attention_cells=max(b*t*t for b,t in micro),max_tokens=max(t for b,t in micro),
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        precision='FP32 inputs, BF16 autocast',weights_saved=False,dev_rows_trained=0,
        caveat='One probe batch; this is not a full ten-window disk-I/O or RSS upper bound.')
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
