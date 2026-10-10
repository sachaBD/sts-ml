"""Root-allocation diagnostic on 200 existing incumbent training fights. No gameplay/training."""
import collections,json,hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from agents.combat.pv.data import Shard
from agents.combat.pv.model import PolicyValue,NAMES

ROOT=Path('runs/schema=combat_v4/date=2026-10-07/id=expert-champ-recency-v1/out')
OUT=Path('runs/schema=combat_v4/date=2026-10-07/id=expert-champ-root-allocation-audit-v1/out')
OUT.mkdir(parents=True,exist_ok=True)
records={}
for line in (ROOT/'ledger.jsonl').open():
    r=json.loads(line)
    if r.get('kind')=='result' and r.get('cell')=='collect-1':
        assert r['status']=='completed'
        records[r['fight_id']]=r['line']
assert len(records)==200
path=ROOT/'update1/rows.parquet';meta=pq.read_table(path,columns=['fight_id','step','moves','has_policy','turn']).to_pylist()
sh=Shard(path,False,{fid:'train' for fid in records})
model=ROOT/'frozen/incumbent/model.pt';ck=torch.load(model,map_location='cpu',weights_only=False)
net=PolicyValue(ck['width'],ck['value_activation']);net.load_state_dict(ck['state_dict']);net.eval();torch.set_num_threads(2)
search={(fid,s['step']):s for fid,f in records.items() for s in f['search']}
result=[]
for start in range(0,len(sh.rows),128):
    idx=sh.rows[start:start+128];b=sh.batch(idx)
    with torch.inference_mode():value,logits=net(*(b[0][n] for n in NAMES));probs=logits.softmax(-1)
    for z,i in enumerate(idx):
        r=meta[int(i)]
        if not r['has_policy']:continue
        s=search[(r['fight_id'],r['step'])];children={c['action']:c for c in s['children']}
        # worker.cpp emits only visited children; encoded legal menu includes zero-visit actions.
        assert set(children)<=set(r['moves'])
        n=len(r['moves']);vis=np.array([children.get(m,{}).get('visits',0) for m in r['moves']]);q=np.array([children.get(m,{}).get('value',-np.inf) for m in r['moves']]);p=probs[z,:n].numpy()
        assert vis.sum()==2000 and (vis>=0).all()
        assert np.allclose(b[2][z,:n].numpy(),vis/vis.sum(),atol=1e-6)
        pick=int(vis.argmax());prior=int(p.argmax());tokens=b[0]['actions'][z,:n].numpy()
        row=dict(fight_id=r['fight_id'],step=r['step'],turn=r['turn'],legal=n,
            visits_le2_fraction=float(np.mean(vis<=2)),visits_le10_fraction=float(np.mean(vis<=10)),
            any_le2=bool(np.any(vis<=2)),any_unvisited=bool(np.any(vis==0)),
            prior_mass_le2=float(p[vis<=2].sum()),prior_mass_le10=float(p[vis<=10].sum()),
            chosen_visit_fraction=float(vis[pick]/vis.sum()),prior_overturned=bool(pick!=prior),
            prior_overturned_distinct_token=bool(pick!=prior and not np.array_equal(tokens[pick],tokens[prior])),
            chosen_prior=float(p[pick]),chosen_prior_rank=int(1+(p>p[pick]).sum()),
            moderate_prior_under10=bool(np.any((p>=.05)&(vis<=10))),
            qmax_differs_from_visitmax=bool(q.argmax()!=pick),
            qmax_barely_visited=bool(vis[q.argmax()]<=10),
            largest_visit_fraction=float(vis.max()/2000))
        result.append(row)
by=collections.defaultdict(list)
for r in result:by[r['fight_id']].append(r)
summary={}
for k in result[0]:
    if k in ('fight_id','step','turn'):continue
    vals=np.array([np.mean([r[k] for r in rr]) for rr in by.values()]);mean=float(vals.mean());se=float(vals.std(ddof=1)/len(vals)**.5)
    summary[k]=dict(fight_balanced_mean=mean,conditional_se=se,approx95=[mean-1.96*se,mean+1.96*se])
report=dict(fights=len(by),searched_states=len(result),total_states=len(meta),
    legal_action_histogram=dict(sorted(collections.Counter(r['legal'] for r in result).items())),
    legal_over16=sum(r['legal']>16 for r in result),metrics=summary,
    prior_model=str(model),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    caveat='Descriptive root allocation, not proof neglected actions are better. Q means have no recorded sampling variance. Approx95 intervals are normal intervals across fight means conditional on fixed model; not training RNG uncertainty. Same-token distinction is representational, not a proof of physical action equivalence.')
pq.write_table(pa.Table.from_pylist(result),OUT/'root-states.parquet',compression='zstd')
(OUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
