"""Read-only Phase5 absorption audit. Run from repo root; writes only its new run directory."""
import collections, hashlib, json, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from agents.combat.pv.data import Shard
from agents.combat.pv.model import PolicyValue, NAMES
from agents.combat.pv.train import policy_confidence, evaluate

BASE=Path('runs/schema=combat_v4/date=2026-10-06/id=rollout-expert-iteration-v1/out')
OUT=Path('runs/schema=combat_v4/date=2026-10-07/id=expert-champ-absorption-audit-v1/out')
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
reports=[load(BASE/f'strong/update{u}/model/train.json') for u in (1,2,3)]
paths=list(dict.fromkeys(reports[-1]['old_shards']+reports[-1]['new_shards']))
manifest=load(BASE/'manifest.json')
checks={}
for name,h in manifest['frozen_sha256'].items():
    checks['frozen/'+name]=sha(BASE/'frozen'/name)==h
for p,h in manifest['old_shards'].items(): checks[p]=sha(p)==h
models=[Path(reports[0]['init'])]+[BASE/f'strong/update{u}/model/model.pt' for u in (1,2,3)]
for u,r in enumerate(reports):
    checks[f'init{u+1}']=sha(r['init'])==r['init_sha256']
    for f,h in r['files_sha256'].items(): checks[f'update{u+1}/{f}']=sha(BASE/f'strong/update{u+1}/model'/f)==h
assert all(checks.values()),checks
meta={}
for p in paths:
    rows=pq.read_table(p,columns=['fight_id','step','turn','won','n_actions','has_policy','policy_target']).to_pylist()
    for r in rows:
        n=r['n_actions']; r['confidence']=max(0.,min(1.,(n*max(r['policy_target'],default=0)-1)/max(n-1,1))) if r['has_policy'] else 0.
    meta[p]=rows

# Exact deterministic row draws, matching Pool.sample RNG call order. No neural inference needed.
exposure=[]
for u,rep in enumerate(reports,1):
    pools=[]
    for half in ('old','new'):
        fights=[]
        for p in rep[half+'_shards']:
            by=collections.defaultdict(list)
            for i,r in enumerate(meta[p]): by[r['fight_id']].append(i)
            fights += [(p,f,by[f]) for f in sorted(by)]
        pools.append(fights)
    rngs=[np.random.default_rng(np.random.SeedSequence([0,u,h])) for h in (0,1)]
    stats=[collections.Counter(),collections.Counter()]; seen=[collections.Counter(),collections.Counter()]
    for step in range(1000):
        picked=[]
        for h in (0,1):
            inds=rngs[h].integers(len(pools[h]),size=32); rr=[]
            for k in inds:
                p,f,ii=pools[h][k]; i=ii[rngs[h].integers(len(ii))]; rr.append(meta[p][i]); seen[h][(p,i)]+=1
            picked.append(rr)
        den=max(1,sum(r['has_policy'] for rr in picked for r in rr))
        for h,rr in enumerate(picked):
            for r in rr:
                s=stats[h]; s['samples']+=1; s['policy_samples']+=r['has_policy']; s['forced_samples']+=r['n_actions']==1
                s['confidence_sum']+=r['confidence']; s['policy_coefficient_mass']+=r['confidence']/den
                s['loss_samples']+=not r['won']; s['low_conf_policy']+=r['has_policy'] and r['confidence']<.2
    exposure.append({'update':u,'old':dict(stats[0]),'new':dict(stats[1]),'distinct_old_rows':len(seen[0]),'distinct_new_rows':len(seen[1]),'new_mass_share':stats[1]['policy_coefficient_mass']/sum(s['policy_coefficient_mass'] for s in stats)})
    print('EXPOSURE',json.dumps(exposure[-1]),flush=True)

nets=[]
for p in models:
    ck=torch.load(p,map_location='cpu',weights_only=False)
    net=PolicyValue(ck['width'],ck['value_activation']); net.load_state_dict(ck['state_dict']); net.eval(); nets.append(net)
results=[]; summaries=[]; parity=[]
for u,p in enumerate(reports[-1]['new_shards'],1):
    split={r['fight_id']:'train' for r in meta[p]}; s=Shard(Path(p),False,split)
    for start in range(0,len(s.rows),128):
        idx=s.rows[start:start+128]; b=s.batch(idx); inputs,y,target,hp,_=b
        with torch.inference_mode():
            conf=policy_confidence(target,inputs['actions'][...,260]!=0)*hp
            assert np.allclose(conf.numpy(),[meta[p][int(i)]['confidence'] for i in idx],atol=2e-6)
            outputs=[net(*(inputs[n] for n in NAMES)) for net in nets]
        for j,(v,logits) in enumerate(outputs):
            lp=logits.log_softmax(-1); probs=lp.exp(); kl=(target*(target.clamp_min(1e-30).log()-lp)).sum(-1)
            brier=((v-y)/100).square()
            if start==0:
                with torch.inference_mode(): vl,pl,_,nn,np_=evaluate(nets[j],b,'cpu',flat_policy_weighting=True)
                assert abs(vl.item()-brier.sum().item())<1e-4
                assert abs(pl.item()-(-(target*lp).sum(-1)*conf).sum().item())<1e-4
                parity.append({'batch':u,'model':j,'loss_reconstruction':True})
            for z,i in enumerate(idx):
                r=meta[p][int(i)]; n=r['n_actions']; pol=target[z,:n].numpy().astype(float)
                # Identical action-token logits must tie. Per-row floor is a lower bound, not global capacity floor.
                tokens=inputs['actions'][z,:n].numpy(); groups=collections.defaultdict(list)
                for a,tok in enumerate(tokens): groups[tok.tobytes()].append(a)
                qstar=pol.copy()
                for group in groups.values(): qstar[group]=pol[group].mean()
                floor=float(np.sum(pol*np.log(np.maximum(pol,1e-30)/np.maximum(qstar,1e-30))))
                best=pol>=pol.max()-1e-8; arg=int(logits[z,:n].argmax()); equiv=next(g for g in groups.values() if arg in g)
                results.append({'batch':u,'model':j,'fight_id':r['fight_id'],'step':r['step'],'turn':r['turn'],'won':r['won'],'has_policy':r['has_policy'],'n_actions':n,'confidence':float(conf[z]),'kl':float(kl[z]),'floor':floor,'brier':float(brier[z]),'value':float(v[z])/100,'top1':float(best[arg]),'class_hit':float(best[equiv].any()),'p_best':float(probs[z,:n].numpy()[best].sum()),'duplicate_tokens':len(groups)<n})
    print('SCORED batch',u,'states',len(s.rows),flush=True)
pq.write_table(pa.Table.from_pylist(results),OUT/'per-state.parquet',compression='zstd')

def summarize(rr):
    # Fight-uniform, state-uniform within fight. Policy metrics average only policy-bearing states.
    by=collections.defaultdict(list)
    for r in rr: by[r['fight_id']].append(r)
    fm=[]
    for fid,rows in by.items():
        pol=[r for r in rows if r['has_policy']]
        f={'fight_id':fid,'brier':float(np.mean([r['brier'] for r in rows])),'value':float(np.mean([r['value'] for r in rows])),'won':float(rows[0]['won'])}
        if pol:
            for k in ('kl','floor','top1','class_hit','p_best','confidence','duplicate_tokens'): f[k]=float(np.mean([r[k] for r in pol]))
        fm.append(f)
    agg={'fights':len(fm),'rows':len(rr),'policy_rows':sum(r['has_policy'] for r in rr)}
    for k in fm[0]:
        if k!='fight_id':
            vals=np.array([f[k] for f in fm if k in f]); agg[k]=float(vals.mean()); agg[k+'_se']=float(vals.std(ddof=1)/len(vals)**.5) if len(vals)>1 else None
    return agg,fm
for u in (1,2,3):
    for j in range(4):
        rr=[r for r in results if r['batch']==u and r['model']==j]; agg,fm=summarize(rr)
        summaries.append({'batch':u,'model':j,**agg})
        (OUT/f'fight-metrics-b{u}-m{j}.json').write_text(json.dumps(fm))
        print('FIT',json.dumps(summaries[-1]),flush=True)
# Paired absorption changes on same fight; conditional SE, not independent decision-level inference.
deltas=[]
for u in (1,2,3):
    pre=load(OUT/f'fight-metrics-b{u}-m{u-1}.json'); post=load(OUT/f'fight-metrics-b{u}-m{u}.json')
    assert [f['fight_id'] for f in pre]==[f['fight_id'] for f in post]
    for k in ('kl','top1','class_hit','brier'):
        d=np.array([b[k]-a[k] for a,b in zip(pre,post)]); se=d.std(ddof=1)/len(d)**.5
        deltas.append({'batch':u,'metric':k,'n_fights':len(d),'post_minus_pre':float(d.mean()),'approx95':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)]})
provenance={'checks':checks,'model_paths':[str(p) for p in models],'model_sha256':[sha(p) for p in models],'source_sha256':{p:sha(p) for p in ['agents/combat/pv/model.py','agents/combat/pv/data.py','agents/combat/pv/train.py',__file__]},'training_reports_sha256':[sha(BASE/f'strong/update{u}/model/train.json') for u in (1,2,3)],'parity':parity,'caveat':'Imported model/data/train modules were not snapshotted by Phase5; current source compatibility checked by checkpoint load and loss reconstruction, not proof of historical identity. No new training or gameplay. Endpoint fit is in-sample; precollection checkpoints have not trained on their upcoming batch, but targets are generated by their own hybrid search.'}
(OUT/'results.json').write_text(json.dumps({'exposure':exposure,'fit':summaries,'paired_deltas':deltas,'provenance':provenance},indent=2))
print('COMPLETE',flush=True)
