"""Four-game approved compatibility/smoke gate. No model selection or strength claims."""
import concurrent.futures as cf
import hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path

ROOT=Path.cwd()
RUN=ROOT/'runs/schema=combat_v4/date=2026-10-07/id=expert-champ-root-allocation-audit-v1'
OUT=RUN/'out/parity'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
NEW=OUT/'pv_worker'
if not NEW.exists():shutil.copy2(ROOT/'build/expert-champ-halving/agents/combat/pv/pv_worker',NEW)
OLD=ROOT/'runs/schema=combat_v4/date=2026-10-06/id=rollout-expert-iteration-v1/out/frozen/pv_worker'
INIT=ROOT/'runs/schema=combat_v4/date=2026-10-06/id=single-deck-demon-form-v1/out/iter015/model/model.onnx'
CHAMP=ROOT/'runs/schema=combat_v4/date=2026-10-07/id=expert-champ-recency-v1/out/frozen/incumbent/model.onnx'
DEV=ROOT/'runs/schema=combat_v4/date=2026-10-06/id=demon-form-decoupled-policy-v1/out/main/baseline/results.jsonl'
ref=next(json.loads(l) for l in DEV.open() if json.loads(l)['fight_id'].endswith(':dev:0'))
request={'fight_id':ref['fight_id'],'start':ref['fight']['start']}
manifest=dict(new_worker_sha256=sha(NEW),old_worker_sha256=sha(OLD),
    model_sha256={str(p):sha(p) for p in [INIT,CHAMP]},request=request,expected_games=4)
if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
else:(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
# The benchmark wrapper adds top-level fight_id/seconds; native fight_id remains checked.
assert ref['fight_id']==ref['fight']['fight_id'];ref.pop('fight_id')
if (OUT/'REPORT.json').exists():
    shutil.copy2(OUT/'REPORT.json',OUT/f'REPORT.previous-{time.time_ns()}.json')

def play(name,worker,model,flags):
    cmd=[str(worker),'play',str(model),'2000',*flags]
    intent=OUT/f'{name}.intent.json'
    if intent.exists():
        prior=json.loads(intent.read_text());assert prior['cmd']==cmd and prior['request']==request
        assert (OUT/f'{name}.result.json').exists(), 'orphan: no automatic retry'
        rec=json.loads((OUT/f'{name}.result.json').read_text());assert rec['returncode']==0
        obj=json.loads((OUT/f'{name}.stdout.json').read_text());assert obj['status']=='completed'
        print('REUSE',name,flush=True);return obj,rec
    intent.write_text(json.dumps(dict(cmd=cmd,request=request,at=time.time())))
    start=time.time();p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        text=True,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    try:out,err=p.communicate(json.dumps(request)+'\n',timeout=120)
    except BaseException:
        os.killpg(p.pid,signal.SIGKILL);p.communicate();raise
    (OUT/f'{name}.stdout.json').write_text(out);(OUT/f'{name}.stderr.txt').write_text(err)
    rec=dict(name=name,seconds=time.time()-start,returncode=p.returncode)
    (OUT/f'{name}.result.json').write_text(json.dumps(rec));print(json.dumps(rec),flush=True)
    assert p.returncode==0,err
    obj=json.loads(out);assert obj['status']=='completed',obj['status']
    return obj,rec

def strip(o):
    if isinstance(o,dict):return {k:strip(v) for k,v in o.items() if k!='seconds'}
    if isinstance(o,list):return [strip(v) for v in o]
    return o

def diff(a,b,path=''):
    if type(a)!=type(b):return path+':type'
    if isinstance(a,dict):
        if a.keys()!=b.keys():return path+':keys '+str(a.keys()^b.keys())
        for k in a:
            d=diff(a[k],b[k],path+'/'+k)
            if d:return d
    elif isinstance(a,list):
        if len(a)!=len(b):return path+':length'
        for i,(x,y) in enumerate(zip(a,b)):
            d=diff(x,y,path+'/'+str(i))
            if d:return d
    elif a!=b:return path+': '+repr(a)+' != '+repr(b)
    return None

report={'complete':False,'games':[]}
try:
    g1,t=play('legacy-new',NEW,INIT,[]);report['games'].append(t)
    d=diff(strip(ref),strip(g1));report['legacy_diff']=d;assert d is None,d
    with cf.ThreadPoolExecutor(2) as pool:
        f1=pool.submit(play,'hybrid-old',OLD,CHAMP,['--rollout-mix','0.5'])
        f2=pool.submit(play,'hybrid-new',NEW,CHAMP,['--rollout-mix','0.5'])
        old,t1=f1.result();new,t2=f2.result();report['games'] += [t1,t2]
    d=diff(strip(old),strip(new));report['hybrid_diff']=d;assert d is None,d
    sh,t=play('halving-smoke',NEW,CHAMP,['--rollout-mix','0.5','--root-halving','--halving-m','16','--gumbel-scale','0'])
    report['games'].append(t)
    report['smoke_won']=sh['fight']['won'];report['smoke_searches']=len(sh['search'])
    assert all(r.get('root_halving') is True and 'root_halving' in r['agent'] for r in sh['search'])
    assert all(r['simulations']==2000 for r in sh['search'])
    report['smoke_phase_flushes']=sum(r.get('phase_flushes',0) for r in sh['search'])
    report['smoke_simulations']=sum(r['simulations'] for r in sh['search'])
    report['complete']=True
finally:
    (OUT/'REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
