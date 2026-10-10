"""Frozen single-branch Champ continuation; no silent retries or partial comparisons.

freeze: audit relevant prior seeds and snapshot runtime artifacts, no gameplay.
run: smoke -> incumbent monitor -> sequential collect/train/monitor -> conditional final.
Reads only its run directory after freeze, except immutable replay shards (hash checked).
"""
import argparse
import concurrent.futures as cf
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
MASK = (1 << 64)-1


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def log(*x): print(now(),*x,flush=True)


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)


class Halt(RuntimeError): pass


def rng_state(seed):
    def murmur(x):
        x ^= x >> 33; x=x*(-49064778989728563 & MASK)&MASK
        x ^= x >> 33; x=x*(-4265267296055464877 & MASK)&MASK
        return (x ^ (x >> 33)) & MASK
    first=murmur(seed if seed else 1<<63)
    return dict(counter=0,seed0=first,seed1=murmur(first))


def generate(base,ns,split,n,used):
    rows=[]
    for i in range(n):
        seed=int.from_bytes(hashlib.sha256(f'{ns}:{split}:{i}'.encode()).digest()[:8],'little')
        if seed in used: raise Halt('seed collision; change namespace with recorded approval, not substitution')
        used.add(seed);r=copy.deepcopy(base);r.update(fight_id=f'{ns}:{split}:{i}',seed_kind=split)
        r['start'].update(seed=seed,misc_rng=rng_state(seed),potion_rng=rng_state(seed));rows.append(r)
    return rows


def seed_audit(repo):
    # These dates contain the source corpus, single-deck experiments and all descendants.
    # Also follow Phase5's explicit source/selection exclusion inventory outside these roots.
    roots=[repo/f'runs/schema=combat_v4/date=2026-10-0{d}' for d in (5,6)]
    p5=roots[1]/'id=rollout-expert-iteration-v1/out'
    files={p.resolve() for root in roots for p in root.rglob('*.parquet')}
    files.update(Path(p).resolve() for p in read(p5/'manifest.json')['starts_meta']['excluded_seed_files'])
    used=set(); inventory=[]
    for k,p in enumerate(sorted(files)):
        pf=pq.ParquetFile(p);names=pf.schema_arrow.names
        cols=[n for n in ('seed','start.seed') if n.split('.')[0] in names]
        if not cols:continue
        n=0
        for batch in pf.iter_batches(batch_size=4096,columns=cols):
            for r in batch.to_pylist():
                for v in (r.get('seed'),(r.get('start') or {}).get('seed')):
                    if v is not None:used.add(int(v));n+=1
        inventory.append(dict(path=str(p),bytes=p.stat().st_size,seed_observations=n))
        if k%200==0:log('seed parquet scan',k,'/',len(files),'unique',len(used))
    pattern=re.compile(r'"seed"\s*:\s*(\d+)')
    for root in roots:
        for p in sorted(root.rglob('*')):
            if p.suffix not in ('.json','.jsonl') or not p.is_file():continue
            n=0
            with p.open(errors='strict') as f:
                for line in f:
                    for s in pattern.findall(line):used.add(int(s));n+=1
            if n:inventory.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,seed_observations=n))
    return used,inventory


def freeze(run):
    run=run.resolve();out=run/'out';cfg=read(out/'config.json')
    if (out/'manifest.json').exists():raise Halt('already frozen; no overwrite')
    cores=sorted(os.sched_getaffinity(0))[:cfg['workers']]
    if len(cores)!=10 or cfg['workers']!=10:raise Halt('exactly ten available cores/workers required')
    os.sched_setaffinity(0,cores)
    used,inventory=seed_audit(REPO);prior_count=len(used)
    p5=REPO/'runs/schema=combat_v4/date=2026-10-06/id=rollout-expert-iteration-v1/out'
    base=read(p5/'starts-u1.json')[0]
    assert (base['start']['hp'],base['start']['max_hp'],base['start']['potions'])==(34,52,[1,1])
    ns=run.name.removeprefix('id=')
    starts={f'train-{u}':generate(base,ns,f'train-{u}',cfg['batch_fights'],used) for u in range(1,cfg['updates']+1)}
    for split,n in [('monitor',cfg['monitor_fights']),('final',cfg['final_fights']),('smoke',2)]:
        starts[split]=generate(base,ns,split,n,used)
    assert len(used)==prior_count+sum(map(len,starts.values()))
    cores=sorted(os.sched_getaffinity(0))[:cfg['workers']]
    if len(cores)!=10 or cfg['workers']!=10:raise Halt('exactly ten available cores/workers required')
    frozen=out/'frozen';frozen.mkdir(exist_ok=False)
    # Copy dependencies as a minimal importable package. Trainer runs with cwd=frozen.
    for name in ['agents/__init__.py','agents/combat/__init__.py','agents/combat/pv/__init__.py',
                 'agents/combat/pv/model.py','agents/combat/pv/data.py','agents/combat/pv/train.py',
                 'agents/combat/pv/recency.py','agents/combat/pv/train_recency.py']:
        dst=frozen/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dst)
    shutil.copy2(Path(__file__),frozen/'controller.py')
    shutil.copy2(cfg['worker'],frozen/'pv_worker')
    incumbent=frozen/'incumbent';incumbent.mkdir()
    for n,h in cfg['incumbent_files_sha256'].items():
        src=Path(cfg['incumbent_model'])/n
        if sha(src)!=h:raise Halt('incumbent changed')
        shutil.copy2(src,incumbent/n)
    if sha(frozen/'pv_worker')!=cfg['worker_sha256']:raise Halt('worker changed')
    write(out/'starts.json',starts);write(out/'seed-audit.json',dict(unique_prior_seeds=prior_count,files=inventory,
        scope='combat_v4 dates 2026-10-05/06 plus explicit Phase5 source/selection exclusions; includes overnight reserved JSON starts',
        limitation='not a scan of all historical unrelated encounters; new SHA256 namespace, exact exclusion of known relevant seeds'))
    cfg.update(status='READY',seed_manifest=str(out/'starts.json'),cores=cores,max_runtime_seconds=6600,
               final_reserve_seconds=1200,game_timeout_seconds=900,trainer_timeout_seconds=900,
               worker=str(frozen/'pv_worker'),incumbent_model=str(incumbent),
               launch_gates=['runtime verification','two fresh smoke games and encoding check'])
    write(out/'config.json',cfg)
    hashes={str(p.relative_to(out)):sha(p) for p in frozen.rglob('*') if p.is_file()}
    for n in ('config.json','starts.json','seed-audit.json'):hashes[n]=sha(out/n)
    for spec in cfg['anchor_shards']+cfg['initial_online_shards']:
        if sha(spec['path'])!=spec['sha256']:raise Halt('replay changed')
    write(out/'manifest.json',dict(frozen_at=now(),hashes=hashes,config_sha256=sha(out/'config.json')))
    write(out/'state.json',dict(status='READY',phase='Seed/code freeze complete; awaiting smoke and run',update=0,failure=None))
    log('FROZEN',sum(map(len,starts.values())),'starts; excluded',prior_count,'cores',cores)


class Ledger:
    def __init__(self,path):
        self.path=Path(path);self.intents={};self.results={};self.lock=threading.Lock()
        if self.path.exists():
            raw=self.path.read_text()
            if raw and not raw.endswith('\n'):raise Halt('partial ledger tail; manual review, no automatic repair/retry')
            for line in raw.splitlines():
                r=json.loads(line);key=r['key'];kind=r['kind']
                if kind=='intent':
                    if key in self.intents:raise Halt('duplicate intent')
                    self.intents[key]=r
                elif kind=='result':
                    if key not in self.intents or key in self.results:raise Halt('invalid result ledger ordering')
                    self.results[key]=r
                else:raise Halt('unknown ledger row')
        self.f=self.path.open('a')
    def append(self,r):
        with self.lock:
            key=r['key'];kind=r['kind']
            if kind=='intent':
                if key in self.intents:raise Halt('refusing repeated dispatch')
                self.intents[key]=r
            elif kind=='result':
                if key not in self.intents or key in self.results:raise Halt('invalid repeated result')
                self.results[key]=r
            else:raise Halt('invalid ledger kind')
            self.f.write(json.dumps(r)+'\n');self.f.flush();os.fsync(self.f.fileno())
    def reusable(self,key,request_sha):
        if key not in self.intents:return None
        if self.intents[key]['request_sha256']!=request_sha:raise Halt('request changed on resume')
        r=self.results.get(key)
        if not r or r['status']!='completed' or not r.get('line'):raise Halt('prior incomplete/error/cap: no automatic gameplay retry')
        return r


def paired(a,b):
    if set(a)!=set(b) or not a:raise Halt('incomplete/mismatched paired comparison')
    ids=sorted(a);d=[int(b[k]['won'])-int(a[k]['won']) for k in ids];n=len(d);gap=sum(d)/n
    se=math.sqrt(sum((x-gap)**2 for x in d)/(n*(n-1))) if n>1 else None
    plus=d.count(1);minus=d.count(-1);disc=plus+minus
    p=min(1.,2*sum(math.comb(disc,k) for k in range(min(plus,minus)+1))/2**disc) if disc else 1.
    return dict(n=n,wins=sum(b[k]['won'] for k in ids),incumbent_wins=sum(a[k]['won'] for k in ids),gap=gap,
                approx95=[gap-1.96*se,gap+1.96*se] if se is not None else None,
                candidate_only=plus,incumbent_only=minus,exact_mcnemar_p=p)


class Controller:
    def __init__(self,run):
        self.run=Path(run).resolve();self.out=self.run/'out';self.cfg=read(self.out/'config.json')
        self.man=read(self.out/'manifest.json');self.starts=read(self.out/'starts.json')
        self.state=read(self.out/'state.json');self.active=set();self.lock=threading.Lock();self.stop=threading.Event()
        self.ledger=Ledger(self.out/'ledger.jsonl');self.deadline=time.time()+self.cfg['max_runtime_seconds']
        self.env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.python=str(Path(sys.executable).absolute());self.frozen=self.out/'frozen'
    def verify(self):
        for p,h in self.man['hashes'].items():
            if sha(self.out/p)!=h:raise Halt(f'frozen artifact changed: {p}')
        for s in self.cfg['anchor_shards']+self.cfg['initial_online_shards']:
            if sha(s['path'])!=s['sha256']:raise Halt('immutable replay changed')
    def save(self,phase=None,**kwargs):
        if phase:self.state['phase']=phase
        self.state.update(kwargs);self.state['updated_at']=now();write(self.out/'state.json',self.state)
    def kill(self,p):
        if p and p.poll() is None:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    def abort(self):
        self.stop.set()
        with self.lock: active=list(self.active)
        for p in active:self.kill(p)
    def execute(self,cmd,input=None,timeout=900,cwd=None,logfile=None):
        p=None
        with self.lock:
            if self.stop.is_set():raise Halt('dispatch stopped')
            p=subprocess.Popen(cmd,stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=logfile or subprocess.PIPE,stderr=subprocess.STDOUT if logfile else subprocess.PIPE,
                text=True,start_new_session=True,env=self.env,cwd=cwd)
            self.active.add(p)
        try:
            stdout,stderr=p.communicate(input,timeout=max(1,min(timeout,self.deadline-time.time())))
            return p.returncode,stdout,stderr
        except BaseException:
            self.kill(p);p.communicate();raise
        finally:
            with self.lock:self.active.discard(p)
    def game(self,key,cmd,row,meta,request_sha):
        if self.stop.is_set():return None
        self.ledger.append(dict(kind='intent',key=key,at=now(),request_sha256=request_sha,cmd=cmd,**meta))
        t=time.time();rec=dict(kind='result',key=key,fight_id=row['fight_id'],status='error',won=False,**meta)
        try:
            rc,out,err=self.execute(cmd,json.dumps({'fight_id':row['fight_id'],'start':row['start']})+'\n',self.cfg['game_timeout_seconds'])
            rec.update(returncode=rc,stderr=(err or '')[-1500:])
            if rc:raise Halt(f'worker exited {rc}')
            obj=json.loads(out)
            if obj.get('status')!='completed':
                if obj.get('status')=='capped':rec['status']='capped'
                raise Halt(f'noncompleted game: {obj.get("status")}')
            f=obj['fight'];agent=f.get('agent') or (obj.get('search') or [{}])[0].get('agent','')
            if 'rollout_mix=0.500000' not in agent or 'sims=2000' not in agent:raise Halt('wrong agent flags')
            if f['fight_id']!=row['fight_id'] or f['start']!=row['start']:raise Halt('worker start/ID mismatch')
            rec.update(status='completed',won=bool(f['won']),agent=agent,
                line=dict(fight_id=f['fight_id'],start=f['start'],actions=f['actions'],won=f['won'],final_hp=f['final_hp'],
                    search=[dict(step=s['step'],root_value=s['root_value'],children=s['children']) for s in obj.get('search',[])]))
        except Exception as e:
            rec['error']=f'{type(e).__name__}: {e}';self.stop.set()
        rec.update(seconds=time.time()-t,at=now());self.ledger.append(rec)
        return rec
    def cell(self,name,rows,model,stage,update=0):
        self.save(f'{name}: {len(rows)} fights',update=update)
        # Include model contents in request identity, not merely its path.
        model=Path(model);model_hash={p.name:sha(p) for p in model.glob('model.onnx*')}
        if not model_hash:raise Halt('missing inference model')
        cmd=[self.cfg['worker'],'play',str(model/'model.onnx'),'2000','--rollout-mix','0.5']
        done={};pending=[]
        for row in rows:
            key=f'{name}|{row["fight_id"]}';request=hashlib.sha256(json.dumps([cmd,model_hash,row],sort_keys=True).encode()).hexdigest()
            prior=self.ledger.reusable(key,request)
            if prior:done[row['fight_id']]=prior
            else:pending.append((key,cmd,row,dict(stage=stage,update=update,cell=name),request))
        log('CELL',name,'cached',len(done),'new',len(pending))
        it=iter(pending)
        with cf.ThreadPoolExecutor(max_workers=self.cfg['workers']) as pool:
            active={}
            def submit():
                item=next(it,None)
                if item is not None:active[pool.submit(self.game,*item)]=item[0]
            for _ in range(self.cfg['workers']):submit()
            while active:
                finished,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
                for fu in finished:
                    active.pop(fu);r=fu.result()
                    if r is not None:done[r['fight_id']]=r
                    if not r or r['status']!='completed':self.abort()
                if self.stop.is_set():
                    self.abort()
                    # Drain and persist all already-dispatched results; don't replace them.
                    for fu in active:
                        try:fu.result()
                        except Exception:pass
                    raise Halt(f'{name}: error/cap/interruption; inspect ledger, no retries')
                if time.time()>=self.deadline:
                    self.abort();raise Halt('global deadline reached')
                for _ in finished:submit()
                if len(done)%50==0:log(name,len(done),'/',len(rows))
        if len(done)!=len(rows) or any(r['status']!='completed' for r in done.values()):raise Halt('incomplete cell')
        return done
    def encode(self,results,path):
        path=Path(path);lines=[results[k]['line'] for k in sorted(results)]
        text=''.join(json.dumps(r)+'\n' for r in lines);source_sha=hashlib.sha256(text.encode()).hexdigest()
        marker=path.with_suffix('.encode.json')
        if path.exists():
            if not marker.exists() or read(marker)!=dict(source_sha256=source_sha,rows_sha256=sha(path)):
                raise Halt('partial/changed encoding; no automatic overwrite')
            return
        tmp=path.with_suffix('.tmp')
        if tmp.exists():raise Halt('partial encoding temporary file; review required')
        rc,out,err=self.execute([self.cfg['worker'],'encode',str(tmp)],text,600)
        if rc:raise Halt(f'encoding failed: {err}')
        table=pq.read_table(tmp,columns=['fight_id','won'])
        if set(table['fight_id'].to_pylist())!=set(results):raise Halt('encoded fight mismatch')
        if table.num_rows!=sum(len(r['actions']) for r in lines):raise Halt('encoded decision count mismatch')
        for r in table.to_pylist():
            if bool(r['won'])!=results[r['fight_id']]['won']:raise Halt('encoded outcome mismatch')
        os.replace(tmp,path);write(marker,dict(source_sha256=source_sha,rows_sha256=sha(path)))
    def train(self,u,init,online):
        d=self.out/f'update{u}';model=d/'model';manifest=d/'train-manifest.json';split=d/'split.json'
        specs=self.cfg['anchor_shards']+online;mapping={}
        for spec in specs:
            if sha(spec['path'])!=spec['sha256']:raise Halt('training shard changed')
            ids=set(pq.read_table(spec['path'],columns=['fight_id'])['fight_id'].to_pylist())
            if ids & mapping.keys():raise Halt('duplicate replay fights')
            mapping.update({i:'train' for i in ids})
        cfg=dict(init=str(Path(init)/'model.pt'),init_sha256=sha(Path(init)/'model.pt'),split=str(split),
            anchor=self.cfg['anchor_shards'],online=online,update=u,half_life=self.cfg['half_life_updates'],
            steps=self.cfg['optimizer_steps'],rng_seed=self.cfg['rng_seed'])
        if model.exists():
            if not (model/'complete.json').exists() or not (model/'train.json').exists():raise Halt('partial training checkpoint; review required')
            report=read(model/'train.json')
            if report['config']!=cfg or read(split)!=mapping or read(manifest)!=cfg:raise Halt('training resume contract differs')
            for n,h in report['files_sha256'].items():
                if sha(model/n)!=h:raise Halt('trained model hash mismatch')
            if read(model/'complete.json')['model_sha256']!=sha(model/'model.pt'):raise Halt('completion marker mismatch')
            return model
        write(split,mapping);write(manifest,cfg)
        self.save(f'Update {u}: training/export',update=u)
        with (d/'train.log').open('w') as lf:
            rc,_,_=self.execute([self.python,'-m','agents.combat.pv.train_recency','--manifest',str(manifest),
                '--out',str(model),'--device','cpu'],timeout=self.cfg['trainer_timeout_seconds'],cwd=self.frozen,logfile=lf)
        if rc:raise Halt(f'training failed {rc}; see {d/"train.log"}')
        # Same validation path as resume before model is used for games.
        return self.train(u,init,online)
    def report(self):
        curves=read(self.out/'curve.json') if (self.out/'curve.json').exists() else []
        lines=['# Expert Champ recency continuation',f'Updated {now()}; status **{self.state.get("status")}**.',
            f'Phase: {self.state.get("phase")}',f'Failure: {self.state.get("failure")}',
            'Single training branch; comparison is to frozen incumbent, not a causal recency ablation.',
            '', '| Update | Candidate | Incumbent | Paired gap |','|---|---:|---:|---:|']
        for r in curves:lines.append(f'| {r["update"]} | {r["wins"]}/{r["n"]} | {r["incumbent_wins"]}/{r["n"]} | {100*r["gap"]:+.1f}pp |')
        lines+=['','Monitor results are selection-consumed. All curves use the same starts; not independent confirmations.']
        if (self.out/'final-summary.json').exists():lines+=['','## Independent final', '```json',json.dumps(read(self.out/'final-summary.json'),indent=2),'```']
        else:lines+=['','Final not played/completed; do not claim final superiority.']
        lines+=['','## Actual coverage and cost']
        by={}
        for r in self.ledger.results.values():by.setdefault(r.get('cell','unknown'),[]).append(r)
        for cell,rr in by.items():
            times=[r['seconds'] for r in rr];wins=sum(r['won'] for r in rr if r['status']=='completed')
            lines.append(f'- {cell}: {len(rr)} results, {wins} wins; mean/median elapsed seconds {statistics.mean(times):.2f}/{statistics.median(times):.2f}.')
        lines+=['Timing is subprocess elapsed wall time, not CPU time; concurrent-load variability unquantified.',
                'One fixed deck/HP and one training RNG; no general-deck or training-repeatability claim.']
        (self.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def run_all(self):
        self.verify();os.sched_setaffinity(0,self.cfg['cores'])
        smoke_report=read(self.out/'smoke/model/train.json')
        if smoke_report['adam_step_after']!=smoke_report['adam_step_before']+2 or smoke_report['onnx_max_abs_diff']>1e-3:
            raise Halt('trainer smoke gate failed')
        def handler(signum,frame):self.abort();raise Halt(f'signal {signum}')
        signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
        olddeadline=self.state.get('deadline_epoch')
        if olddeadline:self.deadline=olddeadline
        self.save('Runtime checks',status='RUNNING',failure=None,deadline_epoch=self.deadline,
                  started_at=self.state.get('started_at',now()),cores=self.cfg['cores'])
        incumbent=Path(self.cfg['incumbent_model'])
        try:
            if time.time()>=self.deadline:raise Halt('original global deadline already expired')
            smoke=self.cell('smoke',self.starts['smoke'],incumbent,'smoke')
            self.encode(smoke,self.out/'gameplay-smoke.parquet')
            write(self.out/'gameplay-smoke-passed.json',dict(at=now(),games=len(smoke)))
            ref=self.cell('monitor-incumbent',self.starts['monitor'],incumbent,'monitor')
            online=list(self.cfg['initial_online_shards']);model=incumbent;curves=[]
            for u in range(1,self.cfg['updates']+1):
                estimate=self.state.get('last_update_seconds',600)*1.25
                if time.time()+estimate>self.deadline-self.cfg['final_reserve_seconds']:
                    self.save('Training budget stop; preserve final reserve',budget_stop_before_update=u);break
                t=time.time();d=self.out/f'update{u}';d.mkdir(exist_ok=True)
                records=self.cell(f'collect-{u}',self.starts[f'train-{u}'],model,'collect',u)
                self.encode(records,d/'rows.parquet')
                online.append(dict(path=str(d/'rows.parquet'),generation=u,sha256=sha(d/'rows.parquet')))
                model=self.train(u,model,online)
                evaluated=self.cell(f'monitor-{u}',self.starts['monitor'],model,'monitor',u)
                metric=dict(update=u,model=str(model),**paired(ref,evaluated));curves.append(metric)
                write(self.out/'curve.json',curves);self.save(last_update_seconds=time.time()-t);self.report()
                log('MONITOR',json.dumps(metric))
            if not curves:raise Halt('no complete scheduled checkpoint evaluation')
            best=max(curves,key=lambda r:(r['wins'],-r['update']))
            if best['wins']<=best['incumbent_wins']:
                self.save('No challenger exceeds incumbent monitor wins; final preserved',status='COMPLETE_NO_CHALLENGER');return
            selected=Path(best['model']);selection=dict(update=best['update'],model=str(selected),
                files_sha256={p.name:sha(p) for p in selected.glob('model.*')},selected_at=now())
            spg=statistics.mean([r['seconds'] for r in self.ledger.results.values() if r['status']=='completed'])
            if time.time()+2*self.cfg['final_fights']*spg/self.cfg['workers']*1.4+300>self.deadline:
                self.save('Insufficient time for complete final; final preserved',status='COMPLETE_FINAL_DEFERRED');return
            if (self.out/'selection.json').exists():
                prior=read(self.out/'selection.json')
                if prior['model']!=selection['model'] or prior['files_sha256']!=selection['files_sha256']:raise Halt('final selection changed')
            else:write(self.out/'selection.json',selection)
            a=self.cell('final-incumbent',self.starts['final'],incumbent,'final')
            b=self.cell('final-challenger',self.starts['final'],selected,'final',best['update'])
            summary=dict(selected_update=best['update'],**paired(a,b))
            summary['confirmed_improvement']=summary['approx95'][0]>0
            summary['batches']=[paired({r['fight_id']:a[r['fight_id']] for r in self.starts['final'][i:i+100]},
                                       {r['fight_id']:b[r['fight_id']] for r in self.starts['final'][i:i+100]}) for i in range(0,len(a),100)]
            write(self.out/'final-summary.json',summary)
            self.save('All scheduled stages complete',status='COMPLETE');log('FINAL',json.dumps(summary))
        except BaseException as e:
            self.abort();self.save('Stopped; review failure and ledger before any resume',status='FAILED',failure=f'{type(e).__name__}: {e}');raise
        finally:
            self.report();self.ledger.f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','run'])
    p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    lock=(a.run/'out/controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('another controller owns this run')
    if a.command=='freeze':freeze(a.run)
    else:Controller(a.run).run_all()


if __name__=='__main__':main()
