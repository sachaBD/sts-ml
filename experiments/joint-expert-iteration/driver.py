#!/usr/bin/env python3
"""Bounded, staged joint expert iteration. Child jobs are managed runs with frozen inputs."""
import argparse
import datetime as dt
import json
import math
import os
import signal
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from runs.run import RUNS
from agents.overworld.value.core import read_runs

PYTHON = sys.executable
EXP = Path('experiments/joint-expert-iteration')
INITIAL_COMBAT = Path('runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_checkpoint.pt')
INITIAL_OVERWORLD = Path('runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt')
SIMS = '500,5000,10000,10000,20000'
DECIDE = ['rest','path','shop','event','neow']
DEADLINE = float('inf')


def log(message):
    line=f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line,flush=True)
    with (EXP/'RUNBOOK.md').open('a') as f: f.write('\n'+line+'\n')


def save(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)


def job(schema, name, cmd, inputs=()):
    matches=list(RUNS.glob(f'schema={schema}/date=*/id={name}/run.json'))
    if matches:
        if len(matches)!=1: raise RuntimeError(f'ambiguous job {name}')
        meta=json.loads(matches[0].read_text())
        if meta['status']!='done': raise RuntimeError(f'incomplete job {name}: inspect {matches[0]}')
        return matches[0].parent/'out', meta['run_id']
    argv=[PYTHON,'-m','runs.run',schema,name,'--no-compact']
    for source in inputs: argv+=['--input',str(source)]
    argv+=['--',*map(str,cmd)]
    log('START '+name+' '+ ' '.join(map(str,cmd)))
    remaining=DEADLINE-time.time()
    if remaining<=0: raise TimeoutError("overnight deadline reached")
    proc=subprocess.Popen(argv,start_new_session=True)
    try:
        result=proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise TimeoutError("overnight deadline reached; partial child outputs preserved")
    if result: raise subprocess.CalledProcessError(result,argv)
    match=next(RUNS.glob(f'schema={schema}/date=*/id={name}/run.json'))
    meta=json.loads(match.read_text())
    log('DONE '+name)
    return match.parent/'out',meta['run_id']


def evaluate(name, worker, combat, overworld, first, seeds, workers, rollout=False):
    cmd=[PYTHON,'apps/run_rl/play.py','--out','{out}','--first-seed',first,'--seeds',seeds,
         '--workers',workers,'--worker',worker,'--policy','net','--ckpt',overworld,
         '--decide',*DECIDE,'--sims',SIMS,'--overworld-record','--collection-id',name]
    if not rollout: cmd+=['--combat-leaf','value_net','--combat-weights',combat.with_name('value_weights.bin')]
    return job('overworld_v1',name,cmd,[overworld,*([] if rollout else [combat])])[0]


def paired(a,b):
    x={r['seed']:r for r in read_runs([a])}; y={r['seed']:r for r in read_runs([b])}
    if set(x)!=set(y): raise RuntimeError('paired evaluation seed mismatch')
    seeds=sorted(x); n=len(seeds)
    clears=lambda r: float(r['status']=='act_complete')
    score=lambda r: clears(r)+.1*r['final_hp']/max(1,r['steps'][0]['state']['max_hp'])
    d=[clears(x[s])-clears(y[s]) for s in seeds]
    v=[score(x[s])-score(y[s]) for s in seeds]
    se=lambda z: statistics.stdev(z)/math.sqrt(n) if n>1 else float('inf')
    return {'n':n,'candidate_clear':sum(clears(r) for r in x.values())/n,
        'incumbent_clear':sum(clears(r) for r in y.values())/n,
        'clear_diff':statistics.mean(d),'clear_se':se(d),'score_diff':statistics.mean(v),'score_se':se(v),
        'score_definition':'Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective',
        'uncertainty':'SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE'}


def promote(result):
    # Conservative provisional dev gate; final fresh confirmation remains mandatory.
    return (result['score_diff']>1.96*result['score_se']
            and result['clear_diff']>=-result['clear_se'])


def report(state, out):
    lines=['# Joint expert iteration report','',f"Status: {state['status']}",
        f"Elapsed: {(time.time()-state['started'])/3600:.2f} h",'',
        'Scope: Ironclad A20 Act 1. No evidence of full-game improvement.',
        'Easy combat: 500 guided-rollout sims; hard/elite/event/boss: neural leaves 5k/10k/10k/20k; 8 particles.',
        'Safety valve: neural combat switches to 5k guided rollout from turn 30 to avoid pathological defensive stalls; observed outcomes are not relabelled.',
        'Collection: one seeded random combat action within first 24 decisions; overworld epsilon 0.1. Evaluations: neither.',
        'Actual combats recorded in combat_v4, with exact replay verified before persistence; overworld macro traces in overworld_v1.',
        '',f"Selected combat checkpoint: `{state['combat']}`",f"Selected overworld checkpoint: `{state['overworld']}`",'',
        '| Comparison | n | Candidate clear | Incumbent clear | Paired difference ± SE | Promoted |',
        '|---|---:|---:|---:|---:|---|']
    for r in state['comparisons']:
        lines.append(f"| {r['name']} | {r['n']} | {r['candidate_clear']:.1%} | {r['incumbent_clear']:.1%} | {r['clear_diff']*100:+.2f} ± {r['clear_se']*100:.2f} pp | {r.get('promoted','—')} |")
    lines+=['','± is one standard error of paired run differences, not a 95% confidence interval. Development results were used for model selection; fresh final comparisons are marked fresh.',
        'Promotion uses a conservative composite-score gate (clear + 0.1 final-HP fraction); it is not a changed training objective.',
        'Combat training uses realized outcomes on played decision rows, with a stable run-seed validation split; old checkpoint architecture unchanged.',
        'Limitations: macro traces are public-state records, not full GameContext replay; hypothetical event samples are not actual transitions. Act-1 objective may harm later strength. Exploration outcomes describe exploratory behavior, not a pure greedy policy.',
        '',f"Completed collection batches: {len(state['collections'])}",
        f"Actual collection fights: {sum(x.get('actual_fights') or 0 for x in state['collections'])}; replay-verified: {sum(x.get('replay_verified_fights') or 0 for x in state['collections'])}; unsupported snapshot boundaries: {sum(x.get('unsupported_snapshot_fights') or 0 for x in state['collections'])}.",
        f"Recorded combat decisions: {sum(x.get('decision_rows') or 0 for x in state['collections'])}; random combat actions: {sum(x.get('random_actions') or 0 for x in state['collections'])}."]
    if 'error' in state: lines+=['', 'Failure: `'+state['error']+'`']
    text='\n'.join(lines)+'\n'; (EXP/'REPORT.md').write_text(text); (out/'REPORT.md').write_text(text)
    save(out/'summary.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    p.add_argument('--worker',required=True,type=Path);p.add_argument('--hours',type=float,default=9)
    p.add_argument('--workers',type=int,default=10);p.add_argument('--batch',type=int,default=1500)
    p.add_argument('--eval-seeds',type=int,default=400);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    statefile=a.out/'state.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else {
        'started':time.time(),'status':'running','combat':str(INITIAL_COMBAT),'overworld':str(INITIAL_OVERWORLD),
        'collections':[],'comparisons':[],'round':0}
    worker=a.out/'run_rl_worker'
    if not worker.exists(): shutil.copy2(a.worker,worker)
    deadline=state['started']+a.hours*3600
    global DEADLINE
    DEADLINE=deadline
    stamp='joint-1001'
    def persist(): save(statefile,state);report(state,a.out)
    persist()
    try:
        # Baseline against old combat, equal category budgets, both using expanded run decisions.
        initial=evaluate(stamp+'-initial',worker,INITIAL_COMBAT,INITIAL_OVERWORLD,910_000_000_000,a.eval_seeds,a.workers)
        rollout=evaluate(stamp+'-rollout',worker,INITIAL_COMBAT,INITIAL_OVERWORLD,910_000_000_000,a.eval_seeds,a.workers,True)
        if not state['comparisons']:
            r=paired(initial,rollout);r.update(name='initial neural vs rollout',promoted=False);state['comparisons'].append(r);persist()
        while time.time()<deadline-150*60:
            i=state['round']; old_combat=Path(state['combat']);old_overworld=Path(state['overworld'])
            # A: frozen pair collects, then combat alone updates.
            name=f'{stamp}-r{i:02d}-a'
            data,rid=job('combat_v4',name,[PYTHON,'apps/run_rl/collect.py','--out','{out}','--id',name,
                '--first-seed',920_000_000_000+i*100_000,'--seeds',a.batch,'--workers',a.workers,
                '--worker',worker,'--combat-weights',old_combat.with_name('value_weights.bin'),'--ckpt',old_overworld],
                [old_combat,old_overworld])
            if rid not in [x['combat_id'] for x in state['collections']]:
                sm=json.loads((data/'summary.json').read_text());state['collections'].append({'combat_id':rid,'overworld_out':sm['overworld_out'],'clear_rate':sm['clear_rate'],'actual_fights':sm.get('actual_fights'),'replay_verified_fights':sm.get('replay_verified_fights'),'unsupported_snapshot_fights':sm.get('unsupported_snapshot_fights'), 'decision_rows':sm.get('decision_rows'), 'random_actions':sm.get('recorded_random_actions'), 'rollout_fallback_fights':sm.get('rollout_fallback_fights')})
            persist()
            recent=[x['combat_id'] for x in state['collections'][-3:]]
            trained,_=job('value_net_v1',f'{stamp}-combat-r{i:02d}',[PYTHON,'-m','apps.run_rl.train_combat',
                '--out','{out}','--init',old_combat,'--data',*recent], [old_combat,*recent])
            candidate=trained/'value_checkpoint.pt'
            first=930_000_000_000+i*100_000
            base=evaluate(f'{stamp}-r{i:02d}-combat-base',worker,old_combat,old_overworld,first,a.eval_seeds,a.workers)
            test=evaluate(f'{stamp}-r{i:02d}-combat-test',worker,candidate,old_overworld,first,a.eval_seeds,a.workers)
            r=paired(test,base);chosen=promote(r);r.update(name=f'combat round {i}',promoted=chosen)
            state['comparisons'].append(r)
            if chosen:state['combat']=str(candidate)
            log('GATE '+json.dumps(r));persist()
            if time.time()>deadline-100*60:state['round']+=1;break
            # B: collect selected combat before teaching run V what that combat policy can achieve.
            name=f'{stamp}-r{i:02d}-b';combat=Path(state['combat'])
            data,rid=job('combat_v4',name,[PYTHON,'apps/run_rl/collect.py','--out','{out}','--id',name,
                '--first-seed',920_000_050_000+i*100_000,'--seeds',a.batch,'--workers',a.workers,
                '--worker',worker,'--combat-weights',combat.with_name('value_weights.bin'),'--ckpt',old_overworld],
                [combat,old_overworld])
            if rid not in [x['combat_id'] for x in state['collections']]:
                sm=json.loads((data/'summary.json').read_text());state['collections'].append({'combat_id':rid,'overworld_out':sm['overworld_out'],'clear_rate':sm['clear_rate'],'actual_fights':sm.get('actual_fights'),'replay_verified_fights':sm.get('replay_verified_fights'),'unsupported_snapshot_fights':sm.get('unsupported_snapshot_fights'), 'decision_rows':sm.get('decision_rows'), 'random_actions':sm.get('recorded_random_actions'), 'rollout_fallback_fights':sm.get('rollout_fallback_fights')})
            # Prefer data from this combat generation; small replay window limits stale-policy TD targets.
            recent=[x['overworld_out'] for x in state['collections'][-4:]]
            trained,_=job('value_net_v1',f'{stamp}-overworld-r{i:02d}',[PYTHON,'apps/run_rl/train.py',
                '--out','{out}/model.pt','--init',old_overworld,'--data',*recent,'--decay','.85','--lam','.7','--epochs',20],
                [old_overworld,combat,*recent])
            candidate=trained/'model.pt'
            # If combat did not change, re-use matching base eval; otherwise refresh it.
            if state['combat']==str(old_combat): baseline=base
            else:baseline=evaluate(f'{stamp}-r{i:02d}-overworld-base',worker,combat,old_overworld,first,a.eval_seeds,a.workers)
            test=evaluate(f'{stamp}-r{i:02d}-overworld-test',worker,combat,candidate,first,a.eval_seeds,a.workers)
            r=paired(test,baseline);chosen=promote(r);r.update(name=f'overworld round {i}',promoted=chosen)
            state['comparisons'].append(r)
            if chosen:state['overworld']=str(candidate)
            state['round']+=1;log('GATE '+json.dumps(r));persist()
            # Stop adding rounds when recent stage durations imply too little time.
        # Use spare budget for useful frozen-policy experience, not extra easy-fight search.
        # Only start this bounded batch if >=100 min remain (pilot batch ~30-40 min).
        if time.time()<deadline-100*60:
            name=stamp+'-final-data';combat=Path(state['combat']);overworld=Path(state['overworld'])
            data,rid=job('combat_v4',name,[PYTHON,'apps/run_rl/collect.py','--out','{out}','--id',name,
                '--first-seed',940_000_000_000,'--seeds',a.batch,'--workers',a.workers,
                '--worker',worker,'--combat-weights',combat.with_name('value_weights.bin'),'--ckpt',overworld],
                [combat,overworld])
            if rid not in [x['combat_id'] for x in state['collections']]:
                sm=json.loads((data/'summary.json').read_text());state['collections'].append({'combat_id':rid,'overworld_out':sm['overworld_out'],'clear_rate':sm['clear_rate'],'actual_fights':sm.get('actual_fights'),'replay_verified_fights':sm.get('replay_verified_fights'),'unsupported_snapshot_fights':sm.get('unsupported_snapshot_fights'), 'decision_rows':sm.get('decision_rows'), 'random_actions':sm.get('recorded_random_actions'), 'rollout_fallback_fights':sm.get('rollout_fallback_fights'),'reserved_for_next_training':True})
            persist()
        # Fresh seeds untouched by all training and previous development selection.
        # Pilot-based conservative ~2 worker-seconds/run wall allowance per paired run at 10 workers.
        fresh_n=max(100,min(800,int((deadline-time.time()-120)/4)))
        log(f'Fresh final confirmation n={fresh_n}')
        final=evaluate(stamp+'-fresh-final',worker,Path(state['combat']),Path(state['overworld']),950_000_000_000,fresh_n,a.workers)
        start=evaluate(stamp+'-fresh-start',worker,INITIAL_COMBAT,INITIAL_OVERWORLD,950_000_000_000,fresh_n,a.workers)
        r=paired(final,start);r.update(name='fresh final vs starting pair',promoted=False);state['comparisons'].append(r)
        state['status']='done';persist()
    except BaseException as e:
        state['status']='failed';state['error']=repr(e);log('FAILED '+repr(e));persist();raise

if __name__=='__main__':main()
