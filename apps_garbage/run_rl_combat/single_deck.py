"""Fresh-network single-deck expert iteration. Reuses combat_loop stages and PV learner.

Fixed loadout/HP; 200 teacher training fights, 100 monitoring seeds, fresh learner
batches and a tapering teacher replay contribution. See experiment PLAN.md.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq

from apps.human_champ import bench
from apps.run_rl.combat_loop import encode, play, read_results, sha, stage, write_starts

PY = sys.executable
DECK = '7a9ada48-a4d6-41fc-a85b-0147618df00a'


def select_loadout(source, deck_id):
    base=next(r for r in source if r['deck_id']==deck_id and r['seed_kind']=='human')
    if base['start']['potions']!=[1,1]:
        raise ValueError('fixed loadout must have no potions')
    if deck_id==DECK and (base['start']['hp'],base['start']['max_hp'])!=(41,75):
        raise ValueError('unexpected original fixed loadout')
    return base


def teacher_count(iteration, bootstrap=200):
    return max(0, bootstrap - 20 * (iteration - 1))


def generate(base, namespace, split, n, used):
    rows = []
    for i in range(n):
        seed = int.from_bytes(hashlib.sha256(f'{namespace}:{split}:{i}'.encode()).digest()[:8], 'little')
        while seed in used:
            seed = (seed+1) & bench.MASK
        used.add(seed)
        r = copy.deepcopy(base)
        r.update(fight_id=f'{namespace}:{split}:{i}', seed_kind=split)
        r['start'].update(seed=seed, misc_rng=bench.rng_state(seed), potion_rng=bench.rng_state(seed))
        rows.append(r)
    return rows


def wilson(wins, n):
    if not n:
        return None
    z = 1.959963984540054
    p = wins / n
    den = 1 + z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]


def score(starts, baseline, candidate):
    # Same physical deck, independent seed trials: cluster by seed, not by the common deck.
    independent = [dict(r, deck_id=r['fight_id']) for r in starts]
    a, b = read_results(baseline), read_results(candidate)
    pairs = [(a[r['fight_id']]['fight']['won'], b[r['fight_id']]['fight']['won']) for r in independent
             if all(d.get(r['fight_id'], {}).get('status') == 'completed' for d in (a, b))]
    n = len(pairs)
    if not n:
        raise ValueError('no paired completed monitoring starts')
    d = [int(b)-int(a) for a, b in pairs]
    diff = sum(d)/n
    se = math.sqrt(sum((x-diff)**2 for x in d)/(n*(n-1))) if n > 1 else None
    wins = sum(b for a, b in pairs)
    return dict(n=n, selected=len(starts), wins=wins, win_rate=wins/n, confidence_95=wilson(wins, n),
                mcts_win_rate=sum(a for a,b in pairs)/n, gap=diff, gap_se=se,
                statuses=json.loads((candidate/'summary.json').read_text())['statuses'])


def plot(out, curves, title='Fixed Barricade deck, 41/75 HP, no potions'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    x = [r['training_fights'] for r in curves]
    y = [100*r['win_rate'] for r in curves]
    lo = [100*r['confidence_95'][0] for r in curves]
    hi = [100*r['confidence_95'][1] for r in curves]
    axes[0].plot(x,y,'o-', label='Fresh learned search, 2k')
    axes[0].fill_between(x,lo,hi,alpha=.2,label='95% Wilson interval (monitoring only)')
    reference = 100*curves[0]['mcts_win_rate']
    axes[0].axhline(reference, color='black', linestyle='--', label='MCTS20k reference')
    axes[0].set(ylabel='Win rate (%)', ylim=(0, 102), title=title)
    axes[0].legend(fontsize=8)
    gap = [100*r['gap'] for r in curves]
    errors = [100*1.96*(r['gap_se'] or 0) for r in curves]
    axes[1].errorbar(x,gap,yerr=errors,fmt='o-',capsize=3)
    axes[1].axhline(0,color='black',linestyle='--')
    axes[1].set(ylabel='Paired gap to MCTS (points)', xlabel='Cumulative generated training fights')
    fig.tight_layout()
    fig.savefig(out/'curve.png',dpi=150)
    plt.close(fig)
    (out/'curve.json').write_text(json.dumps(curves,indent=2))
    lines=['iteration,training_fights,teacher_replay_fights,self_replay_fights,win_rate,ci_low,ci_high,mcts_win_rate,gap,gap_se,n']
    for r in curves:
        lines.append(','.join(map(str,[r['iteration'],r['training_fights'],r['teacher_replay_fights'],r['self_replay_fights'],
                                      r['win_rate'],*r['confidence_95'],r['mcts_win_rate'],r['gap'],r['gap_se'],r['n']])))
    (out/'curve.csv').write_text('\n'.join(lines)+'\n')


def training_point(iteration, collection_results, replay_ids, teacher_results):
    completed=[r for r in collection_results.values() if r.get('status')=='completed']
    all_results={**teacher_results, **collection_results}
    replay=[all_results[fid] for fid in replay_ids]
    def stats(rows):
        n=len(rows);wins=sum(bool(r['fight']['won']) for r in rows)
        return dict(n=n,wins=wins,win_rate=wins/n if n else None,confidence_95=wilson(wins,n))
    return dict(iteration=iteration,batch=stats(completed),replay=stats(replay))


def plot_training(out, points, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    x=[p['iteration'] for p in points]
    for key,label in [('batch','New exploratory self-play batch'),('replay','Active training replay (including teacher)')]:
        y=[100*p[key]['win_rate'] for p in points]
        lo=[100*p[key]['confidence_95'][0] for p in points]
        hi=[100*p[key]['confidence_95'][1] for p in points]
        ax.plot(x,y,'o-',label=label)
        ax.fill_between(x,lo,hi,alpha=.15)
    ax.set(xlabel='Self-play round',ylabel='Completed-fight win rate (%)',ylim=(0,100),title=title)
    ax.legend(fontsize=8)
    fig.text(.5,.01,'Shading: 95% Wilson bounds, descriptive training data only. Replay overlaps and mixes older policies.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    temp=out/'training-win-rate.tmp.png';fig.savefig(temp,dpi=150);temp.replace(out/'training-win-rate.png')
    plt.close(fig)
    temp=out/'training-win-rate.tmp.json';temp.write_text(json.dumps(points,indent=2));temp.replace(out/'training-win-rate.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--deck-id',default=DECK,help='fixed human loadout to specialize on')
    p.add_argument('--selection',type=Path,required=True,help='prior selection starts to exclude')
    p.add_argument('--worker',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--namespace',default='barricade-fresh-v1')
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--monitor',type=int,default=100)
    p.add_argument('--updates',type=int,default=5)
    p.add_argument('--batch-fights',type=int,default=100)
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--sims',type=int,default=2000)
    p.add_argument('--teacher-sims',type=int,default=20000)
    p.add_argument('--bootstrap-epochs',type=int,default=10)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if not 1<=a.workers<=10 or min(a.bootstrap,a.monitor,a.updates,a.batch_fights,a.sims,a.teacher_sims,a.epochs,a.bootstrap_epochs,a.eval_every)<1:
        p.error('positive budgets and 1–10 workers required')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    source=pq.read_table(a.source).to_pylist()
    base=select_loadout(source,a.deck_id)
    hp,max_hp=base['start']['hp'],base['start']['max_hp']
    title=f'Fixed deck {a.deck_id[:8]}, {hp}/{max_hp} HP, no potions'
    config.update(worker_sha=sha(a.worker),source_sha=sha(a.source),selection_sha=sha(a.selection),
                  architecture='width64-sigmoid',fresh_network=True,hp=hp,max_hp=max_hp)
    # Preserve compatibility with existing default-deck resume configurations.
    if a.deck_id==DECK:config.pop('deck_id')
    cp=out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text())
        # Allow extending the iteration horizon; seeds/stages for completed updates are unchanged.
        new=dict(config);old_updates=old.pop('updates');new_updates=new.pop('updates')
        if old!=new or new_updates<old_updates:
            raise ValueError('resume may only increase updates; all other settings must match')
    cp.write_text(json.dumps(config,indent=2))
    worker=out/'frozen/pv_worker'
    if not worker.exists():
        worker.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.worker,worker)
    if sha(worker)!=config['worker_sha']:raise ValueError('frozen worker checksum mismatch')
    used={r['start']['seed'] for r in source}
    used.update(r['start']['seed'] for r in pq.read_table(a.selection).to_pylist())
    bootstrap=generate(base,a.namespace,'bootstrap',a.bootstrap,used)
    monitor=generate(base,a.namespace,'monitor',a.monitor,used)
    # Reserve final seeds without playing or labelling them during development.
    final=generate(base,a.namespace,'final',600,used)
    write_starts(bootstrap,out/'bootstrap.parquet');write_starts(monitor,out/'monitor.parquet');write_starts(final,out/'final-reserved.parquet')
    mapping={r['fight_id']:'train' for r in bootstrap};mapping.update({r['fight_id']:'val' for r in monitor})
    split=out/'split.json';split.write_text(json.dumps(mapping))
    teacher=play(out,'bootstrap-play',out/'bootstrap.parquet',None,worker,a.teacher_sims,a.workers,teacher=True)
    teacher_rows=encode(out,'bootstrap-rows',teacher,worker)
    reference=play(out,'reference-monitor',out/'monitor.parquet',None,worker,a.teacher_sims,a.workers,teacher=True)
    validation_rows=encode(out,'monitor-rows',reference,worker)

    def train(where,shards,ids,init=None,epochs=3,lr=.0003):
        subset=where/'train-fights.json';subset.write_text(json.dumps(ids))
        cmd=[PY,'-m','agents.combat.pv.train','--data',*shards,validation_rows,'--split-manifest',split,
             '--train-fights',subset,'--states-per-fight',64,'--flat-policy-weighting','--width',64,
             '--epochs',epochs,'--lr',lr,'--grad-clip',1,'--device',a.device,'--stream','--out',where/'model']
        if init is None:cmd+=['--value-activation','sigmoid']
        else:cmd+=['--init',init/'model.pt','--resume-optimizer']
        stage(where,'train',cmd)
        return where/'model'

    boot=out/'bootstrap';boot.mkdir(exist_ok=True)
    boot_results=read_results(teacher)
    completed_boot=[r['fight_id'] for r in bootstrap if boot_results.get(r['fight_id'],{}).get('status')=='completed']
    if len(completed_boot)<.95*a.bootstrap:raise RuntimeError('too many incomplete bootstrap fights')
    model=train(boot,[teacher_rows],completed_boot,epochs=a.bootstrap_epochs,lr=.001)
    ev=play(boot,'eval',out/'monitor.parquet',model,worker,a.sims,a.workers)
    curves=[dict(score(monitor,reference,ev),iteration=0,training_fights=a.bootstrap,
                 teacher_replay_fights=len(completed_boot),self_replay_fights=0,model=str(model))]
    plot(out,curves,title)
    previous=[]
    training_points=[]
    learner_results={}
    for iteration in range(1,a.updates+1):
        current=out/f'iter{iteration:03d}';current.mkdir(exist_ok=True)
        starts=generate(base,a.namespace,f'learner-{iteration}',a.batch_fights,used)
        path=current/'starts.parquet';write_starts(starts,path)
        mapping.update({r['fight_id']:'train' for r in starts});split.write_text(json.dumps(mapping))
        collection=play(current,'collect',path,model,worker,a.sims,a.workers,explore=True)
        rows=encode(current,'rows',collection,worker)
        collection_results=read_results(collection)
        completed=[r['fight_id'] for r in starts if collection_results.get(r['fight_id'],{}).get('status')=='completed']
        if len(completed)<.95*len(starts):
            raise RuntimeError('more than 5% incomplete learner fights; inspect caps/errors before continuing')
        previous.append((rows,completed))
        replay=previous[-10:]
        nt=min(len(completed_boot),teacher_count(iteration,a.bootstrap))
        anchors=completed_boot[:nt]
        ids=anchors+[fid for _,fids in replay for fid in fids]
        shards=([teacher_rows] if anchors else [])+[r for r,_ in replay]
        model=train(current,shards,ids,model,epochs=a.epochs)
        learner_results.update(collection_results)
        training_points.append(training_point(iteration,collection_results,ids,{**boot_results,**learner_results}))
        plot_training(out,training_points,title)
        if iteration%a.eval_every==0 or iteration==a.updates:
            ev=play(current,'eval',out/'monitor.parquet',model,worker,a.sims,a.workers)
            result=dict(score(monitor,reference,ev),iteration=iteration,training_fights=a.bootstrap+iteration*a.batch_fights,
                        teacher_replay_fights=nt,self_replay_fights=sum(len(fids) for _,fids in replay),model=str(model))
            curves.append(result);plot(out,curves,title)
            print(json.dumps(result),flush=True)
    summary=dict(deck_id=a.deck_id,hp=hp,max_hp=max_hp,fresh_network=True,value_activation='sigmoid',
                 bootstrap_fights=a.bootstrap,learner_fights=a.updates*a.batch_fights,monitoring_fights=a.monitor,
                 final_test_played=False,latest_model=str(model),curves=curves)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Fresh single-deck expert iteration','',
           title+'; fresh width64 network with bounded 100×sigmoid value head.',
           'Value target: 100×actual win. Policy target: normalized search visits, concentration-weighted.',
           'Monitoring seeds never train. Wilson intervals are descriptive after repeated checkpoint inspection. Final 600 seeds remain untouched.','',
           '| update | training fights | teacher/self replay fights | learned2k win rate (95% Wilson) | MCTS20k | paired gap ±1 SE |',
           '|---|---:|---:|---:|---:|---:|']
    for r in curves:
        lo,hi=r['confidence_95'];se='unknown' if r['gap_se'] is None else f'{100*r["gap_se"]:.1f}'
        lines.append(f'| {r["iteration"]} | {r["training_fights"]} | {r["teacher_replay_fights"]}/{r["self_replay_fights"]} | {100*r["win_rate"]:.1f}% ({100*lo:.1f}–{100*hi:.1f}%) | {100*r["mcts_win_rate"]:.1f}% | {100*r["gap"]:+.1f} ±{se} pts |')
    lines+=['',f'Latest model: `{model}`.','Plot: `curve.png`; raw data: `curve.csv` / `curve.json`.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
