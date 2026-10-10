"""Bounded teacher correction versus matched replay-only update; no new self-play."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import shutil
import sys

import pyarrow.parquet as pq
from apps.run_rl.combat_loop import encode,play,read_results,sha,stage,write_starts
from apps.run_rl.single_deck import score,wilson
from apps.run_rl.single_deck_status import journal_counts,read_json,training_progress


def select_failures(ids,results,n=100):
    failures=sorted(fid for fid in ids if results[fid]['status']=='completed' and not results[fid]['fight']['won'])
    if len(failures)<n:raise ValueError('not enough failed replay fights')
    return random.Random(20261006).sample(failures,n)


def status(run):
    out=run/'out';meta=read_json(run/'run.json') or {};config=read_json(out/'config.json') or {}
    lines=[f'TEACHER CORRECTION: {meta.get("status","unknown").upper()}',f'Workers: 10; 100 teacher fights; two matched 3-epoch branches',f'Source: {config.get("source_run","not configured yet")}']
    c=journal_counts(out/'teacher-play');lines.append(f'Teacher: {c["logged"]}/100 results; {c["wins"]} wins; {c["error"]} errors; {c["capped"]} caps')
    for arm in ['control','correction']:
        lines.append(f'{arm} training: {training_progress(out/arm,3)}')
        c=journal_counts(out/arm/'eval');lines.append(f'{arm} eval: {c["logged"]}/100; {c["wins"]} wins')
    summary=read_json(out/'summary.json')
    if summary:
        for arm in ['control','correction']:
            r=summary[arm];lo,hi=r['confidence_95'];lines.append(f'{arm}: {r["wins"]}/{r["n"]}; 95% Wilson {100*lo:.1f}–{100*hi:.1f}%')
        r=summary['correction_vs_control'];lines.append(f'Correction minus control: {100*r["gap"]:+.1f} pp; paired SE {100*r["gap_se"]:.1f} pp')
    lines.extend([f'Runbook: {out/"RUNBOOK.md"}',f'Logs: {run/"logs/stdout.log"}; detail: {out}/logs/ and {out}/{{control,correction}}/logs/'])
    return '\n'.join(lines)


def run(source_run,out):
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=source_run.resolve()/'out';cfg=json.loads((source/'config.json').read_text())
    config=dict(source_run=str(source_run.resolve()),checkpoint_sha=sha(source/'iter015/model/model.pt'),worker_sha=sha(source/'frozen/pv_worker'),selection_seed=20261006,teacher_fights=100,replay_fights=1000,epochs=3,lr=.0003,workers=10,mixed_shards=True)
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('resume config changed')
    cp.write_text(json.dumps(config,indent=2))
    frozen=out/'frozen';frozen.mkdir(exist_ok=True)
    for src,dest in [(source/'iter015/model/model.pt',frozen/'model.pt'),(source/'frozen/pv_worker',frozen/'pv_worker')]:
        if not dest.exists():shutil.copy2(src,dest)
        if sha(src)!=sha(dest):raise ValueError('snapshot changed')
    ids=json.loads((source/'iter015/train-fights.json').read_text());assert len(ids)==1000 and len(set(ids))==1000
    results={};starts={};shards=[]
    for i in range(6,16):
        d=source/f'iter{i:03d}';results.update(read_results(d/'collect'));starts.update({r['fight_id']:r for r in pq.read_table(d/'starts.parquet').to_pylist()});shards.append(d/'rows/rows.parquet')
    selected=select_failures(ids,results)
    monitor=pq.read_table(source/'monitor.parquet').to_pylist();final=pq.read_table(source/'final-reserved.parquet').to_pylist()
    forbidden={r['start']['seed'] for r in monitor+final}
    if any(starts[fid]['start']['seed'] in forbidden for fid in ids):raise ValueError('training/eval seed overlap')
    teacher_starts=[];replacement={}
    for fid in selected:
        row=copy.deepcopy(starts[fid]);row['fight_id']=fid+':teacher-correction';row['seed_kind']='teacher-correction';teacher_starts.append(row);replacement[fid]=row['fight_id']
    write_starts(teacher_starts,out/'teacher-starts.parquet');(out/'selection.json').write_text(json.dumps(dict(selected=selected,replacement=replacement),indent=2))
    book=out/'RUNBOOK.md'
    if not book.exists():
        book.write_text('# Targeted teacher correction — runbook\n\nHypothesis: learner search reinforces incorrect action preferences; teacher trajectories on failed training starts provide corrective examples.\n\nProtocol frozen before teacher outcomes: select 100 losses uniformly without replacement from update15 active replay (rounds6–15), RNG20261006. No eval/final starts. MCTS20k, 10 workers; preserve actual outcomes even if teacher loses. Require all100 complete, otherwise halt.\n\nTwo branches from identical update15 weights/AdamW moments: control original1000 learner fights; correction replaces selected100 with teacher trajectories, retains900 learner fights. Three epochs, lr3e-4, 64 states/fight/epoch, grad clip1, concentration-weighted policy, actual win value labels. Both use mixed-shard minibatches (a shared batching change from original streaming updates). Equal192000 sampled training states per branch; same losses/settings.\n\nEvaluate each on same100 consumed monitor seeds, against original update15 and MCTS; primary comparison correction-minus-control paired gap. One training RNG seed, one small monitoring set: exploratory, no final-test claim. No automatic follow-up.\n\nThis tests complete teacher trajectories, NOT teacher annotations on learner intermediate states; cannot separate changed policy labels, state distribution and outcome labels. Learner-loss selection itself changes outcome mix in correction.\n\nStatus: collecting teacher correction data.\n')
    teacher=play(out,'teacher-play',out/'teacher-starts.parquet',None,frozen/'pv_worker',20000,10,teacher=True)
    tr=read_results(teacher)
    if len(tr)!=100 or any(r['status']!='completed' for r in tr.values()):raise RuntimeError('teacher collection incomplete; halt without replacement/retries')
    teacher_rows=encode(out,'teacher-rows',teacher,frozen/'pv_worker')
    nt=sum(r['fight']['won'] for r in tr.values());lo,hi=wilson(nt,100)
    with book.open('a') as f:f.write(f'\nTeacher recovery: {nt}/100 wins (95% Wilson {100*lo:.1f}–{100*hi:.1f}%). All100 complete.\n')
    split={fid:'train' for fid in ids};split.update({fid:'train' for fid in replacement.values()});split.update({r['fight_id']:'val' for r in monitor});(out/'split.json').write_text(json.dumps(split))
    validation=source/'monitor-rows/rows.parquet'
    for arm in ['control','correction']:
        d=out/arm;d.mkdir(exist_ok=True)
        chosen=[replacement.get(fid,fid) if arm=='correction' else fid for fid in ids];(d/'train-fights.json').write_text(json.dumps(chosen))
        data=shards+([teacher_rows] if arm=='correction' else [])
        cmd=[sys.executable,'-m','agents.combat.pv.train','--data',*data,validation,'--split-manifest',out/'split.json','--train-fights',d/'train-fights.json','--states-per-fight',64,'--flat-policy-weighting','--width',64,'--epochs',3,'--lr',.0003,'--grad-clip',1,'--device','cuda','--mix-shards','--init',frozen/'model.pt','--resume-optimizer','--out',d/'model']
        stage(d,'train',cmd)
        play(d,'eval',source/'monitor.parquet',d/'model',frozen/'pv_worker',2000,10)
    summary=dict(teacher_recovery=dict(wins=nt,n=100,confidence_95=wilson(nt,100)),control=score(monitor,source/'reference-monitor',out/'control/eval'),correction=score(monitor,source/'reference-monitor',out/'correction/eval'),control_vs_update15=score(monitor,source/'iter015/eval',out/'control/eval'),correction_vs_update15=score(monitor,source/'iter015/eval',out/'correction/eval'),correction_vs_control=score(monitor,out/'control/eval',out/'correction/eval'),final_test_played=False)
    for arm in ['control','correction']:
        if summary[arm]['n']!=100:raise RuntimeError('incomplete evaluation; inspect statuses before reporting')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    gap=summary['correction_vs_control'];lo=100*(gap['gap']-1.96*gap['gap_se']);hi=100*(gap['gap']+1.96*gap['gap_se'])
    with book.open('a') as f:
        f.write(f'\n## Result\n\nControl {summary["control"]["wins"]}/100; correction {summary["correction"]["wins"]}/100; original update15 66/100; MCTS79/100. Correction-minus-control {100*gap["gap"]:+.1f}pp, paired SE {100*gap["gap_se"]:.1f}pp, approximate95% normal interval {lo:+.1f} to {hi:+.1f}pp. Exploratory consumed monitor, one training RNG. All stages complete; final seeds untouched.\n')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--source-run',type=Path,required=True);r.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('status');s.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    if a.command=='status':print(status(a.run))
    else:run(a.source_run,a.out)

if __name__=='__main__':main()
