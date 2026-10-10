"""Replay targeted correction starts with both trained branches; no learning."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from apps.run_rl.combat_loop import play,read_results,sha
from apps.run_rl.single_deck import score
from apps.run_rl.single_deck_status import journal_counts,read_json


def run(source_run,out):
    source=source_run.resolve()/'out';out.mkdir(parents=True,exist_ok=True)
    starts=pq.read_table(source/'teacher-starts.parquet').to_pylist()
    config=dict(source_run=str(source_run.resolve()),starts_sha=sha(source/'teacher-starts.parquet'),worker_sha=sha(source/'frozen/pv_worker'),control_sha=sha(source/'control/model/model.onnx'),correction_sha=sha(source/'correction/model/model.onnx'),sims=2000,workers=10)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    book=out/'RUNBOOK.md'
    book.write_text('# Correction uptake diagnostic\n\nHypothesis: correction may improve the specifically corrected training starts without measurable transfer to monitor seeds.\n\nReplay all100 targeted training starts with control and correction branches, greedy learner2k,10 workers. No training or checkpoint selection. Primary: paired correction-minus-control. Also report teacher-rescued44 and teacher-unrescued56 subsets. These are training-exposed starts, NOT held-out generalisation. Original learner losses were exploratory, so0/100 is not a matched baseline.\n')
    worker=source/'frozen/pv_worker'
    for arm in ['control','correction']:
        play(out,arm,source/'teacher-starts.parquet',source/arm/'model',worker,2000,10)
    teacher=read_results(source/'teacher-play')
    summary=dict(overall=score(starts,out/'control',out/'correction'),subsets={},final_test_played=False,training_exposed=True)
    for name,won in [('teacher_rescued',True),('teacher_unrescued',False)]:
        subset=[r for r in starts if teacher[r['fight_id']]['fight']['won']==won]
        summary['subsets'][name]=score(subset,out/'control',out/'correction')
    if summary['overall']['n']!=100:raise RuntimeError('incomplete paired evaluation')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with book.open('a') as f:
        f.write('\n## Result\n\n')
        for name,r in [('overall',summary['overall']),*summary['subsets'].items()]:
            lo=100*(r['gap']-1.96*r['gap_se']);hi=100*(r['gap']+1.96*r['gap_se'])
            f.write(f'{name}: correction {r["wins"]}/{r["n"]}, control {round(r["mcts_win_rate"]*r["n"])}/{r["n"]}; paired gap {100*r["gap"]:+.1f}pp, approximate95% interval {lo:+.1f} to {hi:+.1f}pp.\n\n')
        f.write('All100 paired starts complete. These are training-data diagnostics, not final-test results. No further compute automatically launched.\n')
    print(json.dumps(summary,indent=2))


def status(run):
    out=run/'out';meta=read_json(run/'run.json') or {}
    lines=[f'CORRECTION UPTAKE: {meta.get("status","unknown").upper()}', '10 workers; two ×100 greedy learner2k plays; no training']
    for arm in ['control','correction']:
        c=journal_counts(out/arm);lines.append(f'{arm}: {c["logged"]}/100 results; {c["wins"]} wins; {c["error"]} errors; {c["capped"]} caps')
    result=read_json(out/'summary.json')
    if result:
        for name,r in [('overall',result['overall']),*result['subsets'].items()]:lines.append(f'{name}: correction {r["wins"]}/{r["n"]}, control {round(r["mcts_win_rate"]*r["n"])}/{r["n"]}; gap {100*r["gap"]:+.1f}pp, paired SE {100*r["gap_se"]:.1f}pp')
    lines.extend([f'Runbook: {out/"RUNBOOK.md"}',f'Log: {run/"logs/stdout.log"}; detailed logs: {out}/logs/'])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--source-run',type=Path,required=True);r.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('status');s.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    if a.command=='status':print(status(a.run))
    else:run(a.source_run,a.out)

if __name__=='__main__':main()
