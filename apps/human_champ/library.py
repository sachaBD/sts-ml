"""Diverse fixed-human-loadout MCTS screening library, with a read-only dashboard."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from apps.human_champ import bench
from apps.run_rl.combat_loop import play, sha, write_starts
from apps.run_rl.single_deck import generate, wilson


CARDS = {v:k for k,v in bench.enum_ids('Cards.h','CardId').items()}
RELICS = {v:k for k,v in bench.enum_ids('Relics.h','RelicId').items()}
COMMON = {'strike_red','defend_red','ascenders_bane','bash'}
QUOTAS = {'block':3,'demon_form':3,'exhaust':4,'strength':4,'mixed':4}


def names(row):
    return {CARDS[c['id']] for c in row['start']['deck']}


def bucket(row):
    cards=names(row)
    if cards & {'barricade','entrench'}:return 'block'
    if 'demon_form' in cards:return 'demon_form'
    if cards & {'corruption','dark_embrace','feel_no_pain'}:return 'exhaust'
    if cards & {'limit_break','inflame','spot_weakness','jax','heavy_blade','flex'}:return 'strength'
    return 'mixed'


def family(row):
    return tuple(sorted(Counter((c['id'],c['upgraded'],c['misc']) for c in row['start']['deck']).items()))


def distance(a,b):
    x,y=names(a)-COMMON,names(b)-COMMON
    return 1-len(x&y)/max(1,len(x|y))


def select(rows, references):
    by_id={r['deck_id']:r for r in rows if r['seed_kind']=='human'}
    chosen=[by_id[k] for k in references]
    used={family(r) for r in chosen}
    for group,n in QUOTAS.items():
        for _ in range(n):
            available=[r for r in by_id.values() if bucket(r)==group and family(r) not in used]
            if not available:raise ValueError(f'not enough distinct deck families in {group}')
            # Farthest-first on non-starter card sets. Stable deterministic tie breaker.
            rank=lambda r:(min(distance(r,s) for s in chosen),hashlib.sha256(r['deck_id'].encode()).hexdigest())
            row=max(available,key=rank);chosen.append(row);used.add(family(row))
    return chosen


def json_file(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def results(path):
    output={}
    try:
        with path.open() as stream:
            for line in stream:
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                output[r['fight_id']]=r
    except FileNotFoundError:pass
    return output


def catalogue(out):
    manifest=json_file(out/'manifest.json') or []
    live=results(out/'play/results.jsonl')
    cached=results(out/'reused-results.jsonl')
    found={**cached,**live}
    report=[]
    for row in manifest:
        records=[found[k] for k in row['fight_ids'] if k in found]
        completed=[r for r in records if r['status']=='completed']
        wins=sum(r['fight']['won'] for r in completed)
        report.append(dict(row, logged=len(records),completed=len(completed),wins=wins,
                           statuses=dict(Counter(r['status'] for r in records)),
                           win_rate=wins/len(completed) if completed else None,
                           confidence_95=wilson(wins,len(completed))))
    return report


def status(run):
    meta=json_file(run/'run.json') or {}
    out=run/'out';config=json_file(out/'config.json')
    if not config:
        print(f'Library {run.name}: {meta.get("status","initializing")}');return
    rows=catalogue(out)
    logged=sum(r['logged'] for r in rows)
    print(f'LIBRARY: {run.name}   {meta.get("status","unknown").upper()}')
    print(f'MCTS{config["sims"]}, {config["workers"]} workers; fixed original HP/relics, no potions')
    print(f'Overall: {logged}/400 results; 40 reused reference results + 360 new trials')
    print(f'Loadouts complete: {sum(r["logged"]==20 for r in rows)}/20\n')
    for r in rows:
        ci=r['confidence_95']
        interval=f'{100*ci[0]:.0f}–{100*ci[1]:.0f}%' if ci else '—'
        suffix=' (reused)' if r['reused'] else ''
        print(f'{r["deck_id"][:8]} {r["bucket"]:10} HP {r["hp"]:2}/{r["max_hp"]:2} '
              f'logged {r["logged"]:2}/20  wins {r["wins"]:2}/{r["completed"]:2} '
              f'95% Wilson {interval}{suffix}')
        bad={k:v for k,v in r['statuses'].items() if k!='completed'}
        if bad:print('  Non-completed:',bad)
    print('\nThese are seed trials per loadout, not an estimate across all human decks.')
    print(f'Catalogue: {out/"catalogue.json"}\nReport: {out/"REPORT.md"}\nDetailed log: {out/"logs/play.log"}')
    if meta.get('status')=='failed':print('Failure log:',run/'logs/stderr.log')


def build(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    references={}
    for run in a.reference:
        summary=json_file(run/'out/summary.json')
        if not summary or summary['attempted']!=20 or summary['simulations']!=20000:
            raise ValueError('reference must be a completed 20-trial MCTS20k screen')
        references[summary['deck_id']]=run.resolve()
    if len(references)!=2:raise ValueError('exactly two distinct reference loadouts required')
    config=dict(kind='human_deck_library',sources={str(p.resolve()):sha(p) for p in a.source},
                references={k:str(v) for k,v in references.items()},worker_sha=sha(a.worker),
                workers=a.workers,sims=20000,n=20,namespace='human-deck-library-v1',quotas=QUOTAS)
    path=out/'config.json'
    if path.exists() and json_file(path)!=config:raise ValueError('configuration changed on resume')
    path.write_text(json.dumps(config,indent=2))
    frozen=out/'frozen';frozen.mkdir(exist_ok=True);worker=frozen/'pv_worker'
    if not worker.exists():shutil.copy2(a.worker,worker)
    if sha(worker)!=config['worker_sha']:raise ValueError('frozen worker checksum mismatch')
    pool=[]
    for source in a.source:pool.extend(pq.read_table(source).to_pylist())
    selected=select(pool,references)
    used={r['start']['seed'] for r in pool}
    reused=[];old_results=[]
    for r in selected[:2]:
        run=references[r['deck_id']];rs=pq.read_table(run/'out/starts.parquet').to_pylist()
        if len(rs)!=20:raise ValueError('reference starts count mismatch')
        # Compare all fixed start fields, except RNGs and seed; never reuse different HP/loadouts.
        omit={'seed','misc_rng','potion_rng'}
        if any({k:v for k,v in x['start'].items() if k not in omit} !=
               {k:v for k,v in r['start'].items() if k not in omit} for x in rs):
            raise ValueError('reference and library fixed loadout differ')
        reused.extend(rs);used.update(x['start']['seed'] for x in rs)
        old_results.extend(results(run/'out/play/results.jsonl').values())
    new=[]
    for r in selected[2:]:
        new.extend(generate(r,f'human-deck-library-v1:{r["deck_id"]}','selection',20,used))
    write_starts(new,out/'new-starts.parquet');write_starts(reused+new,out/'starts.parquet')
    (out/'reused-results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old_results))
    manifest=[]
    for row in selected:
        st=row['start'];rs=[r for r in reused+new if r['deck_id']==row['deck_id']]
        manifest.append(dict(deck_id=row['deck_id'],source_run_id=row['source_run_id'],
                             build_version=row['build_version'],human_won=row['human_won'],
                             bucket=bucket(row),hp=st['hp'],max_hp=st['max_hp'],deck_size=len(st['deck']),
                             cards=[dict(name=CARDS[c['id']],upgraded=c['upgraded'],misc=c['misc']) for c in st['deck']],
                             relics=[dict(name=RELICS[r['id']],data=r['data']) for r in st['relics']],
                             changed_cards=row['changed_cards'],reset_counters=row['reset_counters'],
                             fight_ids=[r['fight_id'] for r in rs],
                             reused=str(references[row['deck_id']]) if row['deck_id'] in references else None))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Library: 20 loadouts × 20 MCTS20k seeds; 40 reference trials reused, 360 new trials',flush=True)
    play(out,'play',out/'new-starts.parquet',None,worker,20000,a.workers,teacher=True)
    report=catalogue(out)
    (out/'catalogue.json').write_text(json.dumps(report,indent=2))
    fields=['deck_id','bucket','hp','max_hp','deck_size','logged','completed','wins','win_rate','ci_low','ci_high','human_won','reused']
    with (out/'catalogue.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in report:
            x={k:r[k] for k in fields if k in r};ci=r['confidence_95']
            x.update(ci_low=ci[0] if ci else None,ci_high=ci[1] if ci else None);writer.writerow(x)
    lines=['# Human fixed-loadout MCTS screening library','',
           '20 loadouts × 20 seeds, including two reused references. MCTS20k; fixed original HP and relics/counters, no potions.',
           'Selection is deterministic archetype stratification plus non-starter-card diversity, not a representative population sample.',
           'HP differs between loadouts: this screens combat starts, not deck strength independent of HP.',
           'Wilson intervals describe independent seed trials within each loadout; small differences cannot be reliably ranked at n=20.',
           'Rows remain in the catalogue regardless of outcome. Caps/errors are reported and excluded from outcome-only rates.',
           'These selection seeds must not later be called untouched final-test seeds. Historical reconstruction/balance caveats remain unchanged.','',
           '| deck | group | HP | size | wins/completed | 95% Wilson | reused |','|---|---|---:|---:|---:|---:|---|']
    for r in report:
        ci=r['confidence_95'];interval=f'{100*ci[0]:.1f}–{100*ci[1]:.1f}%' if ci else '—'
        lines.append(f'| {r["deck_id"][:8]} | {r["bucket"]} | {r["hp"]}/{r["max_hp"]} | {r["deck_size"]} | '
                     f'{r["wins"]}/{r["completed"]} | {interval} | {bool(r["reused"])} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Done: catalogue and report at {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--source',type=Path,action='append',required=True)
    r.add_argument('--reference',type=Path,action='append',required=True)
    r.add_argument('--worker',type=Path,required=True);r.add_argument('--workers',type=int,default=10)
    r.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('status');s.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    if a.command=='status':status(a.run)
    else:
        if not 1<=a.workers<=10:p.error('1–10 workers required')
        build(a)


if __name__=='__main__':main()
