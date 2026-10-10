"""Fixed-scale win-only rescue diagnostic. Hard cap50 worker games, no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import pyarrow.parquet as pq
from apps.human_champ.bench import play_one
from apps.run_rl.combat_loop import sha, wait_for_cpu
from apps.run_rl.single_deck import wilson

ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'build/win-only-diagnostic'


def load(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def records(p):
    rows={}
    try:
        for l in p.read_text().splitlines():
            try:r=json.loads(l)
            except json.JSONDecodeError:continue
            rows[r['job_id']]=r
    except FileNotFoundError:pass
    return rows


def prepare(out):
    src=BUILD/'source';src.mkdir(parents=True,exist_ok=True)
    original=(ROOT.parent/'sts_lightspeed/src/sim/search/PublicBeliefCombatSearch.cpp').read_text()
    needle='    if (objectiveMode == 0)\n'
    assert original.count(needle)==1
    helper='''namespace sts::search {
double diagnosticWinOnlyValue(const BattleContext& state, int rootMaxHp) {
    if(state.outcome != Outcome::PLAYER_VICTORY || state.player.curHp <= 0 || state.escapedCombat) return 0;
    return (35.0 + rootMaxHp) / (55.0 + rootMaxHp);
}
}
'''
    native=original.replace(needle,'    if (objectiveMode == 2) return diagnosticWinOnlyValue(state, normalizationMaxHp);\n'+needle)
    native=native.replace('mode > 1)','mode > 2)')
    # Header already exposes int objectiveMode; no ABI/header alteration. Other modes remain unchanged.
    # Fully qualified source methods use namespace imports; declare helper before those definitions.
    pos=native.index('\n',native.index('#include "sim/search/PublicBeliefCombatSearch.h"'))
    native=native[:pos+1]+helper+native[pos+1:]
    (src/'PublicBeliefCombatSearch.cpp').write_text(native)
    worker=(ROOT/'apps/pv/worker.cpp').read_text()
    result=worker[worker.index('Json result('):worker.index('// PV plays directly')]
    teach=worker[worker.index('Json teach('):worker.index('int main(int argc')]
    prefix='''#include "agents/combat/search/teacher_search.hpp"
#include "environments/combat/record_v4.hpp"
#include <chrono>
#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
double since(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
namespace sts::search {double diagnosticWinOnlyValue(const BattleContext&,int);}
'''
    main='''int main(int argc,char**argv){try {
 if(argc==2 && std::string(argv[1])=="selftest") {
  sts::BattleContext s{}; s.escapedCombat=false; s.outcome=sts::Outcome::PLAYER_VICTORY; s.player.curHp=1;
  const double expected=110.0/130;
  if(std::abs(sts::search::diagnosticWinOnlyValue(s,75)-expected)>1e-12)throw std::runtime_error("low HP value");
  s.player.curHp=40;s.turn=100;s.potionCount=4;
  if(std::abs(sts::search::diagnosticWinOnlyValue(s,75)-expected)>1e-12)throw std::runtime_error("resource invariance");
  s.outcome=sts::Outcome::UNDECIDED;
  if(sts::search::diagnosticWinOnlyValue(s,75)!=0)throw std::runtime_error("unresolved");
  s.outcome=sts::Outcome::PLAYER_LOSS;
  if(sts::search::diagnosticWinOnlyValue(s,75)!=0)throw std::runtime_error("loss");
  s.outcome=sts::Outcome::PLAYER_VICTORY;s.escapedCombat=true;
  if(sts::search::diagnosticWinOnlyValue(s,75)!=0)throw std::runtime_error("escape");
  s.escapedCombat=false;s.player.curHp=0;
  if(sts::search::diagnosticWinOnlyValue(s,75)!=0)throw std::runtime_error("invalid victory");
  std::cout<<"selftest passed\\n";return 0;
 }
 if(argc<3 || argc>4 || std::string(argv[1])!="teacher")throw std::runtime_error("usage: diagnostic_worker teacher SIMS [--win-only]");
 const bool winOnly=argc==4;
 if(winOnly && std::string(argv[3])!="--win-only")throw std::runtime_error("unknown flag");
 const auto sims=std::stoll(argv[2]);
 const auto searcher=[sims,winOnly](sts::search::PublicBeliefCombatSearch& search,std::size_t legal){
  if(winOnly)search.setObjective(2,0,0,0,0);
  return stsrl::teacher::run_teacher_search(search,sims,legal,true);
 };
 std::string line;
 while(std::getline(std::cin,line))if(!line.empty())std::cout<<teach(Json::parse(line),searcher,winOnly?"mcts20k fixed-scale-win-only":"mcts20k legacy").dump()<<std::endl;
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}
'''
    # Outcome enum spelling checked during isolated compilation; no game is spent by selftest.
    (src/'worker.cpp').write_text(prefix+result+teach+main)
    cmake=f'''cmake_minimum_required(VERSION 3.25)
project(win_only_diagnostic LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
add_subdirectory("{ROOT}" core)
add_executable(diagnostic_worker worker.cpp PublicBeliefCombatSearch.cpp)
target_link_libraries(diagnostic_worker PRIVATE combat_core)
'''
    (src/'CMakeLists.txt').write_text(cmake)
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    for p in src.iterdir():(snapshot/p.name).write_bytes(p.read_bytes())
    with (out/'build.log').open('w') as log:
        subprocess.run(['cmake','-S',str(src),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release'],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['cmake','--build',str(BUILD),'--target','diagnostic_worker','-j2'],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(BUILD/'diagnostic_worker'),'selftest'],stdout=log,stderr=subprocess.STDOUT,check=True)
    frozen=out/'diagnostic_worker';frozen.write_bytes((BUILD/'diagnostic_worker').read_bytes());frozen.chmod(0o755)
    return frozen


def status(run):
    out=run/'out';meta=load(run/'run.json') or {};config=load(out/'config.json') or {}
    r=records(out/'results.jsonl');s=load(out/'state.json') or {}
    print('DIAGNOSTIC',meta.get('status','unknown').upper(),'phase:',s.get('phase','initializing'))
    print('Games persisted:',len(r),'/50 hard cap; dispatched:',len(load(out/'dispatched.json') or []))
    for group in ['gate','rescue','sanity','fallback']:
        subset=[v for v in r.values() if v['group']==group];done=[v for v in subset if v['status']=='completed']
        print(group, 'logged',len(subset),'completed',len(done),'wins',sum(v['fight']['won'] for v in done))
    if (out/'summary.json').exists():print((out/'summary.json').read_text())
    print('Build log:',out/'build.log','\nStage log:',run/'logs/stdout.log','\nErrors:',run/'logs/stderr.log')


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'dispatched.json').exists():raise RuntimeError('Already dispatched games: no automatic retry; inspect results and budget first')
    parent=a.final.resolve();starts=pq.read_table(parent/'out/starts.parquet').to_pylist();by_id={r['fight_id']:r for r in starts}
    def original(n):return {r['fight_id']:r for r in map(json.loads,(parent/f'out/{n}/results.jsonl').read_text().splitlines())}
    m,l=original('mcts'),original('learned')
    rescue=sorted(k for k in by_id if l[k]['fight']['won'] and not m[k]['fight']['won'])
    both=sorted(k for k in by_id if l[k]['fight']['won'] and m[k]['fight']['won'])
    if len(rescue)!=82:raise ValueError('expected82 rescue cases')
    randomizer=random.Random(20261005);chosen=randomizer.sample(rescue,44);sanity=randomizer.sample(both,2)
    gates=chosen[:2]+sanity
    config=dict(seed=20261005,final=str(parent),cap=50,workers=10,particles=8,simulations=20000,
                rescue_ids=chosen,sanity_ids=sanity,gate_ids=gates,win_scale=110/130,
                interpretation='enriched rescue diagnostic, not attribution')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    def phase(name):(out/'state.json').write_text(json.dumps(dict(phase=name,time=time.time())))
    phase('isolated build and unit tests');worker=prepare(out);config['worker_sha']=sha(worker)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    phase('waiting for other gameplay CPUs');wait_for_cpu(out)
    dispatched=[];done={}
    def batch(ids,objective,group):
        jobs=[]
        for k in ids:
            if len(dispatched)>=50:raise RuntimeError('budget cap')
            job_id=f'{group}:{objective}:{k}';dispatched.append(job_id)
            jobs.append((job_id,k,([str(worker),'teacher','20000']+(['--win-only'] if objective=='win_only' else []),by_id[k],600)))
        (out/'dispatched.json').write_text(json.dumps(dispatched))
        phase(f'{group} / {objective}')
        with ThreadPoolExecutor(10) as pool,(out/'results.jsonl').open('a') as stream:
            futures={pool.submit(play_one,job): (jid,k) for jid,k,job in jobs}
            for future in as_completed(futures):
                jid,k=futures[future];r=future.result();r.update(job_id=jid,group=group,objective=objective)
                stream.write(json.dumps(r)+'\n');stream.flush();done[jid]=r
                print(jid,r['status'],round(r['seconds'],1),flush=True)
        return [done[jid] for jid,_,_ in jobs]
    g=batch(gates,'legacy','gate')
    passed=all(r['status']=='completed' and all(r['fight'][key]==m[k]['fight'][key] for key in ['actions','won','final_hp']) for r,k in zip(g,gates))
    (out/'gates.json').write_text(json.dumps(dict(passed=passed,ids=gates),indent=2))
    fallback_pairs=None
    if passed:
        resc=batch(chosen,'win_only','rescue');controls=batch(sanity,'win_only','sanity')
    else:
        ids=chosen[:23];x=batch(ids,'legacy','fallback');y=batch(ids,'win_only','fallback')
        fallback_pairs=[(u,v) for u,v in zip(x,y) if u['status']=='completed' and v['status']=='completed']
        resc=[v for u,v in fallback_pairs if not u['fight']['won']];controls=[]
        (out/'fallback.json').write_text(json.dumps(dict(ids=ids,legacy_wins=sum(r.get('fight',{}).get('won',False) for r in x),win_only_wins=sum(r.get('fight',{}).get('won',False) for r in y)),indent=2))
    completed=[r for r in resc if r['status']=='completed'];wins=sum(r['fight']['won'] for r in completed)
    n=len(completed);interval=wilson(wins,n)
    summary=dict(gates_passed=passed,design='44 rescue +2 sanity' if passed else '23 pairs on rebuilt binary',
                 games_dispatched=len(dispatched),games_logged=len(done),rescued=wins,completed_rescue=n,
                 requested_rescue=len(resc),confidence_95_wilson=interval,
                 statuses=dict(__import__('collections').Counter(r['status'] for r in done.values())),
                 unfinished=sum(r['status']!='completed' for r in done.values()),
                 sanity=[dict(fight_id=r['fight_id'],status=r['status'],won=r.get('fight',{}).get('won')) for r in controls],
                 interpretation='Conditional enriched rescue diagnostic; cannot attribute net13.5pp or equate residual with learned evaluation.',
                 caveats=['sparse rollout rewards','fixed-scale UCB calibration','selection enrichment','other tree/policy/search differences'])
    if fallback_pairs is not None:
        summary['fallback_paired_coverage']=len(fallback_pairs)
        summary['fallback_rescues']=sum(not u['fight']['won'] and v['fight']['won'] for u,v in fallback_pairs)
        summary['fallback_regressions']=sum(u['fight']['won'] and not v['fight']['won'] for u,v in fallback_pairs)
        summary['fallback_note']='Recovery denominator is rebuilt-legacy losses among complete pairs, not44 or the cached82. See fallback.json for all23 arms.'
    summary['unfinished_fraction']=summary['unfinished']/summary['games_dispatched']
    summary['cutoff_note']='Unfinished GAME statuses recorded. Internal unresolved rollout fraction is not instrumented; unknown, not estimated from game statuses.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2));phase('complete')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    r=s.add_parser('run');r.add_argument('--final',type=Path,required=True);r.add_argument('--out',type=Path,required=True)
    t=s.add_parser('status');t.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a) if a.command=='run' else status(a.run)

if __name__=='__main__':main()
