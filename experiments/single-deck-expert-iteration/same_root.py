"""Same-root continuations: original update15 (2k, frozen worker `play` semantics) from the exact teacher-label roots.

Binary: apps/continuation (pv_continuation), built in build/continuation-v1 and copied to the run's frozen/ before any
game. States: the frozen 278-state eligible manifest of demon-form-continuation-v1; roots: particles 0-3 per state.

  freeze   : copy binary + sources + hashes
  gates    : G1 true-start replay of 5 consumed iter015/eval games; G2 root fingerprints (no play);
             G3 teacher-label reproduction (8 teacher continuations); G4 learner smoke (<=12 unique + <=4 repeats)
  bulk     : <=1,112 learner continuations (10 processes, 300 s/state)
  analyze  : paired same-root teacher vs update15 and V vs update15 continuation
"""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import continuation as T  # noqa: E402
import correction_absorption as C  # noqa: E402
import decoupled_policy as DP  # noqa: E402

EXP = 'demon-form-same-root-v1'
OUT = C.DATE / f'id={EXP}/out'
TEACHER_RUN = T.OUT
BUILD = Path('build/continuation-v1/pv_continuation')
MODEL = DP.A_DIR / 'model.onnx'
SIMS = 2000
K = 4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def cmd_freeze(out):
    d = out / 'frozen'
    if d.exists(): raise FileExistsError('frozen exists')
    d.mkdir(parents=True); shutil.copy2(BUILD, d / 'pv_continuation')
    for s in ['apps/continuation/continuation.cpp', 'apps/continuation/CMakeLists.txt']:
        (d / 'src' / s).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(s, d / 'src' / s)
    meta = dict(binary_sha=sha(d / 'pv_continuation'), binary_mtime=datetime.datetime.fromtimestamp(BUILD.stat().st_mtime).isoformat(),
                frozen_at=now(), sources={s: sha(s) for s in ['apps/continuation/continuation.cpp', 'apps/continuation/CMakeLists.txt']},
                pv_sources={s: sha(s) for s in ['agents/combat/pv/search.cpp', 'agents/combat/pv/search.hpp', 'agents/combat/pv/features.cpp',
                                                'agents/combat/pv/evaluator.cpp', 'agents/combat/search/teacher_leaves.cpp',
                                                'agents/combat/search/teacher_search.cpp', 'environments/combat/environment.cpp']},
                head=T.git('.', 'rev-parse', 'HEAD'), sts_lightspeed_head=T.git(T.LS, 'rev-parse', 'HEAD'),
                model=str(MODEL), model_sha=sha(MODEL), model_data_sha=sha(str(MODEL) + '.data'),
                teacher_run_eligible_sha=sha(TEACHER_RUN / 'eligible.json'))
    (d / 'snapshot.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))


def call(out, requests, timeout):
    meta = json.loads((out / 'frozen/snapshot.json').read_text())
    if sha(out / 'frozen/pv_continuation') != meta['binary_sha']: raise ValueError('frozen binary changed')
    proc = subprocess.run([str(out / 'frozen/pv_continuation'), str(MODEL)], input=''.join(json.dumps(r) + '\n' for r in requests),
                          text=True, capture_output=True, timeout=timeout)
    if proc.returncode: raise RuntimeError(f'pv_continuation exit {proc.returncode}: {proc.stderr[-500:]}')
    return [json.loads(l) for l in proc.stdout.splitlines()]


def state_request(x, mode, particles, **kw):
    return dict(start=x['start'], ops=x['prefix'], mode=mode, sims=SIMS if mode == 'learner' else T.SIMS, particles=particles, **kw)


def cmd_gates(out):
    res = {}
    # G1: true-start learner replay == recorded iter015/eval (consumed monitor seeds), 5 games
    rec = {r['fight_id']: r for r in map(json.loads, (C.SRC / 'iter015/eval/results.jsonl').read_text().splitlines())}
    ids = sorted(rec)[:5]; t = time.time()
    resp = call(out, [dict(start=rec[i]['fight']['start'], ops=[], mode='learner', sims=SIMS, true_state=True, record=True) for i in ids], 900)
    g1 = []
    for i, o in zip(ids, resp):
        r = o['results'][0]; f = rec[i]
        mine = [(s['step'], round(s['root_value'], 9), sorted((c['action'], c['visits']) for c in s['children'])) for s in r['search']]
        theirs = [(s['step'], round(s['root_value'], 9), sorted((c['action'], c['visits']) for c in s['children'])) for s in f['search']]
        g1.append(dict(fight_id=i, actions_equal=r['actions'] == f['fight']['actions'], search_equal=mine == theirs,
                       root_values_bitwise=[s['root_value'] for s in r['search']] == [s['root_value'] for s in f['search']],
                       won_equal=r['won'] == f['fight']['won'], status=r['status']))
    res['G1_true_start_replay'] = dict(games=len(g1), seconds=time.time() - t, all_equal=all(
        x['actions_equal'] and x['search_equal'] and x['root_values_bitwise'] and x['won_equal'] for x in g1), detail=g1)
    # G2: root fingerprints (no play): deterministic across calls, distinct particles, never the true state
    xs = T.eligible_states(TEACHER_RUN)
    a = call(out, [state_request(x, 'roots', list(range(K))) for x in xs], 600)
    b = call(out, [state_request(x, 'roots', [3, 1, 0, 2]) for x in xs], 600)
    fa = [{r['particle']: r['root_fingerprint'] for r in o['results']} for o in a]
    fb = [{r['particle']: r['root_fingerprint'] for r in o['results']} for o in b]
    res['G2_roots'] = dict(states=len(xs), errors=sum('error' in o for o in a + b),
                           deterministic=sum(x == y for x, y in zip(fa, fb)),
                           distinct_particles=sum(len(set(x.values())) == K for x in fa),
                           none_equal_true=sum(o['true_fingerprint'] not in x.values() for o, x in zip(a, fa)))
    # G3: teacher-label reproduction from the snapshot labels (2 states x 4 particles = 8 teacher continuations)
    labels = {(l['fight_id'], l['step']): l for l in map(json.loads, (TEACHER_RUN / 'labels.jsonl').read_text().splitlines())}
    picks = [xs[0], xs[len(xs) // 2]]; t = time.time()
    resp = call(out, [state_request(x, 'teacher', list(range(K))) for x in picks], 900)
    g3 = []
    for x, o in zip(picks, resp):
        stored = {p['particle']: (p['won'], p['hp'], p['turns']) for p in labels[(x['fight_id'], x['step'])]['playouts']}
        mine = {r['particle']: (r['won'], r['hp'], r['turns']) for r in o['results']}
        g3.append(dict(fight_id=x['fight_id'], step=x['step'], equal=stored == mine, stored=stored, new=mine))
    res['G3_teacher_reproduction'] = dict(continuations=8, seconds=time.time() - t, all_equal=all(g['equal'] for g in g3), detail=g3)
    res['pass'] = bool(res['G1_true_start_replay']['all_equal'] and res['G2_roots']['errors'] == 0
                       and res['G2_roots']['deterministic'] == len(xs) and res['G2_roots']['distinct_particles'] == len(xs)
                       and res['G2_roots']['none_equal_true'] == len(xs) and res['G3_teacher_reproduction']['all_equal'])
    (out / 'gates.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1, default=str))
    if not res['pass']: raise SystemExit('GATES FAILED: stop')


def job(args):
    out, x, particles, timeout = args
    t = time.time()
    try:
        o = call(out, [state_request(x, 'learner', particles)], timeout)[0]
        status = 'error' if 'error' in o else 'completed'
        return dict(fight_id=x['fight_id'], step=x['step'], status=status, seconds=time.time() - t, results=o.get('results'),
                    error=o.get('error'), true_fingerprint=o.get('true_fingerprint'))
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        return dict(fight_id=x['fight_id'], step=x['step'], status='error', seconds=time.time() - t, error=repr(e))


def run(out, name, jobs, workers):
    from concurrent.futures import ThreadPoolExecutor
    dest = out / f'{name}.jsonl'
    if dest.exists(): raise FileExistsError(f'{dest} exists; no resume/retry')
    total, done, statuses = len(jobs), 0, collections.Counter()
    with ThreadPoolExecutor(workers) as pool, dest.open('w') as f:
        for r in pool.map(job, jobs):
            f.write(json.dumps(r) + '\n'); f.flush(); done += 1
            caps = sum(x['status'] == 'capped' for x in r.get('results') or [])
            statuses[r['status']] += 1
            wins = sum(x['won'] for x in r['results']) if r.get('results') else None
            print(f'{done}/{total} {r["fight_id"]} step {r["step"]} {r["status"]} wins={wins} capped={caps} {r["seconds"]:.1f}s '
                  f'statuses={dict(statuses)}', flush=True)


def cmd_smoke(out):
    """<=12 unique learner continuations: 3 eligible states of different kinds x 4 particles (+4 repeats of the first)."""
    xs = T.eligible_states(TEACHER_RUN); picks = []
    for kind in ['dual_wield_select', 'setup_legal', 'other_play']:
        picks.append(next(x for x in xs if x['stratum'] == kind))
    jobs = [(out, x, list(range(K)), 300) for x in picks] + [(out, picks[0], list(range(K)), 300)]
    run(out, 'smoke', jobs, 4)


def cmd_bulk(out):
    xs = T.eligible_states(TEACHER_RUN)
    run(out, 'learner', [(out, x, list(range(K)), 300) for x in xs], 10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['freeze', 'gates', 'smoke', 'bulk'])
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args(); (a.out / 'logs').mkdir(parents=True, exist_ok=True)
    dict(freeze=cmd_freeze, gates=cmd_gates, smoke=cmd_smoke, bulk=cmd_bulk)[a.stage](a.out)


if __name__ == '__main__':
    main()
