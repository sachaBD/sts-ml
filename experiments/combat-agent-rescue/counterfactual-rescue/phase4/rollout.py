#!/usr/bin/env python3
"""Phase 4: rollout-leaf learned search (frozen pv_worker --rollout-mix) on 100 dev starts. Run from repo root.

  parity : <=2 zero-flag replays (original iter015 dev:0 exact actions/search; rescue u3 dev:0 won/hp/actions)
  main   : manifest -> pilot (dev:0-4 x 2 arms) -> cost gate -> bulk (190) -> REPORT.md (always written)
No training; frozen worker/model unaltered; errors/caps/timeouts halt (never relabelled).
"""
import argparse, concurrent.futures, hashlib, json, math, os, shutil, signal, statistics, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path.cwd(); assert (REPO / 'AGENTS.md').exists(), 'run from repo root'
D6 = REPO / 'runs/schema=combat_v4/date=2026-10-06'
RUN = D6 / 'id=counterfactual-rollout-leaf-v1'; OUT = RUN / 'out'
WORKER = D6 / 'id=single-deck-demon-form-v1/out/frozen/pv_worker'
WORKER_SHA = '58c5c4ab8d3dbbd6868a22bb86b467b08d3686fd8e0f368206b9b5368d83a6b0'
ORIG = D6 / 'id=single-deck-demon-form-v1/out/iter015/model/model.onnx'
S2 = D6 / 'id=counterfactual-rescue-v1/out/stage2'
U3 = S2 / 'rescue/update3/model/model.onnx'
DEV = D6 / 'id=demon-form-decoupled-policy-v1/out/main'
DEV_REF = D6 / 'id=demon-form-budget-teacher-v1/out/per_fight.jsonl'
ARMS = {'mix0.5': '0.5', 'mix1.0': '1'}
AGENT_MARK = {'mix0.5': 'rollout_mix=0.500000', 'mix1.0': 'rollout_mix=1.000000'}
WORKERS, TIMEOUT, PILOT, GATE_MIN, GAME_MAX = 10, 900, 5, 45, 300
ACTIVE = set()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def log(*x): print(now(), *x, flush=True)


class Stop(Exception): pass


def kill(p):
    if p and p.poll() is None:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def play(model, mix, fight):
    cmd = [str(OUT / 'frozen/pv_worker' if (OUT / 'frozen/pv_worker').exists() else WORKER), 'play', str(model), '2000'] + (['--rollout-mix', mix] if mix is not None else [])
    t0 = time.time(); p = None; err = None; out = ''
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True); ACTIVE.add(p)
        out, se = p.communicate(json.dumps({'fight_id': fight['fight_id'], 'start': fight['start']}) + '\n', timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        kill(p); err = f'{TIMEOUT}s timeout'; se = ''
    finally:
        if p: ACTIVE.discard(p)
    rec = {'fight_id': fight['fight_id'], 'cmd': cmd[1:], 'started': t0, 'finished': time.time(), 'seconds': time.time() - t0, 'error': err,
           'returncode': None if err else p.returncode, 'stderr': (se or '')[-1500:]}
    try:
        r = json.loads(out) if not err and p.returncode == 0 and out.count('\n') == 1 else None
    except json.JSONDecodeError:
        r = None
    if not r or r.get('status') not in ('completed', 'capped'): rec['status'] = 'error'; return rec, r
    rec['status'] = r['status']
    if r['status'] == 'completed':
        rec.update(won=bool(r['fight']['won']), hp=r['fight']['final_hp'], actions=len(r['fight']['actions']), agent=r['fight']['agent'])
    return rec, r


def dev_starts():
    d = pq.read_table(DEV / 'starts.parquet').to_pylist()
    if len(d) != 100: raise Stop('dev starts')
    return d


def parity():
    OUT.mkdir(parents=True, exist_ok=True); led = OUT / 'parity-ledger.jsonl'
    if led.exists(): raise SystemExit('parity already run')
    if sha(WORKER) != WORKER_SHA: raise SystemExit('worker sha')
    dev = {f['fight_id']: f for f in dev_starts()}; fid = 'demon-form-decoupled-policy-v1:dev:0'
    ref = json.loads(open(DEV / 'baseline/results.jsonl').readline()); assert ref['fight_id'] == fid
    a, ra = play(ORIG, '0', dev[fid])
    s2 = [json.loads(l) for l in (S2 / 'ledger.jsonl').read_text().splitlines()]
    s2 = [x for x in s2 if x['stage'] == 'dev-eval' and x['job']['fight_id'] == fid and '/rescue/update3/' in x['job']['model']][0]
    b, rb = play(U3, '0', dev[fid])
    with open(led, 'w') as f:
        for x in (a, b): f.write(json.dumps(x) + '\n')
    ok1 = ra is not None and ra['fight']['actions'] == ref['fight']['actions'] and ra['search'] == ref['search'] and ra['fight']['won'] == ref['fight']['won']
    ok2 = b['status'] == 'completed' and (b['won'], b['hp'], b['actions']) == (s2['won'], s2['hp'], s2['actions'])
    res = {'orig_dev0_exact_actions_search': ok1, 'rescue_u3_dev0_won_hp_actions': ok2, 'agent_zero': ra and ra['fight']['agent'],
           'orig': {k: a.get(k) for k in ('status', 'won', 'hp', 'actions', 'seconds')}, 'u3': {k: b.get(k) for k in ('status', 'won', 'hp', 'actions', 'seconds')},
           'u3_stage2': {k: s2.get(k) for k in ('won', 'hp', 'actions')}}
    print(json.dumps(res, indent=1), flush=True)
    if not (ok1 and ok2): raise SystemExit('parity FAILED')
    (OUT / 'parity-passed.json').write_text(json.dumps(res, indent=1))


STATE = {'start': None, 'stages': [], 'last_success': None, 'failure': None, 'results': {}}


def freeze():
    fz = OUT / 'frozen'
    if fz.exists(): raise Stop('frozen exists')
    fz.mkdir(); shutil.copy2(WORKER, fz / 'pv_worker'); shutil.copy2(Path(__file__), fz / 'rollout.py')
    man = {'worker_sha256': sha(fz / 'pv_worker'), 'runner_sha256': sha(fz / 'rollout.py'), 'model': str(U3), 'model_sha256': sha(U3),
           'model_data_sha256': sha(str(U3) + '.data'), 'starts': str(DEV / 'starts.parquet'), 'starts_sha256': sha(DEV / 'starts.parquet'),
           'arms': {k: {'cmd': [str(fz / 'pv_worker'), 'play', str(U3), '2000', '--rollout-mix', v], 'stdin': '{"fight_id", "start"} per game'} for k, v in ARMS.items()},
           'start_order': [f['fight_id'] for f in dev_starts()], 'pilot_fight_ids': [f['fight_id'] for f in dev_starts()[:PILOT]],
           'rollout_cutoff_note': 'internal GuidedRollout undecided after 512 rollout actions -> 0 (existing objective approximation, source-based); '
                                  'whole-game cap (turn>=50 or 512 actions) or error halts the run',
           'cost_gate': f'predicted bulk wall = sum_arms(mean pilot s x {100 - PILOT}) / {WORKERS} > {GATE_MIN} min, or any pilot game > {GAME_MAX}s -> stop',
           'timeout_s': TIMEOUT, 'workers': WORKERS, 'frozen_at': now()}
    if man['worker_sha256'] != WORKER_SHA: raise Stop('worker sha')
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1)); return man


def run_batch(pool, led, jobs, stage):
    futs = {pool.submit(play, U3, ARMS[arm], f): (arm, f) for arm, f in jobs}; out = []
    for n, fu in enumerate(concurrent.futures.as_completed(futs), 1):
        arm, f = futs[fu]; rec, _ = fu.result(); rec.update(arm=arm, stage=stage, logged=now())
        led.write(json.dumps(rec) + '\n'); led.flush(); os.fsync(led.fileno()); out.append(rec)
        STATE['results'].setdefault(arm, {})[rec['fight_id']] = {k: rec.get(k) for k in ('status', 'won', 'hp', 'actions', 'seconds')}
        if rec['status'] != 'completed' or AGENT_MARK[arm] not in rec.get('agent', ''):
            for x in futs: x.cancel()
            raise Stop(f'{stage}: {arm} {rec["fight_id"]} status={rec["status"]} err={rec.get("error")} agent={rec.get("agent")} stderr={rec["stderr"][-300:]}')
        if n % 20 == 0 or n == len(jobs): log(stage, f'{n}/{len(jobs)}', {a: sum(bool(v.get('won')) for v in r.values()) for a, r in STATE['results'].items()})
    return out


def main_run():
    STATE['start'] = now()
    if not (OUT / 'parity-passed.json').exists(): raise Stop('parity not passed')
    STATE['manifest'] = freeze(); dev = dev_starts()
    with open(OUT / 'ledger.jsonl', 'a') as led, concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        t0 = now(); recs = run_batch(pool, led, [(a, f) for f in dev[:PILOT] for a in ARMS], 'pilot')
        means = {a: statistics.mean(r['seconds'] for r in recs if r['arm'] == a) for a in ARMS}
        pred = sum(means[a] * (100 - PILOT) for a in ARMS) / WORKERS / 60
        STATE['cost_gate'] = {'pilot_mean_s': means, 'pilot_max_s': max(r['seconds'] for r in recs), 'predicted_bulk_min': pred}
        STATE['stages'].append({'stage': 'pilot', 'start': t0, 'finish': now()}); STATE['last_success'] = 'pilot'
        log('COST GATE', STATE['cost_gate'])
        if pred > GATE_MIN or STATE['cost_gate']['pilot_max_s'] > GAME_MAX: raise Stop(f'cost gate: predicted {pred:.1f} min / max game {STATE["cost_gate"]["pilot_max_s"]:.0f}s')
        t0 = now(); run_batch(pool, led, [(a, f) for f in dev[PILOT:] for a in ARMS], 'bulk')
        STATE['stages'].append({'stage': 'bulk', 'start': t0, 'finish': now()}); STATE['last_success'] = 'bulk'


def paired(a, b):
    keys = sorted(set(a) & set(b)); n = len(keys)
    d = np.array([int(a[k]) - int(b[k]) for k in keys], float); se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan')
    ao, bo = int((d > 0).sum()), int((d < 0).sum()); m = ao + bo
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(ao, bo) + 1)) / 2 ** m) if m else 1.0
    return n, d.mean() if n else float('nan'), se, ao, bo, p


def wilson(k, n):
    z = 1.96; ph = k / n; c = (ph + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def report():
    L = ['# Phase 4 rollout-leaf learned search — REPORT', '', f'Generated {now()}; started {STATE["start"]}; last successful stage **{STATE["last_success"]}**; '
         f'outcome **{"COMPLETE" if not STATE["failure"] else "INCOMPLETE — " + STATE["failure"]}**.', '',
         'Rescue update3 ONNX, frozen pv_worker 58c5c4ab `play MODEL 2000 --rollout-mix L`, 100 existing dev starts, greedy. No training.', '']
    s2 = [json.loads(l) for l in (S2 / 'ledger.jsonl').read_text().splitlines()]
    arms = {}
    for a, r in STATE['results'].items(): arms[a] = {k: v['won'] for k, v in r.items() if v['status'] == 'completed'}
    base_s = [x['seconds'] for x in s2 if x['stage'] == 'dev-eval' and '/rescue/update3/' in x['job']['model']]
    arms['rescue_u3_2k'] = {x['job']['fight_id']: x['won'] for x in s2 if x['stage'] == 'dev-eval' and '/rescue/update3/' in x['job']['model']}
    ref = [json.loads(l) for l in DEV_REF.read_text().splitlines()]; arms['mcts_20k'] = {x['fight_id']: x['won_teacher'] for x in ref}
    L += ['| agent | wins/completed (intended 100) | Wilson 95% | mean s | median s |', '|---|---|---|---|---|']
    for a, w in arms.items():
        secs = [v['seconds'] for v in STATE['results'].get(a, {}).values()] or (base_s if a == 'rescue_u3_2k' else [])
        k, n = sum(w.values()), len(w); lo, hi = wilson(k, n) if n else (float('nan'),) * 2
        L.append(f'| {a} | {k}/{n} | {lo:.2f}–{hi:.2f} | {statistics.mean(secs):.1f} | {statistics.median(secs):.1f} |' if secs else f'| {a} | {k}/{n} | {lo:.2f}–{hi:.2f} | rec. | rec. |')
    L += ['', '| paired (a − b) | pairs | gap | approx 95% CI | a-only | b-only | exact McNemar p |', '|---|---|---|---|---|---|---|']
    for a, b in [('mix0.5', 'rescue_u3_2k'), ('mix1.0', 'rescue_u3_2k'), ('mix0.5', 'mcts_20k'), ('mix1.0', 'mcts_20k'), ('mix1.0', 'mix0.5')]:
        if a in arms and b in arms and arms[a]:
            n, g, se, ao, bo, p = paired(arms[a], arms[b])
            L.append(f'| {a} − {b} | {n} | {g:+.2f} | {g - 1.96 * se:+.3f} to {g + 1.96 * se:+.3f} | {ao} | {bo} | {p:.3f} |')
    L += ['', f'Cost gate: {json.dumps(STATE.get("cost_gate"))}', '', '| stage | start | finish |', '|---|---|---|'] + [f'| {s["stage"]} | {s["start"]} | {s["finish"]} |' for s in STATE['stages']]
    disp = len((OUT / 'ledger.jsonl').read_text().splitlines()) if (OUT / 'ledger.jsonl').exists() else 0
    L += ['', f'Dispatched games (ledger): {disp} of max 200, plus 2 parity replays. Semantics/units are source-based (corroborated by flag/parity gates), not a binary proof.']
    L += ['', f'Exit {STATE.get("exit")}; finish {STATE.get("finish")}; orphans after: {STATE.get("orphans")}. Parity: out/parity-passed.json. Ledger: out/ledger.jsonl (completion order). '
          'Residual network dependency: priors (policy) and root/depth-cutoff values remain network; L=1 replaces only leaf values. Final seeds untouched.', '']
    if STATE['failure']: L += ['```', STATE.get('traceback', ''), '```']
    (OUT / 'REPORT.md').write_text('\n'.join(L)); (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str))


def on_term(*_):
    for p in list(ACTIVE): kill(p)
    raise Stop('SIGTERM (hard bound)')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['parity', 'main']); a = ap.parse_args()
    if a.command == 'parity': parity(); return
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    if (OUT / 'ledger.jsonl').exists(): raise SystemExit('refusing rerun')
    code = 0
    try: main_run()
    except BaseException as e:
        STATE['failure'] = f'{type(e).__name__}: {e}'[:1500]; STATE['traceback'] = traceback.format_exc()[-3000:]; code = 2
        for p in list(ACTIVE): kill(p)
    finally:
        time.sleep(1); STATE['orphans'] = [l for l in subprocess.run(['pgrep', '-af', 'pv_worker play'], capture_output=True, text=True).stdout.splitlines() if 'pgrep' not in l]
        STATE['finish'] = now(); STATE['exit'] = code
        try: report()
        except Exception: (OUT / 'REPORT.md').write_text(f'# REPORT failed\n\n{STATE.get("failure")}\n\n```\n{traceback.format_exc()}\n```\n'); code = code or 3
        log('REPORT written exit', code)
    sys.exit(code)


if __name__ == '__main__': main()
