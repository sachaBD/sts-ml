#!/usr/bin/env python3
"""Phase 6 final confirmation (FINAL_CONFIRMATION.md). Run from repo root: `confirm.py main` (or `report`).
Candidate: Phase5 strong-u3, frozen pv_worker 58c5c4ab `play MODEL 2000 --rollout-mix 0.5`. Reference: same worker `teacher 20000`.
600 paired fights (reserved final seeds, audited unused), 1200 games, 10 workers, no retries; error/cap pauses (partial report).
Primary: paired win-rate gap candidate − teacher, 95% CI (normal, paired) + exact McNemar. Success: lower 95% bound > 0.
Six consecutive 100-seed batches (frozen) reported descriptively."""
import argparse, concurrent.futures, hashlib, json, math, os, shutil, signal, statistics, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path.cwd(); assert (REPO / 'AGENTS.md').exists(), 'run from repo root'
D6 = REPO / 'runs/schema=combat_v4/date=2026-10-06'
RUN = D6 / 'id=demon-form-rollout-confirmation-v1'; OUT = RUN / 'out'
WORKER = D6 / 'id=single-deck-demon-form-v1/out/frozen/pv_worker'; WORKER_SHA = '58c5c4ab8d3dbbd6868a22bb86b467b08d3686fd8e0f368206b9b5368d83a6b0'
FINAL = D6 / 'id=single-deck-demon-form-v1/out/final-reserved.parquet'
CAND = D6 / 'id=rollout-expert-iteration-v1/out/strong/update3/model'
CAND_SHA = {'model.onnx': '49eeb12c382571f9fc401d836fbf3420daf270fcb4b1ee8f52c73934a5ffd02b',
            'model.onnx.data': 'b5c08a33ca879febd67a871f88ae26303e64d90abff963d1fcad721aa31a1ba2',
            'model.pt': 'a93944e12ce5ce654ffde62a640507d842f833e2e85069c6abf21b90a5e97ebd'}
AGENTS = {'candidate': ('rollout_mix=0.500000', 'pv model='), 'teacher': ('mcts leaf=guided_rollout sims=20000', 'mcts')}
WORKERS, TIMEOUT = 10, 900
ACTIVE = set()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def log(*x): print(now(), *x, flush=True)


class Stop(Exception): pass


def kill(p):
    if p and p.poll() is None:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def cmd_for(agent):
    w = str(OUT / 'frozen/pv_worker')
    return [w, 'play', str(OUT / 'frozen/model/model.onnx'), '2000', '--rollout-mix', '0.5'] if agent == 'candidate' else [w, 'teacher', '20000']


def play(agent, fight):
    cmd = cmd_for(agent); t0 = time.time(); p = None; err = None; out = se = ''
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True); ACTIVE.add(p)
        out, se = p.communicate(json.dumps({'fight_id': fight['fight_id'], 'start': fight['start']}) + '\n', timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        kill(p); err = f'{TIMEOUT}s timeout'
    finally:
        if p: ACTIVE.discard(p)
    rec = {'agent': agent, 'fight_id': fight['fight_id'], 'batch': fight['batch'], 'started': t0, 'finished': time.time(), 'seconds': time.time() - t0,
           'error': err, 'returncode': None if err else p.returncode, 'stderr': (se or '')[-1000:]}
    try: r = json.loads(out) if not err and p.returncode == 0 and out.count('\n') == 1 else None
    except json.JSONDecodeError: r = None
    if not r or r.get('status') not in ('completed', 'capped'): rec['status'] = 'error'; return rec
    rec['status'] = r['status']
    if r['status'] == 'completed':
        a = r['fight']['agent']; rec.update(won=bool(r['fight']['won']), hp=r['fight']['final_hp'], actions=len(r['fight']['actions']), agent_string=a)
        if not all(m in a for m in AGENTS[agent]): rec.update(status='error', error=f'agent string mismatch: {a}')
    return rec


STATE = {'start': None, 'failure': None, 'results': {'candidate': {}, 'teacher': {}}, 'statuses': {}}


def freeze():
    if sha(WORKER) != WORKER_SHA: raise Stop('worker sha')
    for n, h in CAND_SHA.items():
        if sha(CAND / n) != h: raise Stop(f'candidate {n} hash')
    fz = OUT / 'frozen'; (fz / 'model').mkdir(parents=True)
    shutil.copy2(WORKER, fz / 'pv_worker'); shutil.copy2(Path(__file__), fz / 'confirm.py')
    for n in CAND_SHA: shutil.copy2(CAND / n, fz / 'model' / n)
    rows = pq.read_table(FINAL).to_pylist()
    if len(rows) != 600 or [r['fight_id'] for r in rows] != [f'demon-form-fresh-v1:final:{i}' for i in range(600)]: raise Stop('final manifest order')
    if any((r['deck_id'], r['start']['hp'], r['start']['max_hp'], r['start']['potions']) != ('249e5246-153d-4030-bc11-598774146147', 34, 52, [1, 1]) for r in rows):
        raise Stop('final loadout')
    fights = [{'fight_id': r['fight_id'], 'start': r['start'], 'batch': i // 100} for i, r in enumerate(rows)]
    (OUT / 'fights.json').write_text(json.dumps(fights))
    man = {'preregistration': 'single primary comparison: paired win-rate gap candidate − teacher on 600 reserved final seeds; success iff 95% lower bound > 0 '
                              '(normal paired CI); exact McNemar also reported; 6 consecutive 100-seed batches (by final index 0–99, …, 500–599) descriptive only; '
                              'no early stopping, no retries, errors/caps pause and are reported, never relabelled',
           'final_source': str(FINAL), 'final_sha256': sha(FINAL), 'fights_sha256': sha(OUT / 'fights.json'),
           'batches': {b: [f['fight_id'] for f in fights if f['batch'] == b][::99] for b in range(6)},
           'candidate': {'model_dir': str(CAND), 'sha256': CAND_SHA, 'command': cmd_for('candidate')},
           'teacher': {'command': cmd_for('teacher')}, 'worker_sha256': WORKER_SHA, 'runner_sha256': sha(fz / 'confirm.py'),
           'audit': 'logs/audit-text-hits.txt (grep -F prefix+600 seeds over all repo text artifacts: 0 hits), logs/audit-parquet.json (1051 parquet '
                    'files under runs/ modified after 2026-10-06 09:08: 0 fight_id/seed/start.seed hits)', 'frozen_at': now()}
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1)); return fights


def main_run():
    STATE['start'] = now(); t0 = time.time(); fights = freeze(); STATE['manifest_sha256'] = sha(OUT / 'manifest.json')
    with open(OUT / 'ledger.jsonl', 'a') as led, concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        futs = {pool.submit(play, a, f): (a, f) for f in fights for a in ('candidate', 'teacher')}
        for n, fu in enumerate(concurrent.futures.as_completed(futs), 1):
            rec = fu.result(); rec['logged'] = now(); led.write(json.dumps(rec) + '\n'); led.flush(); os.fsync(led.fileno())
            STATE['statuses'][rec['status']] = STATE['statuses'].get(rec['status'], 0) + 1
            STATE['results'][rec['agent']][rec['fight_id']] = {k: rec.get(k) for k in ('status', 'won', 'seconds', 'batch')}
            if rec['status'] != 'completed':
                for x in futs: x.cancel()
                raise Stop(f'pause: {rec["agent"]} {rec["fight_id"]} {rec["status"]} {rec.get("error")} {rec["stderr"][-200:]}')
            if n % 100 == 0 or n == len(futs):
                log(f'{n}/{len(futs)}', {a: sum(bool(v.get('won')) for v in r.values()) for a, r in STATE['results'].items()})
    STATE['wall_seconds'] = time.time() - t0


def report():
    c, t = STATE['results']['candidate'], STATE['results']['teacher']
    cw = {k: v['won'] for k, v in c.items() if v['status'] == 'completed'}; tw = {k: v['won'] for k, v in t.items() if v['status'] == 'completed'}
    keys = sorted(set(cw) & set(tw)); n = len(keys)
    L = ['# Phase 6 final confirmation — REPORT', '', f'Generated {now()}; started {STATE["start"]}; outcome **{"COMPLETE" if not STATE["failure"] else "INCOMPLETE — " + STATE["failure"]}**.', '',
         'Candidate: Phase5 strong-u3, frozen pv_worker 58c5c4ab `play MODEL 2000 --rollout-mix 0.5` (greedy). Reference: same worker `teacher 20000` (MCTS, guided rollouts).',
         '600 reserved final seeds (demon-form-fresh-v1:final:0–599), 34/52 HP Demon Form deck, no potions. These seeds are now consumed.', '',
         f'Coverage: intended 600 pairs; candidate completed {len(cw)}, teacher completed {len(tw)}, complete pairs {n}; statuses {STATE["statuses"]}.', '']
    if n:
        d = np.array([int(cw[k]) - int(tw[k]) for k in keys], float); se = d.std(ddof=1) / math.sqrt(n)
        ao, bo = int((d > 0).sum()), int((d < 0).sum()); m = ao + bo
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(ao, bo) + 1)) / 2 ** m) if m else 1.0
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se; ok = n == 600 and lo > 0
        STATE['primary'] = {'pairs': n, 'candidate_wins': int(sum(cw[k] for k in keys)), 'teacher_wins': int(sum(tw[k] for k in keys)), 'gap': d.mean(),
                            'ci95': [lo, hi], 'candidate_only': ao, 'teacher_only': bo, 'mcnemar_exact_p': p, 'success': ok}
        L += ['## Primary (preregistered)', '', '| pairs | candidate wins | MCTS20k wins | gap | 95% CI | candidate-only | MCTS-only | exact McNemar p |', '|---|---|---|---|---|---|---|---|',
              f'| {n} | {STATE["primary"]["candidate_wins"]} | {STATE["primary"]["teacher_wins"]} | {d.mean():+.4f} | {lo:+.4f} to {hi:+.4f} | {ao} | {bo} | {p:.4g} |', '',
              f'Preregistered success (complete 600 pairs and lower 95% bound > 0): **{"MET" if ok else "NOT MET"}**.', '',
              '## Six consecutive 100-seed batches (descriptive only; not separate tests)', '', '| batch | pairs | candidate | MCTS | gap | cand-only/MCTS-only |', '|---|---|---|---|---|---|']
        for b in range(6):
            kb = [k for k in keys if c[k]['batch'] == b]; db = [int(cw[k]) - int(tw[k]) for k in kb]
            L.append(f'| {b} ({b * 100}–{b * 100 + 99}) | {len(kb)} | {sum(cw[k] for k in kb)} | {sum(tw[k] for k in kb)} | {np.mean(db) if db else float("nan"):+.2f} | '
                     f'{sum(x > 0 for x in db)}/{sum(x < 0 for x in db)} |')
    L += ['', '## Cost', '', '| agent | games | mean s | median s | total CPU s |', '|---|---|---|---|---|']
    for a, r in STATE['results'].items():
        s = [v['seconds'] for v in r.values()]
        if s: L.append(f'| {a} | {len(s)} | {statistics.mean(s):.2f} | {statistics.median(s):.2f} | {sum(s):.0f} |')
    L += ['', f'Wall s {STATE.get("wall_seconds")}; exit {STATE.get("exit")}; orphans {STATE.get("orphans")}; manifest sha256 {STATE.get("manifest_sha256")}. '
          '10 concurrent workers; system-load variability not quantified.', '',
          'Caveats: one training RNG, one fixed loadout (Demon Form 34/52, A20 Champ), conditional on this deck; not a general-recipe proof.', '']
    if STATE['failure']: L += ['```', STATE.get('traceback', ''), '```']
    (OUT / 'REPORT.md').write_text('\n'.join(L)); (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str))


def on_term(*_):
    for p in list(ACTIVE): kill(p)
    raise Stop('SIGTERM (hard bound)')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['main', 'report']); a = ap.parse_args()
    if a.command == 'report': STATE.update(json.loads((OUT / 'state.json').read_text())); report(); return
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    if (OUT / 'ledger.jsonl').exists() or (OUT / 'frozen').exists(): raise SystemExit('refusing rerun')
    OUT.mkdir(parents=True, exist_ok=True); code = 0
    try: main_run()
    except BaseException as e:
        STATE['failure'] = f'{type(e).__name__}: {e}'[:1500]; STATE['traceback'] = traceback.format_exc()[-3000:]; code = 2
        for p in list(ACTIVE): kill(p)
    finally:
        time.sleep(1); STATE['orphans'] = [l for l in subprocess.run(['pgrep', '-af', 'pv_worker'], capture_output=True, text=True).stdout.splitlines() if 'pgrep' not in l]
        STATE['finish'] = now(); STATE['exit'] = code
        try: report()
        except Exception: (OUT / 'REPORT.md').write_text(f'# REPORT failed\n\n{STATE.get("failure")}\n\n```\n{traceback.format_exc()}\n```\n'); code = code or 3
        log('REPORT written exit', code)
    sys.exit(code)


if __name__ == '__main__': main()
