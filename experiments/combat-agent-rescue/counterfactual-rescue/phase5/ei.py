#!/usr/bin/env python3
"""Phase 5: rollout-search expert iteration on fresh full combats (phase5/README.md). Run from repo root.

  smoke : zero-game re-encode parity (dev:0) + 1 L0.5 game encoded/loaded/evaluated
  main  : 3 updates x (200 fresh starts x {control L0, strong L0.5} -> encode -> train both) -> eval 4 cells x 100 dev -> REPORT
Frozen pv_worker 58c5c4ab (play/encode), Stage2 frozen train_fixed.py. Errors halt; no retries; REPORT always written.
"""
import argparse, concurrent.futures, copy, hashlib, json, math, os, shutil, signal, statistics, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path.cwd(); assert (REPO / 'AGENTS.md').exists(), 'run from repo root'
sys.path.insert(0, str(REPO))
from apps.human_champ import bench  # noqa: E402
from apps.run_rl.single_deck import select_loadout, generate  # noqa: E402

D5, D6 = REPO / 'runs/schema=combat_v4/date=2026-10-05', REPO / 'runs/schema=combat_v4/date=2026-10-06'
RUN = D6 / 'id=rollout-expert-iteration-v1'; OUT = RUN / 'out'
ORIG = D6 / 'id=single-deck-demon-form-v1/out'
WORKER = ORIG / 'frozen/pv_worker'; WORKER_SHA = '58c5c4ab8d3dbbd6868a22bb86b467b08d3686fd8e0f368206b9b5368d83a6b0'
S2 = D6 / 'id=counterfactual-rescue-v1/out/stage2'
TRAINER = S2 / 'frozen/train_fixed.py'
INIT = S2 / 'rescue/update3/model'
OLD_SHARDS = [ORIG / f'iter{i:03d}/rows/rows.parquet' for i in range(6, 16)] + [S2 / f'rescue/update{u}/new-rows.parquet' for u in (1, 2, 3)]
DEV = D6 / 'id=demon-form-decoupled-policy-v1/out/main'
DEV_REF = D6 / 'id=demon-form-budget-teacher-v1/out/per_fight.jsonl'
P4 = D6 / 'id=counterfactual-rollout-leaf-v1/out'
PY = REPO / '.venv/bin/python'
NS = 'rollout-expert-iteration-v1'
ARMS = {'control': None, 'strong': '0.5'}
MODES = {'L0': None, 'L0.5': '0.5'}
UPDATES, N, MIN_DONE, WORKERS, TIMEOUT = (1, 2, 3), 200, 180, 10, 900
ACTIVE = set()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def log(*x): print(now(), *x, flush=True)


class Stop(Exception): pass


def kill(p):
    if p and p.poll() is None:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def worker_bin(): return OUT / 'frozen/pv_worker' if (OUT / 'frozen/pv_worker').exists() else WORKER


def play(model, mix, fight_id, start):
    cmd = [str(worker_bin()), 'play', str(model), '2000'] + (['--rollout-mix', mix] if mix else [])
    t0 = time.time(); p = None; err = None; out = se = ''
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True); ACTIVE.add(p)
        out, se = p.communicate(json.dumps({'fight_id': fight_id, 'start': start}) + '\n', timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        kill(p); err = f'{TIMEOUT}s timeout'
    finally:
        if p: ACTIVE.discard(p)
    rec = {'fight_id': fight_id, 'cmd': cmd[1:], 'started': t0, 'finished': time.time(), 'seconds': time.time() - t0, 'error': err,
           'returncode': None if err else p.returncode, 'stderr': (se or '')[-1500:]}
    try: r = json.loads(out) if not err and p.returncode == 0 and out.count('\n') == 1 else None
    except json.JSONDecodeError: r = None
    if not r or r.get('status') not in ('completed', 'capped'): rec['status'] = 'error'; return rec, None
    rec['status'] = r['status']
    agent = (r.get('fight') or {}).get('agent') or (r['search'][0]['agent'] if r.get('search') else '')
    rec['agent'] = agent
    if ('rollout_mix=0.500000' in agent) != (mix == '0.5'): rec.update(status='error', error=f'flag/agent mismatch {agent}'); return rec, None
    if r['status'] == 'completed':
        f = r['fight']; rec.update(won=bool(f['won']), hp=f['final_hp'], actions=len(f['actions']))
        return rec, {'fight_id': fight_id, 'start': f['start'], 'actions': f['actions'], 'won': f['won'], 'final_hp': f['final_hp'],
                     'search': [{'step': s['step'], 'root_value': s['root_value'], 'children': s['children']} for s in r['search']]}
    return rec, None


def encode(lines, dest):
    tmp = Path(str(dest) + '.tmp')
    p = subprocess.run([str(worker_bin()), 'encode', str(tmp)], input=''.join(json.dumps(l) + '\n' for l in lines), text=True, capture_output=True, timeout=600)
    if p.returncode != 0 or not tmp.exists(): tmp.unlink(missing_ok=True); raise Stop(f'encode failed: {p.stderr[-800:]}')
    tmp.rename(dest)
    t = pq.read_table(dest, columns=['fight_id']); ids = set(t['fight_id'].to_pylist())
    if ids != {l['fight_id'] for l in lines}: raise Stop('encoded fight set differs')
    return t.num_rows


def used_seeds():
    cfg = json.loads((ORIG / 'config.json').read_text())
    files = [Path(cfg['source']), Path(cfg['selection']), ORIG / 'bootstrap.parquet', ORIG / 'monitor.parquet', ORIG / 'final-reserved.parquet',
             DEV / 'starts.parquet', D6 / 'id=demon-form-budget-teacher-v1/out/starts.parquet'] + sorted(ORIG.glob('iter*/starts.parquet'))
    used = set()
    for f in files: used |= {r['start']['seed'] for r in pq.read_table(f, columns=['start']).to_pylist()}
    used |= {r['start']['seed'] for r in json.loads((D6 / 'id=counterfactual-rescue-v1/out/starts.json').read_text())['rows']}
    return used, [str(f) for f in files], cfg


def make_starts():
    used, files, cfg = used_seeds(); n_used = len(used)
    base = select_loadout(pq.read_table(cfg['source']).to_pylist(), cfg['deck_id'])
    if (base['start']['hp'], base['start']['max_hp'], base['start']['potions'], cfg['deck_id']) != (34, 52, [1, 1], '249e5246-153d-4030-bc11-598774146147'):
        raise Stop('loadout is not the original 34/52 no-potion Demon Form deck')
    dev0 = pq.read_table(DEV / 'starts.parquet').to_pylist()[0]['start']
    out = {}
    for u in UPDATES:
        rows = generate(base, NS, f'train-u{u}', N, used); out[u] = rows
    seeds = [r['start']['seed'] for rows in out.values() for r in rows]
    if len(set(seeds)) != 3 * N or len(used) != n_used + 3 * N: raise Stop('seed collision/duplication')
    strip = lambda s: {k: v for k, v in s.items() if k not in ('seed', 'misc_rng', 'potion_rng')}
    if strip(out[1][0]['start']) != strip(dev0): raise Stop('generated start differs from dev start beyond seed/RNG fields')
    return out, {'excluded_seed_files': files, 'excluded_seeds': n_used, 'deck_id': cfg['deck_id'],
                 'hp': base['start']['hp'], 'max_hp': base['start']['max_hp']}


# ------------------------------------------------------------------ smoke
def smoke():
    OUT.mkdir(parents=True, exist_ok=True); sm = OUT / 'smoke'; sm.mkdir(exist_ok=True)
    if sha(WORKER) != WORKER_SHA: raise SystemExit('worker sha')
    # 1. zero-game re-encode parity of the recorded dev:0 baseline result line
    r = json.loads(open(DEV / 'baseline/results.jsonl').readline()); f = r['fight']
    line = {'fight_id': f['fight_id'], 'start': f['start'], 'actions': f['actions'], 'won': f['won'], 'final_hp': f['final_hp'],
            'search': [{'step': s['step'], 'root_value': s['root_value'], 'children': s['children']} for s in r['search']]}
    dest = sm / 'reencode.parquet'; dest.unlink(missing_ok=True); encode([line], dest)
    rows = lambda p: [{k: v for k, v in x.items() if k not in ('schema', 'date', 'id')} for x in pq.read_table(p).to_pylist()]
    ref = [x for x in rows(DEV / 'baseline/encoded/rows.parquet') if x['fight_id'] == f['fight_id']]
    got = rows(dest)
    diffs = [(i, k) for i, (a, b) in enumerate(zip(got, ref)) for k in a if a[k] != b[k]]
    print('CHECK reencode_parity', len(got) == len(ref) and not diffs, diffs[:5], flush=True)
    if len(got) != len(ref) or diffs: raise SystemExit('re-encode parity failed')
    # 2. starts constructor/exclusion (no games)
    starts, meta = make_starts(); print('CHECK starts', meta, flush=True)
    # 3. one L0.5 game on a fresh start, encoded, loaded, evaluated
    led = sm / 'smoke-ledger.jsonl'
    if led.exists(): raise SystemExit('smoke game already dispatched; no retry')
    s0 = starts[1][0]; rec, line = play(INIT / 'model.onnx', '0.5', f'rei5:smoke:{s0["fight_id"]}', s0['start'])
    led.write_text(json.dumps(rec) + '\n')
    if rec['status'] != 'completed': raise SystemExit(f'smoke game {rec}')
    g = sm / 'game.parquet'; encode([line], g)
    import torch
    from agents.combat.pv.data import Shard
    from agents.combat.pv.model import PolicyValue
    from agents.combat.pv.train import evaluate
    sh = Shard(g, False, {line['fight_id']: 'train'}); b = sh.batch(sh.rows)
    ck = torch.load(INIT / 'model.pt', map_location='cpu', weights_only=False); net = PolicyValue(ck['width'], ck['value_activation']); net.load_state_dict(ck['state_dict']); net.eval()
    with torch.no_grad(): v, q, _, n, npol = evaluate(net, b, 'cpu', flat_policy_weighting=True)
    ok = torch.isfinite(v) and torch.isfinite(q) and (b[1] == 100 * rec['won']).all() and len(sh.rows) == rec['actions']
    print('CHECK smoke_game_rows', bool(ok), {'won': rec['won'], 'rows': len(sh.rows), 'agent': rec['agent'], 'seconds': rec['seconds']}, flush=True)
    if not ok: raise SystemExit('smoke rows failed')
    (OUT / 'smoke-passed.json').write_text(json.dumps({'at': now(), 'games': 1, 'starts_meta': meta}, indent=1))


# ------------------------------------------------------------------ main
STATE = {'start': None, 'stages': [], 'last_success': None, 'failure': None, 'collect': {}, 'train': {}, 'eval': {}}


def stage_done(name, t0, **kw):
    STATE['stages'].append({'stage': name, 'start': t0, 'finish': now(), **kw}); STATE['last_success'] = name
    (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str)); log('STAGE DONE', name)


def run_jobs(pool, led, stage, jobs):
    """jobs: list of (meta dict, model, mix, fight_id, start). Halts on first error/flag mismatch, cancelling pending."""
    futs = {pool.submit(play, m, mix, fid, st): meta for meta, m, mix, fid, st in jobs}; out = []
    for n, fu in enumerate(concurrent.futures.as_completed(futs), 1):
        meta = futs[fu]; rec, line = fu.result(); rec.update(meta, stage=stage, logged=now())
        led.write(json.dumps(rec) + '\n'); led.flush(); os.fsync(led.fileno()); out.append((rec, line))
        if rec['status'] == 'error':
            for x in futs: x.cancel()
            raise Stop(f'{stage}: error {rec["fight_id"]} {rec.get("error")} {rec["stderr"][-300:]}')
        if n % 50 == 0 or n == len(jobs): log(stage, f'{n}/{len(jobs)}')
    return out


def main_run():
    STATE['start'] = now(); t_start = time.time()
    if not (OUT / 'smoke-passed.json').exists(): raise Stop('smoke not passed')
    fz = OUT / 'frozen'; fz.mkdir()
    for src, n in ((WORKER, 'pv_worker'), (TRAINER, 'train_fixed.py'), (Path(__file__), 'ei.py')): shutil.copy2(src, fz / n)
    starts, meta = make_starts()
    for u, rows in starts.items():
        (OUT / f'starts-u{u}.json').write_text(json.dumps(rows))
    for p in OLD_SHARDS:
        if not p.exists(): raise Stop(f'missing old shard {p}')
    man = {'frozen_sha256': {n: sha(fz / n) for n in ('pv_worker', 'train_fixed.py', 'ei.py')}, 'init_model_pt': sha(INIT / 'model.pt'),
           'init_onnx': sha(INIT / 'model.onnx'), 'init_onnx_data': sha(INIT / 'model.onnx.data'), 'old_shards': {str(p): sha(p) for p in OLD_SHARDS},
           'starts_sha256': {u: sha(OUT / f'starts-u{u}.json') for u in UPDATES}, 'starts_meta': meta, 'dev_starts_sha256': sha(DEV / 'starts.parquet'),
           'arms': {a: f'play MODEL 2000' + (f' --rollout-mix {m}' if m else '') for a, m in ARMS.items()},
           'eval_cells': {f'{a}-{k}': f'endpoint u3 {a} model, play 2000' + (f' --rollout-mix {m}' if m else '') for a in ARMS for k, m in MODES.items()},
           'training': 'Stage2 train_fixed.py: 1000 steps x (32 old of 1432 fights + 32 new own fresh fights), lr3e-4 wd.01 clip1, Adam resume, outcome 100/0, '
                       'concentration-weighted visits; old RNG SeedSequence([0,u,0]) new ([0,u,1]); endpoint u3 only', 'frozen_at': now()}
    if man['frozen_sha256']['pv_worker'] != WORKER_SHA: raise Stop('worker sha')
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1)); STATE['manifest_sha256'] = sha(OUT / 'manifest.json')
    models = {a: INIT for a in ARMS}; new_files = {a: [] for a in ARMS}; n_new = {a: 0 for a in ARMS}
    with open(OUT / 'ledger.jsonl', 'a') as led, concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for u in UPDATES:
            t0 = now()
            jobs = [({'arm': a, 'update': u}, models[a] / 'model.onnx', mix, f'rei5:{a}:u{u}:{i}', s['start'])
                    for i, s in enumerate(starts[u]) for a, mix in ARMS.items()]
            res = run_jobs(pool, led, f'collect-u{u}', jobs)
            for a in ARMS:
                mine = [(r, l) for r, l in res if r['arm'] == a]; done = [l for r, l in mine if r['status'] == 'completed']
                if len(done) < MIN_DONE: raise Stop(f'{a} u{u}: {len(done)}/200 completed')
                d = OUT / a / f'update{u}'; d.mkdir(parents=True, exist_ok=True)
                rows = encode(sorted(done, key=lambda l: l['fight_id']), d / 'rows.parquet')
                new_files[a].append(d / 'rows.parquet'); n_new[a] += len(done)
                STATE['collect'][f'{a}-u{u}'] = {'intended': 200, 'completed': len(done), 'capped': sum(r['status'] == 'capped' for r, _ in mine),
                                                 'wins': sum(bool(r.get('won')) for r, _ in mine), 'rows': rows, 'cpu_s': sum(r['seconds'] for r, _ in mine)}
            stage_done(f'collect-u{u}', t0)
            t0 = now(); procs = []
            for a in ARMS:
                d = OUT / a / f'update{u}'
                ids = []
                for f in OLD_SHARDS + new_files[a]: ids += list(set(pq.read_table(f, columns=['fight_id'])['fight_id'].to_pylist()))
                if len(ids) != len(set(ids)): raise Stop('duplicate fight ids')
                (d / 'split.json').write_text(json.dumps({i: 'train' for i in ids}))
                cmd = [str(PY), str(fz / 'train_fixed.py'), '--old', *map(str, OLD_SHARDS), '--new', *map(str, new_files[a]), '--split', str(d / 'split.json'),
                       '--init', str(models[a] / 'model.pt'), '--out', str(d / 'model'), '--update', str(u), '--expect-old-fights', '1432',
                       '--expect-new-fights', str(n_new[a])]
                lf = open(d / 'train.log', 'w'); p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); ACTIVE.add(p); procs.append((a, p, lf, d))
            for a, p, lf, d in procs:
                p.wait(); lf.close(); ACTIVE.discard(p)
                if p.returncode: raise Stop(f'train {a} u{u} rc={p.returncode}: {(d / "train.log").read_text()[-800:]}')
                rep = json.loads((d / 'model/train.json').read_text())
                if rep['counts'] != {'old': 32000, 'new': 32000, 'steps': 1000} or rep['old_fights'] != 1432 or rep['test_override_steps']: raise Stop('locked counts')
                STATE['train'][f'{a}-u{u}'] = {k: rep[k] for k in ('counts', 'old_fights', 'new_fights', 'new_states', 'adam_step_before', 'adam_step_after',
                                                                   'onnx_max_abs_diff', 'files_sha256', 'exported_at', 'train_seconds', 'init_sha256')}
                STATE['train'][f'{a}-u{u}']['final_log'] = rep['log'][-1]
                models[a] = d / 'model'
            stage_done(f'train-u{u}', t0)
        t0 = now(); dev = pq.read_table(DEV / 'starts.parquet').to_pylist()
        jobs = [({'cell': f'{a}-{k}', 'arm': a, 'mode': k}, models[a] / 'model.onnx', mix, f['fight_id'], f['start'])
                for f in dev for a in ARMS for k, mix in MODES.items()]
        res = run_jobs(pool, led, 'eval', jobs)
        for r, _ in res:
            c = STATE['eval'].setdefault(r['cell'], {'won': {}, 'statuses': {}, 'seconds': []})
            c['statuses'][r['status']] = c['statuses'].get(r['status'], 0) + 1; c['seconds'].append(r['seconds'])
            if r['status'] == 'completed': c['won'][r['fight_id']] = r['won']
        stage_done('eval', t0)
    STATE['main_wall_seconds'] = time.time() - t_start


def paired(a, b):
    keys = sorted(set(a) & set(b)); n = len(keys); d = np.array([int(a[k]) - int(b[k]) for k in keys], float)
    se = d.std(ddof=1) / math.sqrt(n); ao, bo = int((d > 0).sum()), int((d < 0).sum()); m = ao + bo
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(ao, bo) + 1)) / 2 ** m) if m else 1.0
    return n, d.mean(), se, ao, bo, p


def wilson(k, n):
    z = 1.96; ph = k / n; c = (ph + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def report():
    L = ['# Phase 5 rollout-search expert iteration — REPORT', '', f'Generated {now()}; started {STATE["start"]}; last successful stage **{STATE["last_success"]}**; '
         f'outcome **{"COMPLETE" if not STATE["failure"] else "INCOMPLETE — " + STATE["failure"]}**.', '']
    cells = {k: v['won'] for k, v in STATE['eval'].items()}
    ref = [json.loads(l) for l in DEV_REF.read_text().splitlines()]; cells['mcts_20k (cached)'] = {x['fight_id']: x['won_teacher'] for x in ref}
    p4 = [json.loads(l) for l in (P4 / 'ledger.jsonl').read_text().splitlines()] if (P4 / 'ledger.jsonl').exists() else []
    cells['rescue-u3 L0.5 (Phase4)'] = {x['fight_id']: x['won'] for x in p4 if x.get('arm') == 'mix0.5' and x['status'] == 'completed'}
    s2 = [json.loads(l) for l in (S2 / 'ledger.jsonl').read_text().splitlines()]
    cells['rescue-u3 L0 (Stage2)'] = {x['job']['fight_id']: x['won'] for x in s2 if x['stage'] == 'dev-eval' and '/rescue/update3/' in x['job']['model']}
    if STATE['eval']:
        L += ['## Endpoint (update 3) dev evaluation, 100 dev starts, greedy 2k', '', '| cell | wins/completed (intended 100) | Wilson 95% | caps | mean/median s |', '|---|---|---|---|---|']
        for k, w in cells.items():
            n = len(w); lo, hi = wilson(sum(w.values()), n) if n else (float('nan'),) * 2; e = STATE['eval'].get(k)
            L.append(f'| {k} | {sum(w.values())}/{n} | {lo:.2f}–{hi:.2f} | {e["statuses"].get("capped", 0) if e else "rec."} | '
                     + (f'{statistics.mean(e["seconds"]):.1f}/{statistics.median(e["seconds"]):.1f} |' if e else 'rec. |'))
        L += ['', 'Primary: strong-L0.5 vs control-L0.5, vs MCTS74, vs untrained rescue L0.5 (76). Secondary: L0 cells.', '',
              '| paired (a − b) | pairs | gap | approx 95% CI | a-only | b-only | exact McNemar p |', '|---|---|---|---|---|---|---|']
        for a, b in [('strong-L0.5', 'control-L0.5'), ('strong-L0.5', 'mcts_20k (cached)'), ('strong-L0.5', 'rescue-u3 L0.5 (Phase4)'),
                     ('control-L0.5', 'rescue-u3 L0.5 (Phase4)'), ('strong-L0', 'control-L0'), ('strong-L0', 'rescue-u3 L0 (Stage2)'),
                     ('control-L0', 'rescue-u3 L0 (Stage2)'), ('strong-L0', 'mcts_20k (cached)')]:
            if cells.get(a) and cells.get(b):
                n, g, se, ao, bo, p = paired(cells[a], cells[b])
                L.append(f'| {a} − {b} | {n} | {g:+.2f} | {g - 1.96 * se:+.3f} to {g + 1.96 * se:+.3f} | {ao} | {bo} | {p:.3f} |')
        sw = sum(cells.get('strong-L0.5', {}).values())
        L += ['', f'Descriptive gate (>=80/100 strong-L0.5, not a test): **{"reached" if sw >= 80 else "not reached"}** ({sw}).', '']
    if STATE['collect']:
        L += ['## Collection (fresh starts, same 200 per update for both arms)', '', '| arm-update | completed/intended | caps | wins | rows | CPU s |', '|---|---|---|---|---|---|']
        L += [f'| {k} | {c["completed"]}/{c["intended"]} | {c["capped"]} | {c["wins"]} | {c["rows"]} | {c["cpu_s"]:.0f} |' for k, c in STATE['collect'].items()]
    if STATE['train']:
        L += ['', '## Training', '', '| arm-update | counts | old/new fights | new states | Adam | final loss v/p | ONNX diff | s |', '|---|---|---|---|---|---|---|---|']
        L += [f'| {k} | {t["counts"]["old"]}/{t["counts"]["new"]}/{t["counts"]["steps"]} | {t["old_fights"]}/{t["new_fights"]} | {t["new_states"]} | '
              f'{t["adam_step_before"]}→{t["adam_step_after"]} | {t["final_log"]["value_loss"]:.3f}/{t["final_log"]["policy_loss"]:.3f} | {t["onnx_max_abs_diff"]:.1e} | {t["train_seconds"]:.0f} |'
              for k, t in STATE['train'].items()]
    L += ['', '| stage | start | finish |', '|---|---|---|'] + [f'| {s["stage"]} | {s["start"]} | {s["finish"]} |' for s in STATE['stages']]
    disp = len((OUT / 'ledger.jsonl').read_text().splitlines()) if (OUT / 'ledger.jsonl').exists() else 0
    L += ['', f'Dispatched games: {disp} of cap 1600 (+1 smoke). Main wall s {STATE.get("main_wall_seconds")}; exit {STATE.get("exit")}; orphans {STATE.get("orphans")}; '
          f'manifest sha256 {STATE.get("manifest_sha256")}. Development starts reused (not confirmation). Final seeds excluded, never played.', '']
    if STATE['failure']: L += ['```', STATE.get('traceback', ''), '```']
    (OUT / 'REPORT.md').write_text('\n'.join(L)); (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str))


def on_term(*_):
    for p in list(ACTIVE): kill(p)
    raise Stop('SIGTERM (hard bound)')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['smoke', 'main', 'report']); a = ap.parse_args()
    if a.command == 'smoke': smoke(); return
    if a.command == 'report': STATE.update(json.loads((OUT / 'state.json').read_text())); report(); return
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    if (OUT / 'ledger.jsonl').exists() or (OUT / 'frozen').exists(): raise SystemExit('refusing rerun')
    code = 0
    try: main_run()
    except BaseException as e:
        STATE['failure'] = f'{type(e).__name__}: {e}'[:1500]; STATE['traceback'] = traceback.format_exc()[-3000:]; code = 2
        for p in list(ACTIVE): kill(p)
    finally:
        time.sleep(1); STATE['orphans'] = [l for l in subprocess.run(['pgrep', '-af', 'pv_worker|train_fixed'], capture_output=True, text=True).stdout.splitlines() if 'pgrep' not in l]
        STATE['finish'] = now(); STATE['exit'] = code
        try: report()
        except Exception: (OUT / 'REPORT.md').write_text(f'# REPORT failed\n\n{STATE.get("failure")}\n\n```\n{traceback.format_exc()}\n```\n'); code = code or 3
        log('REPORT written exit', code)
    sys.exit(code)


if __name__ == '__main__': main()
