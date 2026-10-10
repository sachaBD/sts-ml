#!/usr/bin/env python3
"""Stage 2 chained runner: smoke | main.  Locked design: stage2/PROTOCOL.md (+ README locked decisions).

main: for update u=1..3: collect control+rescue train episodes (144/arm) -> merge rows -> train both arms
(train_fixed.py, 1000 steps x 32 old + 32 new) -> rescue probe 48 at the current primary level -> allocation.
Then dev eval of both update-3 endpoints (frozen pv_worker play 2k, 100 dev starts each) -> REPORT.md.
A report is always written, also on failure (last successful stage, partial ledger). No retries.
"""
import argparse, concurrent.futures, hashlib, itertools, json, math, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path.cwd()  # run from the repository root (asserted); works from the frozen copy too
assert (REPO / 'AGENTS.md').exists(), 'run from repo root'
HERE = Path(__file__).resolve().parent
D6 = REPO / 'runs/schema=combat_v4/date=2026-10-06'
ROOT = D6 / 'id=counterfactual-rescue-v1'
S1 = ROOT / 'out'
OUT = ROOT / 'out/stage2'
ORIG = D6 / 'id=single-deck-demon-form-v1/out'
ORIG_MODEL = ORIG / 'iter015/model'
OLD_SHARDS = [ORIG / f'iter{i:03d}/rows/rows.parquet' for i in range(6, 16)]
DEV = D6 / 'id=demon-form-decoupled-policy-v1/out/main'
DEV_STARTS = DEV / 'starts.parquet'
DEV_REF = D6 / 'id=demon-form-budget-teacher-v1/out/per_fight.jsonl'
WORKER = ORIG / 'frozen/pv_worker'
WORKER_SHA = '58c5c4ab8d3dbbd6868a22bb86b467b08d3686fd8e0f368206b9b5368d83a6b0'
BUILD_BIN = REPO / 'build/continuation-v1/pv_continuation'
PY = REPO / '.venv/bin/python'

ARMS, UPDATES, LEVELS, KINDS = ('control', 'rescue'), (1, 2, 3), ('full', 'early', 'middle', 'late'), ('train', 'probe')
BASE, BLOCK = 1000, 16
SMOKE_INDEX = 900  # smoke-only particles 900..903: outside Stage1 (0..3) and all main blocks (>=1000)
EPISODES, PROBES, THRESHOLD, MIN_DONE, MIN_PROBE = 12, 4, 35, 130, 44
WORKERS, JOB_TIMEOUT = 10, 300
EARLIER = {'middle': 'early', 'early': 'full'}
ACTIVE = set()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def log(*x): print(now(), *x, flush=True)


class Stop(Exception): pass


def block(arm, update, path, level, kind):
    """Disjoint actual particle-index range for every (arm, update, path, level, train/probe)."""
    b = list(itertools.product(ARMS, UPDATES, range(12), LEVELS, KINDS)).index((arm, update, path, level, kind))
    return range(BASE + BLOCK * b, BASE + BLOCK * (b + 1))


def paths():
    starts = json.loads((S1 / 'starts.json').read_text())
    rows = starts['rows']
    if len(rows) != 12: raise Stop('need 12 pilot paths')
    out = []
    for r in rows:
        lv = {'full': []}; lv.update({x['label']: x['prefix_actions'] for x in r['restarts']})
        if set(lv) != set(LEVELS): raise Stop(f'missing restart level in {r["fight_id"]}')
        out.append({'fight_id': r['fight_id'], 'start': r['start'], 'ops': lv})
    return out


# ---------------------------------------------------------------- dispatch
def kill_group(proc):
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def run_one(command, line, timeout=JOB_TIMEOUT):
    started = time.time(); proc = None
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        ACTIVE.add(proc)
        stdout, stderr = proc.communicate(line + '\n', timeout=timeout)
        return proc.returncode, stdout, stderr[-2000:], started, None
    except subprocess.TimeoutExpired:
        kill_group(proc); return None, '', '', started, f'{timeout}s timeout'
    finally:
        if proc: ACTIVE.discard(proc)


def continuation(binary, model, job):
    """One adapter learner continuation; status completed/capped/error, never a false label."""
    req = {k: job[k] for k in ('start', 'ops', 'particles', 'fight_id') if k in job}
    req.update(mode='learner', sims=2000, true_state=job.get('true_state', False), record=False)
    if job.get('shard_path'): req['shard_path'] = job['shard_path']
    if 'turn_cap' in job: req['turn_cap'] = job['turn_cap']
    rc, stdout, stderr, started, err = run_one([str(binary), str(model)], json.dumps(req))
    meta = {k: v for k, v in job.items() if k not in ('start', 'ops')}
    rec = {'job': meta, 'started': started, 'finished': time.time(), 'returncode': rc, 'stderr': stderr, 'error': err}
    rec['seconds'] = rec['finished'] - started
    try:
        resp = json.loads(stdout) if rc == 0 and stdout.count('\n') == 1 else None
    except json.JSONDecodeError:
        resp = None
    res = resp.get('results', []) if resp and not resp.get('error') else []
    if len(res) != 1 or res[0].get('status') not in ('completed', 'capped'):
        rec.update(status='error', response=resp); return rec
    r = res[0]
    rec.update(status=r['status'], won=r['won'] if r['status'] == 'completed' else None, hp=r['hp'], turns=r['turns'],
               actions=r['continuation_actions'], root_fingerprint=r['root_fingerprint'], true_fingerprint=resp['true_fingerprint'],
               legal_actions=resp['legal_actions'], particle=r['particle'])
    sp = job.get('shard_path')
    if sp:
        published, tmp = Path(sp).exists(), Path(sp + '.tmp').exists()
        rec['shard_published'] = published
        if tmp or published != (r['status'] == 'completed'):
            rec['status'] = 'error'; rec['error'] = f'shard publication mismatch published={published} tmp={tmp}'
    return rec


def dev_fight(model, fight):
    rc, stdout, stderr, started, err = run_one([str(WORKER), 'play', str(model), '2000'],
                                               json.dumps({'fight_id': fight['fight_id'], 'start': fight['start']}))
    rec = {'job': {'fight_id': fight['fight_id'], 'model': str(model)}, 'started': started, 'finished': time.time(),
           'returncode': rc, 'stderr': stderr, 'error': err}
    rec['seconds'] = rec['finished'] - started
    try:
        resp = json.loads(stdout) if rc == 0 and stdout.count('\n') == 1 else None
    except json.JSONDecodeError:
        resp = None
    if not resp or resp.get('status') not in ('completed', 'capped'):
        rec.update(status='error'); return rec
    rec['status'] = resp['status']
    if resp['status'] == 'completed':
        rec.update(won=bool(resp['fight']['won']), hp=resp['fight']['final_hp'], actions=len(resp['fight']['actions']))
    return rec


class Ledger:
    def __init__(self, path):
        self.f = open(path, 'a'); self.counts = {}

    def write(self, stage, rec):
        rec = dict(rec, stage=stage, logged=now())
        self.f.write(json.dumps(rec) + '\n'); self.f.flush(); os.fsync(self.f.fileno())
        c = self.counts.setdefault(stage, {'completed': 0, 'capped': 0, 'error': 0, 'won': 0})
        c[rec['status']] += 1; c['won'] += bool(rec.get('won'))


def dispatch(pool, ledger, stage, fn_jobs):
    futures = [pool.submit(fn, job) for fn, job in fn_jobs]
    out = []
    for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
        rec = fut.result(); ledger.write(stage, rec); out.append(rec)
        if n % 24 == 0 or n == len(futures): log(stage, f'{n}/{len(futures)}', ledger.counts[stage])
    errors = [r for r in out if r['status'] == 'error']
    if errors: raise Stop(f'{stage}: {len(errors)} errors, e.g. {errors[0].get("error") or errors[0].get("stderr", "")[-300:]}')
    return out


# ---------------------------------------------------------------- data
def merge_rows(files, dest):
    tables = [pq.read_table(f) for f in files]
    meta = tables[0].schema.metadata
    if any(t.schema.metadata.get(b'pv_contract') != meta.get(b'pv_contract') for t in tables): raise Stop('contract mismatch')
    t = pa.concat_tables(tables).replace_schema_metadata(meta)
    tmp = Path(str(dest) + '.tmp'); pq.write_table(t, tmp); tmp.rename(dest)
    return t.num_rows


def write_split(dest, new_ids):
    old = []
    for p in OLD_SHARDS: old += list(set(pq.read_table(p, columns=['fight_id'])['fight_id'].to_pylist()))
    if len(old) != 1000 or len(set(old)) != 1000: raise Stop('old half must be exactly 1000 fights')
    if set(old) & set(new_ids): raise Stop('old/new fight id overlap')
    dest.write_text(json.dumps({fid: 'train' for fid in old + list(new_ids)}))


def train(arm, update, init, new_files, n_new, test_steps=None):
    d = OUT / arm / f'update{update}'
    split = d / 'split.json'
    ids = []
    for f in new_files: ids += list(set(pq.read_table(f, columns=['fight_id'])['fight_id'].to_pylist()))
    write_split(split, ids)
    cmd = [str(PY), str(OUT / 'frozen/train_fixed.py'), '--old', *map(str, OLD_SHARDS), '--new', *map(str, new_files),
           '--split', str(split), '--init', str(init), '--out', str(d / 'model'), '--update', str(update),
           '--expect-old-fights', '1000', '--expect-new-fights', str(n_new)]
    if test_steps: cmd += ['--test-steps', str(test_steps)]
    return cmd, d


def run_trainings(cmds):
    procs = []
    for cmd, d in cmds:
        f = open(d / 'train.log', 'w'); procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True), d, f))
        ACTIVE.add(procs[-1][0])
    bad = []
    for p, d, f in procs:
        p.wait(); f.close(); ACTIVE.discard(p)
        if p.returncode != 0: bad.append((str(d), p.returncode, (d / 'train.log').read_text()[-1500:]))
    if bad: raise Stop(f'training failed: {bad}')
    return [json.loads((d / 'model/train.json').read_text()) for _, d in cmds]


# ---------------------------------------------------------------- stats
def paired(a, b):
    """a, b: dict fight -> bool. Paired difference a-b with normal approx 95% CI, discordance, exact McNemar p."""
    keys = sorted(set(a) & set(b)); n = len(keys)
    if not n: return {'pairs': 0, 'gap': float('nan'), 'se': float('nan'), 'ci95': [float('nan')] * 2, 'a_only': 0, 'b_only': 0,
                      'both_won': 0, 'both_lost': 0, 'mcnemar_exact_p': float('nan')}
    d = np.array([int(a[k]) - int(b[k]) for k in keys], float)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan')
    only_a, only_b = int((d > 0).sum()), int((d < 0).sum()); m = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** m) if m else 1.0
    both = sum(a[k] and b[k] for k in keys)
    return {'pairs': n, 'gap': d.mean(), 'se': se, 'ci95': [d.mean() - 1.96 * se, d.mean() + 1.96 * se],
            'a_only': only_a, 'b_only': only_b, 'both_won': both, 'both_lost': n - both - m, 'mcnemar_exact_p': p}


def wilson(k, n):
    if not n: return [float('nan')] * 2
    z, ph = 1.96, k / n; c = (ph + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


# ---------------------------------------------------------------- main
STATE = {'stages': [], 'last_success': None, 'start': None, 'allocation': [], 'probes': [], 'train': {}, 'collect': {},
         'eval': {}, 'timing': {}, 'failure': None}


def stage_done(name, t0):
    STATE['stages'].append({'stage': name, 'start': t0, 'finish': now()}); STATE['last_success'] = name
    (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str))
    log('STAGE DONE', name)


def main_run():
    STATE['start'] = now(); t_start = time.time()
    binary, model0 = OUT / 'frozen/pv_continuation', ORIG_MODEL
    if not (OUT / 'smoke-passed.json').exists(): raise Stop('smoke has not passed')
    hashes = json.loads((OUT / 'frozen/sha256.json').read_text())
    for name, h in hashes.items():
        if sha(OUT / 'frozen' / name) != h: raise Stop(f'frozen file changed: {name}')
    if sha(WORKER) != WORKER_SHA: raise Stop('frozen worker sha mismatch')
    P = paths()
    # disjointness of every main range (and Stage1 0..3 / smoke 900..903)
    seen = set()
    for key in itertools.product(ARMS, UPDATES, range(12), LEVELS, KINDS):
        r = set(block(*key)); assert not (r & seen) and min(r) >= BASE; seen |= r
    if len(seen) != 2 * 3 * 12 * 4 * 2 * BLOCK: raise Stop('range layout')
    ledger = Ledger(OUT / 'ledger.jsonl')
    models = {arm: model0 for arm in ARMS}  # directory with model.onnx / model.pt
    alloc = {'middle': 9, 'late': 3}; primary = 'middle'
    new_files = {arm: [] for arm in ARMS}; n_new = {arm: 0 for arm in ARMS}
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for u in UPDATES:
            t0 = now(); STATE['allocation'].append({'update': u, 'allocation': dict(alloc), 'primary': primary})
            jobs = []
            for arm in ARMS:
                ep = OUT / arm / f'update{u}' / 'episodes'; ep.mkdir(parents=True, exist_ok=True)
                plan = {'full': EPISODES} if arm == 'control' else alloc
                for pi, p in enumerate(P):
                    for level, n in plan.items():
                        for idx in list(block(arm, u, pi, level, 'train'))[:n]:
                            fid = f'cfr2:{arm}:u{u}:p{pi:02d}:{level}:train:{idx}'
                            jobs.append((lambda j, m=models[arm]: continuation(binary, m / 'model.onnx', j),
                                         {'arm': arm, 'update': u, 'path': pi, 'level': level, 'kind': 'train', 'fight_id': fid,
                                          'start': p['start'], 'ops': p['ops'][level], 'particles': [idx],
                                          'shard_path': str(ep / f'{fid.replace(":", "_")}.parquet')}))
            keys = [j['fight_id'] for _, j in jobs]
            if len(keys) != len(set(keys)) or len(keys) != 2 * 12 * EPISODES: raise Stop('job keys')
            recs = dispatch(pool, ledger, f'collect-u{u}', jobs)
            for arm in ARMS:
                mine = [r for r in recs if r['job']['arm'] == arm]
                done = [r for r in mine if r['status'] == 'completed']
                if len(done) < MIN_DONE: raise Stop(f'{arm} u{u}: only {len(done)}/144 completed')
                div = {}
                for r in mine: div.setdefault((r['job']['path'], r['job']['level']), []).append(r['root_fingerprint'])
                diversity = {f'p{k[0]:02d}:{k[1]}': [len(set(v)), len(v)] for k, v in sorted(div.items())}
                if arm == 'control' and any(a != b for a, b in diversity.values()):
                    raise Stop(f'control u{u}: duplicate actual roots {diversity}')
                dest = OUT / arm / f'update{u}' / 'new-rows.parquet'
                rows = merge_rows([r['job']['shard_path'] for r in sorted(done, key=lambda r: r['job']['fight_id'])], dest)
                new_files[arm].append(dest); n_new[arm] += len(done)
                STATE['collect'][f'{arm}-u{u}'] = {'intended': 144, 'completed': len(done), 'capped': sum(r['status'] == 'capped' for r in mine),
                    'wins': sum(bool(r.get('won')) for r in done), 'rows': rows, 'root_diversity': diversity,
                    'by_level': {lv: [sum(bool(r.get('won')) for r in done if r['job']['level'] == lv),
                                      sum(1 for r in done if r['job']['level'] == lv), sum(1 for r in mine if r['job']['level'] == lv)]
                                 for lv in LEVELS if any(r['job']['level'] == lv for r in mine)},
                    'dispatch_seconds': sum(r['seconds'] for r in mine)}
            stage_done(f'collect-u{u}', t0)
            t0 = now()
            cmds = [train(arm, u, models[arm] / 'model.pt', new_files[arm], n_new[arm]) for arm in ARMS]
            if u == 1 and sha(cmds[0][0][cmds[0][0].index('--init') + 1]) != sha(cmds[1][0][cmds[1][0].index('--init') + 1]):
                raise Stop('arms do not share the initial checkpoint')
            reports = run_trainings(cmds)
            for arm, rep, (_, d) in zip(ARMS, reports, cmds):
                if rep['counts'] != {'old': 32000, 'new': 32000, 'steps': 1000} or rep['test_override_steps']: raise Stop('locked counts')
                STATE['train'][f'{arm}-u{u}'] = {k: rep[k] for k in ('counts', 'new_fights', 'new_states', 'old_fights', 'adam_step_before',
                    'adam_step_after', 'onnx_max_abs_diff', 'files_sha256', 'saved_at', 'exported_at', 'train_seconds', 'init_sha256', 'rng_seed_sequences')}
                STATE['train'][f'{arm}-u{u}']['final_log'] = rep['log'][-1]
                models[arm] = d / 'model'
            stage_done(f'train-u{u}', t0)
            t0 = now(); jobs = []
            for pi, p in enumerate(P):
                for idx in list(block('rescue', u, pi, primary, 'probe'))[:PROBES]:
                    jobs.append((lambda j, m=models['rescue']: continuation(binary, m / 'model.onnx', j),
                                 {'arm': 'rescue', 'update': u, 'path': pi, 'level': primary, 'kind': 'probe', 'particles': [idx],
                                  'fight_id': f'cfr2:rescue:u{u}:p{pi:02d}:{primary}:probe:{idx}', 'start': p['start'], 'ops': p['ops'][primary]}))
            recs = dispatch(pool, ledger, f'probe-u{u}', jobs)
            done = [r for r in recs if r['status'] == 'completed']; wins = sum(bool(r['won']) for r in done)
            per_path = {f'p{pi:02d}': [sum(bool(r.get('won')) for r in done if r['job']['path'] == pi), sum(1 for r in done if r['job']['path'] == pi)] for pi in range(12)}
            passed = wins >= THRESHOLD
            STATE['probes'].append({'update': u, 'level': primary, 'wins': wins, 'completed': len(done), 'intended': 48,
                                    'passed': passed, 'per_path': per_path, 'distinct_roots': len({r['root_fingerprint'] for r in recs})})
            if len(done) < MIN_PROBE: raise Stop(f'probe u{u}: only {len(done)}/48 completed')
            if passed and primary in EARLIER:
                primary = EARLIER[primary]; alloc = {primary: 9, {'early': 'middle', 'full': 'early'}[primary]: 3}
            stage_done(f'probe-u{u}', t0)
        t0 = now()
        dev = pq.read_table(DEV_STARTS).to_pylist()
        if len(dev) != 100: raise Stop('dev starts')
        jobs = [(lambda f, m=models[arm], arm=arm: dict(dev_fight(m / 'model.onnx', f), arm=arm), f) for arm in ARMS for f in dev]
        recs = dispatch(pool, ledger, 'dev-eval', jobs)
        for arm in ARMS:
            mine = [r for r in recs if r['arm'] == arm]
            STATE['eval'][arm] = {'intended': 100, 'statuses': {s: sum(r['status'] == s for r in mine) for s in ('completed', 'capped', 'error')},
                                  'won': {r['job']['fight_id']: r['won'] for r in mine if r['status'] == 'completed'},
                                  'seconds': [r['seconds'] for r in mine]}
        stage_done('dev-eval', t0)
    STATE['timing']['main_wall_seconds'] = time.time() - t_start


def report():
    lines = ['# Stage 2 counterfactual-rescue pilot — REPORT', '',
             f'Generated {now()} by stage2/runner.py. Started {STATE["start"]}. Last successful stage: **{STATE["last_success"]}**.',
             f'Outcome: **{"COMPLETE" if STATE["failure"] is None else "INCOMPLETE — " + STATE["failure"]}**', '']
    ref = {}
    if DEV_REF.exists():
        for l in DEV_REF.read_text().splitlines():
            x = json.loads(l); ref[x['fight_id']] = x
    ev = STATE['eval']
    if ev:
        arms = {a: ev[a]['won'] for a in ARMS if a in ev}
        arms['original_2k'] = {k: v['won_2k'] for k, v in ref.items()}
        arms['mcts_20k'] = {k: v['won_teacher'] for k, v in ref.items()}
        lines += ['## Dev evaluation (100 existing dev starts, greedy 2k full fights, frozen pv_worker)', '',
                  '| agent | wins / completed (intended 100) | Wilson 95% | caps | errors | mean s/game |', '|---|---|---|---|---|---|']
        for a, w in arms.items():
            k, n = sum(w.values()), len(w); lo, hi = wilson(k, n)
            st = ev.get(a, {}).get('statuses', {}); secs = ev.get(a, {}).get('seconds')
            lines.append(f'| {a} | {k}/{n} | {lo:.2f}–{hi:.2f} | {st.get("capped", "rec.")} | {st.get("error", "rec.")} | '
                         f'{np.mean(secs):.1f} |' if secs else f'| {a} | {k}/{n} | {lo:.2f}–{hi:.2f} | recorded | recorded | — |')
        lines += ['', '| paired comparison (a − b) | pairs | gap | approx 95% CI | a-only | b-only | both won | both lost | exact McNemar p |',
                  '|---|---|---|---|---|---|---|---|---|']
        for a, b in [('rescue', 'control'), ('rescue', 'original_2k'), ('control', 'original_2k'), ('rescue', 'mcts_20k'), ('control', 'mcts_20k')]:
            if a in arms and b in arms:
                s = paired(arms[a], arms[b])
                lines.append(f'| {a} − {b} | {s["pairs"]} | {s["gap"]:+.2f} | {s["ci95"][0]:+.3f} to {s["ci95"][1]:+.3f} | {s["a_only"]} | '
                             f'{s["b_only"]} | {s["both_won"]} | {s["both_lost"]} | {s["mcnemar_exact_p"]:.3f} |')
        lines.append('')
    if STATE['probes']:
        lines += ['## Rescue probes (held out, never trained; threshold 35 of intended 48)', '',
                  '| after update | level | wins/completed (intended 48) | passed | distinct roots | per path wins/completed |', '|---|---|---|---|---|---|']
        for p in STATE['probes']:
            lines.append(f'| {p["update"]} | {p["level"]} | {p["wins"]}/{p["completed"]} | {p["passed"]} | {p["distinct_roots"]} | '
                         + ' '.join(f'{k}:{v[0]}/{v[1]}' for k, v in p['per_path'].items()) + ' |')
        lines += ['', 'Allocation per update: ' + '; '.join(f'u{a["update"]} {a["allocation"]}' for a in STATE['allocation']), '']
    if STATE['collect']:
        lines += ['## Training collection (greedy 2k, adapter)', '', '| arm-update | completed/intended | caps | wins | rows | by level wins/completed/intended | dispatch CPU-s |',
                  '|---|---|---|---|---|---|---|']
        for k, c in STATE['collect'].items():
            lines.append(f'| {k} | {c["completed"]}/{c["intended"]} | {c["capped"]} | {c["wins"]} | {c["rows"]} | '
                         + ' '.join(f'{lv}:{v[0]}/{v[1]}/{v[2]}' for lv, v in c['by_level'].items()) + f' | {c["dispatch_seconds"]:.0f} |')
        lines.append('')
    if STATE['train']:
        lines += ['## Training (1000 steps × 32 old + 32 new each)', '', '| arm-update | counts | new fights/states | Adam step | final 100-step loss v/p | ONNX diff | train s | exported |',
                  '|---|---|---|---|---|---|---|---|']
        for k, t in STATE['train'].items():
            fl = t['final_log']
            lines.append(f'| {k} | {t["counts"]["old"]}/{t["counts"]["new"]}/{t["counts"]["steps"]} | {t["new_fights"]}/{t["new_states"]} | '
                         f'{t["adam_step_before"]}→{t["adam_step_after"]} | {fl["value_loss"]:.3f}/{fl["policy_loss"]:.3f} | {t["onnx_max_abs_diff"]:.1e} | '
                         f'{t["train_seconds"]:.0f} | {t["exported_at"]} |')
        lines.append('')
    lines += ['## Stages / timing', '', '| stage | start | finish |', '|---|---|---|'] + [f'| {s["stage"]} | {s["start"]} | {s["finish"]} |' for s in STATE['stages']]
    lines += ['', f'Timing: {json.dumps(STATE["timing"])}', '']
    if STATE.get('post_hoc_note'): lines += ['## Deviation', '', STATE['post_hoc_note'], '']
    if STATE['failure']: lines += ['## Failure', '', '```', STATE.get('traceback', ''), '```', '']
    lines += ['Artifacts: `out/stage2/` (ledger.jsonl in completion order, state.json, {control,rescue}/update{1,2,3}/{model,episodes,new-rows.parquet,train.log}, frozen/ with sha256.json). '
              'Final reserved seeds untouched.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str))


def on_term(*_):
    for p in list(ACTIVE): kill_group(p)
    raise Stop('SIGTERM (hard wall-time bound or external stop)')


def orphans():
    out = subprocess.run(['pgrep', '-af', 'pv_continuation|pv_worker play|train_fixed.py'], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if 'pgrep' not in l]


def rebuild_report():
    """Post-hoc: regenerate REPORT.md from state.json + ledger (no compute). Dev-eval arms are assigned from the
    model path each ledger record actually played (fixes the frozen run's late-bound arm label)."""
    STATE.update(json.loads((OUT / 'state.json').read_text()))
    recs = [json.loads(l) for l in (OUT / 'ledger.jsonl').read_text().splitlines()]
    dev = [r for r in recs if r['stage'] == 'dev-eval']
    for arm in ARMS:
        mine = [r for r in dev if f'/stage2/{arm}/update3/model/model.onnx' in r['job']['model']]
        won = {r['job']['fight_id']: r['won'] for r in mine if r['status'] == 'completed'}
        if len(mine) != 100 or len(won) != len({r['job']['fight_id'] for r in mine if r['status'] == 'completed'}): raise Stop('dev ledger')
        STATE['eval'][arm] = {'intended': 100, 'statuses': {s: sum(r['status'] == s for r in mine) for s in ('completed', 'capped', 'error')},
                              'won': won, 'seconds': [r['seconds'] for r in mine]}
    STATE['timing']['ledger'] = {st: {'dispatches': sum(r['stage'] == st for r in recs), 'cpu_seconds': sum(r['seconds'] for r in recs if r['stage'] == st),
        'wall_seconds': max(r['finished'] for r in recs if r['stage'] == st) - min(r['started'] for r in recs if r['stage'] == st)}
        for st in dict.fromkeys(r['stage'] for r in recs)}
    STATE['post_hoc_note'] = ('Frozen runner completed every stage but its report step crashed (KeyError) after dev-eval because a '
        'late-bound lambda labelled all 200 dev records rescue; models actually played were correct (recorded per record). '
        'REPORT rebuilt by `runner.py report` from ledger model paths; no games re-run.')
    report()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['main', 'report']); a = ap.parse_args()
    if a.command == 'report': rebuild_report(); return
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    if (OUT / 'ledger.jsonl').exists(): raise SystemExit('refusing rerun/resume: ledger exists')
    code = 0
    try:
        main_run()
    except BaseException as e:
        STATE['failure'] = f'{type(e).__name__}: {e}'[:2000]; STATE['traceback'] = traceback.format_exc()[-4000:]; code = 2
        for p in list(ACTIVE): kill_group(p)
    finally:
        time.sleep(1); STATE['timing']['orphans_after'] = orphans(); STATE['finish'] = now(); STATE['exit'] = code
        report(); log('REPORT written', OUT / 'REPORT.md', 'exit', code)
    sys.exit(code)


if __name__ == '__main__': main()
