#!/usr/bin/env python3
"""Phase 3 counterfactual action branching (phase3/PROTOCOL.md). Run from the repo root.

  smoke : <=6 adapter games + zero-game sampler/force/query tests; writes out/smoke-passed.json
  main  : replay(85) -> freeze state manifest -> 32 queries -> screen(<=1024) -> confirm(<=512) -> REPORT.md
No training. A report is always written (also on failure); no retries; first replay mismatch stops.
"""
import argparse, concurrent.futures, hashlib, itertools, json, math, os, random, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path.cwd(); assert (REPO / 'AGENTS.md').exists(), 'run from repo root'
D6 = REPO / 'runs/schema=combat_v4/date=2026-10-06'
S1 = D6 / 'id=counterfactual-rescue-v1/out'
S2 = S1 / 'stage2'
RUN = D6 / 'id=counterfactual-branch-v1'
OUT = RUN / 'out'
BUILD_BIN = REPO / 'build/continuation-v1/pv_continuation'
U3 = S2 / 'rescue/update3/model/model.onnx'
DEV = D6 / 'id=demon-form-decoupled-policy-v1/out/main'
TAG = 'phase3-branch-v1'
N_STATES, PER_PATH, MAX_CAND, SCREEN, CONFIRM, MAX_PAIRS, GAP = 32, 3, 4, 8, 16, 16, 2
STATE_BASE, STATE_BLOCK = 2000, 64          # state s: screen 2000+64s+[0,8), confirm 2000+64s+[16,32)
WORKERS, JOB_TIMEOUT = 10, 300
HIVE = ('schema', 'date', 'id')
ACTIVE = set()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hsh(*x): return hashlib.sha256((TAG + ':' + ':'.join(map(str, x))).encode()).hexdigest()
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def log(*x): print(now(), *x, flush=True)


class Stop(Exception): pass


def screen_idx(s): return list(range(STATE_BASE + STATE_BLOCK * s, STATE_BASE + STATE_BLOCK * s + SCREEN))
def confirm_idx(s): return list(range(STATE_BASE + STATE_BLOCK * s + 16, STATE_BASE + STATE_BLOCK * s + 16 + CONFIRM))


def paths():
    rows = json.loads((S1 / 'starts.json').read_text())['rows']
    out = []
    for r in rows:
        lv = {'full': []}; lv.update({x['label']: x['prefix_actions'] for x in r['restarts']})
        out.append({'fight_id': r['fight_id'], 'start': r['start'], 'ops': lv})
    return out


# ------------------------------------------------------------------ dispatch
def kill_group(proc):
    if proc and proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def call(binary, model, requests, timeout=JOB_TIMEOUT):
    """One process, requests -> responses (one line each). Returns (responses or None, record)."""
    started = time.time(); proc = None; err = None; stdout = stderr = ''
    try:
        proc = subprocess.Popen([str(binary)] + ([str(model)] if model else []), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        ACTIVE.add(proc)
        stdout, stderr = proc.communicate(''.join(json.dumps(r) + '\n' for r in requests), timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_group(proc); err = f'{timeout}s timeout'
    finally:
        if proc: ACTIVE.discard(proc)
    resp = None
    if not err and proc.returncode == 0:
        try:
            lines = stdout.splitlines(); resp = [json.loads(l) for l in lines] if len(lines) == len(requests) else None
        except json.JSONDecodeError:
            resp = None
    return resp, {'started': started, 'finished': time.time(), 'seconds': time.time() - started, 'returncode': None if err else proc.returncode,
                  'stderr': stderr[-1500:], 'error': err}


def continuation(binary, model, job):
    """job: start, ops, chain, particle, force?, record?, shard_path?  -> ledger record (completed/capped/error)."""
    req = {'start': job['start'], 'ops': job['ops'], 'chain': job.get('chain', []), 'particles': [job['particle']], 'mode': 'learner',
           'sims': 2000, 'record': job.get('record', False)}
    for k in ('force', 'shard_path', 'fight_id'):
        if k in job: req[k] = job[k]
    resp, rec = call(binary, model, [req])
    rec['job'] = {k: v for k, v in job.items() if k not in ('start', 'ops', 'chain')}
    r = resp[0] if resp else None
    res = r.get('results', []) if r and not r.get('error') else []
    if len(res) != 1 or res[0].get('status') not in ('completed', 'capped'):
        rec.update(status='error', response_error=(r or {}).get('error')); return rec
    x = res[0]
    rec.update(status=x['status'], won=x['won'] if x['status'] == 'completed' else None, hp=x['hp'], turns=x['turns'],
               continuation_actions=x['continuation_actions'], root_fingerprint=x['root_fingerprint'], root_semantic=x['root_semantic'],
               true_fingerprint=r['true_fingerprint'], legal_actions=r['legal_actions'])
    for k in ('forced', 'after_force_fingerprint', 'actions', 'search', 'trace'):
        if k in x: rec[k] = x[k]
    return rec


class Ledger:
    def __init__(self, path):
        self.f = open(path, 'a'); self.counts = {}

    def write(self, stage, rec):
        rec = dict(rec, stage=stage, logged=now())
        self.f.write(json.dumps(rec) + '\n'); self.f.flush(); os.fsync(self.f.fileno())
        c = self.counts.setdefault(stage, {'completed': 0, 'capped': 0, 'error': 0, 'won': 0})
        c[rec['status']] += 1; c['won'] += bool(rec.get('won'))


def dispatch(pool, ledger, stage, fn, jobs, stop_on=None):
    futures = {pool.submit(fn, j): j for j in jobs}; out = []
    for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
        rec = fut.result(); ledger.write(stage, rec); out.append(rec)
        if n % 50 == 0 or n == len(jobs): log(stage, f'{n}/{len(jobs)}', ledger.counts[stage])
        if rec['status'] == 'error' or (stop_on and stop_on(rec)):
            for f in futures: f.cancel()
            raise Stop(f'{stage}: {rec.get("error") or rec.get("response_error") or rec.get("mismatch")} job={json.dumps(rec["job"])[:300]}')
    incomplete = sum(r['status'] != 'completed' for r in out)
    if incomplete > 0.05 * len(jobs): raise Stop(f'{stage}: {incomplete}/{len(jobs)} incomplete > 5%')
    return out


# ------------------------------------------------------------------ data helpers
def pool_episodes():
    eps = []
    for l in (S2 / 'ledger.jsonl').read_text().splitlines():
        x = json.loads(l); j = x['job']
        if j.get('arm') == 'rescue' and j.get('kind') == 'train' and j.get('level') == 'early' and x['status'] == 'completed' and not x['won']:
            eps.append({'fight_id': j['fight_id'], 'update': j['update'], 'path': j['path'], 'particle': j['particles'][0],
                        'shard': j['shard_path'], 'won': x['won'], 'hp': x['hp'], 'turns': x['turns'], 'actions': x['actions'],
                        'model': str(S2 / f'rescue/update{j["update"] - 1}/model/model.onnx')})
    eps.sort(key=lambda e: e['fight_id'])
    if len(eps) != 85 or len({e['fight_id'] for e in eps}) != 85: raise Stop(f'pool has {len(eps)} episodes, expected 85')
    return eps


def rows(path):
    t = pq.read_table(path); return [{k: v for k, v in r.items() if k not in HIVE} for r in t.to_pylist()]


def replay_job(ep, dest, P):
    return {'kind': 'replay', 'fight_id': ep['fight_id'], 'path': ep['path'], 'update': ep['update'], 'start': P[ep['path']]['start'],
            'ops': P[ep['path']]['ops']['early'], 'particle': ep['particle'], 'record': True, 'shard_path': str(dest), 'model': ep['model']}


def check_replay(rec, ep):
    """Exact equality with the stored episode: outcome and every stored row (original float precision)."""
    if rec['status'] != 'completed': return f'replay status {rec["status"]}'
    for k, k2 in (('won', 'won'), ('hp', 'hp'), ('turns', 'turns'), ('continuation_actions', 'actions')):
        if rec[k] != ep[k2]: return f'{k} {rec[k]} != stored {ep[k2]}'
    a, b = rows(rec['job']['shard_path']), rows(ep['shard'])
    if len(a) != len(b): return f'rows {len(a)} != {len(b)}'
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f'row {i} differs in {[k for k in x if x[k] != y.get(k)]}'
    if len(rec['trace']) != len(a) or len(rec['actions']) != len(a): return 'trace/actions length'
    return None


def known_top(trace):
    """Stage0 infer_known_prefix rule over the continuation decisions: k before each decision.
    Headbutt selection puts a known card on top (+1); a draw consumes; a reshuffle (draw grows) resets.
    Warcry/Setup selections (top-of-draw effects) are treated conservatively as known (+1)."""
    k, ks, anomalies = 0, [0], []
    for j in range(len(trace) - 1):
        a, b = trace[j], trace[j + 1]; delta = b['draw'] - a['draw']
        if a['input'] == 2 and a['task'] in (10, 19, 20):  # CARD_SELECT HEADBUTT / SETUP / WARCRY
            if delta != 1: anomalies.append([j, 'top-card select draw delta', delta])
            k += 1
        elif delta < 0: k = max(0, k + delta)
        elif delta > 0: k = 0
        ks.append(k)
    return ks, anomalies


def eligible(rec):
    tr = rec['trace']; ks, anomalies = known_top(tr); turn0 = tr[0]['turn']
    out = [i for i, t in enumerate(tr) if t['legal'] >= 2 and t['turn'] <= turn0 + 1 and ks[i] == 0]
    return out, anomalies


def select_states(replays, P):
    by_path = {}
    for ep, rec in replays:
        el, anomalies = eligible(rec)
        if anomalies: raise Stop(f'known-top audit anomaly {ep["fight_id"]} {anomalies}')
        if not el: continue
        pick = el[int(hsh('state', ep['fight_id']), 16) % len(el)]
        by_path.setdefault(ep['path'], []).append((hsh('episode', ep['fight_id']), ep, rec, pick, len(el)))
    for v in by_path.values(): v.sort(key=lambda x: x[0])
    order = sorted(by_path, key=lambda p: hsh('path', p))
    chosen = []
    for rnd in range(PER_PATH):
        for p in order:
            if len(chosen) < N_STATES and len(by_path[p]) > rnd: chosen.append(by_path[p][rnd])
    states = []
    for s, (_, ep, rec, pick, n_el) in enumerate(chosen):
        t = rec['trace'][pick]
        states.append({'state': s, 'episode': ep['fight_id'], 'path': ep['path'], 'update': ep['update'], 'restart_particle': ep['particle'],
                       'continuation_step': pick, 'absolute_step': t['step'], 'turn': t['turn'], 'legal_count': t['legal'], 'input': t['input'],
                       'task': t['task'], 'eligible_in_episode': n_el, 'chain': [{'particle': ep['particle'], 'ops': rec['actions'][:pick]}],
                       'episode_action_at_state': rec['actions'][pick], 'screen_particles': screen_idx(s), 'confirm_particles': confirm_idx(s),
                       'replay_shard': rec['job']['shard_path']})
    return states, {p: len(by_path.get(p, [])) for p in range(12)}


def candidates(st, q):
    legal = q['legal_actions']; base, teacher = q['learner']['action'], q['teacher']['action']
    cands = [{'action': base, 'source': ['baseline_learner_u3']}]
    if teacher == base: cands[0]['source'].append('teacher_20k')
    else: cands.append({'action': teacher, 'source': ['teacher_20k']})
    rest = sorted(a for a in legal if a not in {c['action'] for c in cands})
    random.Random(int(hsh('cands', st['episode']), 16)).shuffle(rest)
    for a in rest[:MAX_CAND - len(cands)]: cands.append({'action': a, 'source': ['uniform_other']})
    return cands


# ------------------------------------------------------------------ stats
def boot(diffs_by_cluster, n=10000, seed=0):
    """Cluster bootstrap of the pooled mean paired difference; clusters = dict key -> list of diffs."""
    keys = sorted(diffs_by_cluster); rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        pick = rng.integers(len(keys), size=len(keys)); d = [x for k in pick for x in diffs_by_cluster[keys[k]]]
        vals.append(np.mean(d))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


STATE = {'start': None, 'stages': [], 'last_success': None, 'failure': None}


def stage_done(name, t0, **extra):
    STATE['stages'].append({'stage': name, 'start': t0, 'finish': now(), **extra}); STATE['last_success'] = name
    (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str)); log('STAGE DONE', name)


def main_run():
    STATE['start'] = now(); t_start = time.time()
    if not (OUT / 'smoke-passed.json').exists(): raise Stop('smoke not passed')
    hashes = json.loads((OUT / 'frozen/sha256.json').read_text())
    for n, h in hashes.items():
        if sha(OUT / 'frozen' / n) != h: raise Stop(f'frozen changed {n}')
    binary = OUT / 'frozen/pv_continuation'; P = paths(); eps = pool_episodes()
    ledger = Ledger(OUT / 'ledger.jsonl'); rd = OUT / 'replays'; rd.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        # A. replay (first material mismatch stops)
        t0 = now(); byid = {e['fight_id']: e for e in eps}

        def rep(job):
            rec = continuation(binary, job['model'], job)
            if rec['status'] != 'error':
                m = check_replay(rec, byid[job['fight_id']])
                if m: rec['mismatch'] = m
            return rec
        jobs = [replay_job(e, rd / (e['fight_id'].replace(':', '_') + '.parquet'), P) for e in eps]
        recs = dispatch(pool, ledger, 'replay', rep, jobs, stop_on=lambda r: 'mismatch' in r)
        if len(recs) != 85 or any(r['status'] != 'completed' for r in recs): raise Stop('replay incomplete')
        rb = {r['job']['fight_id']: r for r in recs}
        stage_done('replay', t0, dispatches=len(recs))
        # B. freeze manifest
        t0 = now()
        states, avail = select_states([(e, rb[e['fight_id']]) for e in eps], P)
        manifest = {'tag': TAG, 'rule': 'pool=rescue early TRAIN losses u2+u3 (85); eligible = >=2 legal, turn<=restart turn+1, known-top k=0; '
                    'one state/episode uniform by sha256(tag:state:fid); episodes per path by sha256(tag:episode:fid); paths by sha256(tag:path:p); '
                    'round-robin <=3/path to 32', 'available_by_path': avail, 'states': states}
        (OUT / 'states.json').write_text(json.dumps(manifest, indent=1))
        STATE['manifest_sha256'] = sha(OUT / 'states.json'); STATE['n_states'] = len(states)
        stage_done('manifest', t0, n_states=len(states))
        # C. queries (u3 2k + teacher 20k at the recorded state); features must equal the replayed row exactly
        t0 = now()

        def qry(st):
            p = P[st['path']]
            resp, rec = call(binary, U3, [{'start': p['start'], 'ops': p['ops']['early'], 'chain': st['chain'], 'mode': 'query', 'sims': 2000,
                                            'teacher_sims': 20000}], timeout=600)
            rec['job'] = {'kind': 'query', 'state': st['state']}
            q = resp[0].get('query') if resp and not resp[0].get('error') else None
            if not q: rec.update(status='error', response_error=resp and resp[0].get('error')); return rec
            row = rows(st['replay_shard'])[st['continuation_step']]
            feats = dict(zip(('context', 'cards', 'monsters', 'potions', 'relics', 'actions'), q['features']))
            same = row['moves'] == q['legal_actions'] and all(list(np.float32(feats[k])) == list(np.float32(row[k])) for k in feats)
            rec.update(status='completed' if same else 'error', query=q, true_semantic=resp[0]['true_semantic'],
                       error=None if same else 'query state features/legal differ from replayed row')
            return rec
        qrecs = dispatch(pool, ledger, 'query', qry, states)
        qb = {r['job']['state']: r for r in qrecs}
        for st in states:
            st['query'] = {k: qb[st['state']]['query'][k] for k in ('learner', 'teacher', 'legal_actions')}
            st['candidates'] = candidates(st, qb[st['state']]['query'])
        (OUT / 'candidates.json').write_text(json.dumps(states, indent=1))
        STATE['query_cpu_seconds'] = sum(r['seconds'] for r in qrecs)
        stage_done('query', t0)
        # D. screen
        t0 = now(); jobs = []
        for st in states:
            p = P[st['path']]
            for c in st['candidates']:
                for j in st['screen_particles']:
                    jobs.append({'kind': 'screen', 'state': st['state'], 'path': st['path'], 'action': c['action'], 'source': c['source'],
                                 'start': p['start'], 'ops': p['ops']['early'], 'chain': st['chain'], 'particle': j, 'force': c['action']})
        if len(jobs) > 1024: raise Stop('screen budget')
        srecs = dispatch(pool, ledger, 'screen', lambda j: continuation(binary, U3, j), jobs)
        # same root across actions for every (state, particle)
        roots = {}
        for r in srecs: roots.setdefault((r['job']['state'], r['job']['particle']), set()).add(r['root_fingerprint'])
        if any(len(v) != 1 for v in roots.values()): raise Stop('root differs across actions for a (state, particle)')
        stage_done('screen', t0, dispatches=len(srecs))
        # E. confirmation selection (predeclared) and confirmation
        t0 = now(); res = {}
        for r in srecs: res[(r['job']['state'], r['job']['action'], r['job']['particle'])] = r
        pairs = []
        for st in states:
            base = st['candidates'][0]['action']
            b = [res[(st['state'], base, j)] for j in st['screen_particles']]
            best = None
            for c in st['candidates'][1:]:
                a = [res[(st['state'], c['action'], j)] for j in st['screen_particles']]
                complete = all(x['status'] == 'completed' for x in a + b)
                g = sum(bool(x['won']) for x in a) - sum(bool(x['won']) for x in b) if complete else None
                c['screen'] = {'alt_wins': sum(bool(x.get('won')) for x in a), 'base_wins': sum(bool(x.get('won')) for x in b), 'gap': g, 'complete': complete}
                if g is not None and g >= GAP:
                    key = (-g, 0 if 'teacher_20k' in c['source'] else 1, hsh('pair', st['state'], c['action']))
                    if best is None or key < best[0]: best = (key, st, c)
            st['screen_base_wins'] = sum(bool(x.get('won')) for x in b)
            if best: pairs.append(best)
        pairs.sort(key=lambda x: x[0]); pairs = pairs[:MAX_PAIRS]
        STATE['screen'] = [{k: st[k] for k in ('state', 'path', 'episode', 'turn', 'legal_count', 'input', 'task', 'candidates', 'screen_base_wins', 'query')} for st in states]
        STATE['confirm_pairs'] = [{'state': st['state'], 'path': st['path'], 'alt': c['action'], 'source': c['source'], 'screen_gap': c['screen']['gap']} for _, st, c in pairs]
        (OUT / 'candidates.json').write_text(json.dumps(states, indent=1))
        jobs = []
        for _, st, c in pairs:
            p = P[st['path']]
            for act, role in ((c['action'], 'alt'), (st['candidates'][0]['action'], 'base')):
                for j in st['confirm_particles']:
                    jobs.append({'kind': 'confirm', 'state': st['state'], 'path': st['path'], 'action': act, 'role': role, 'start': p['start'],
                                 'ops': p['ops']['early'], 'chain': st['chain'], 'particle': j, 'force': act})
        crecs = dispatch(pool, ledger, 'confirm', lambda j: continuation(binary, U3, j), jobs) if jobs else []
        STATE['confirm_records'] = [{k: r.get(k) for k in ('job', 'status', 'won', 'root_fingerprint')} for r in crecs]
        stage_done('confirm', t0, dispatches=len(crecs), pairs=len(pairs))
    STATE['main_wall_seconds'] = time.time() - t_start


def analysis():
    lines = ['# Phase 3 counterfactual branching — REPORT', '', f'Generated {now()}. Started {STATE.get("start")}. Last successful stage: **{STATE.get("last_success")}**.',
             f'Outcome: **{"COMPLETE" if not STATE.get("failure") else "INCOMPLETE — " + STATE["failure"]}**', '']
    if STATE.get('confirm_pairs') is not None and 'confirm_records' in STATE:
        recs = STATE['confirm_records']; out = []; by_path, by_state = {}, {}
        tw = tl = tt = 0
        for p in STATE['confirm_pairs']:
            alt = {r['job']['particle']: r for r in recs if r['job']['state'] == p['state'] and r['job']['role'] == 'alt'}
            base = {r['job']['particle']: r for r in recs if r['job']['state'] == p['state'] and r['job']['role'] == 'base'}
            ks = [k for k in alt if k in base and alt[k]['status'] == 'completed' and base[k]['status'] == 'completed']
            d = [int(alt[k]['won']) - int(base[k]['won']) for k in ks]
            w, l = sum(x > 0 for x in d), sum(x < 0 for x in d); t = len(d) - w - l; tw += w; tl += l; tt += t
            by_path.setdefault(p['path'], []).extend(d); by_state.setdefault(p['state'], []).extend(d)
            out.append(f'| {p["state"]} | p{p["path"]:02d} | {p["alt"]} | {"+".join(p["source"])} | {p["screen_gap"]:+d}/8 | '
                       f'{sum(alt[k]["won"] for k in ks)}/{len(ks)} | {sum(base[k]["won"] for k in ks)}/{len(ks)} | {w}/{l}/{t} | {np.mean(d) if d else float("nan"):+.3f} |')
        alld = [x for v in by_state.values() for x in v]
        lines += ['## Confirmation (fresh disjoint particles; alt vs baseline forced at the same sampled root, then frozen rescue-u3 2k)', '',
                  f'Selected pairs: {len(STATE["confirm_pairs"])} (rule: screen gap >= +2/8, <=1 per state, top 16). Intended {len(STATE["confirm_pairs"]) * 16} paired particles; complete pairs {len(alld)}.', '',
                  '| state | path | alt action | source | screen gap | alt wins | base wins | alt-better/base-better/tie | mean diff |', '|---|---|---|---|---|---|---|---|---|'] + out
        if alld:
            pb, sb = boot(by_path), boot(by_state)
            STATE['evidence'] = {'pooled_diff': float(np.mean(alld)), 'path_cluster_ci95': pb, 'state_cluster_ci95': sb, 'paths': len(by_path),
                                 'states': len(by_state), 'pairs_complete': len(alld), 'alt_better': tw, 'base_better': tl, 'ties': tt,
                                 'improving_alternatives_found': bool(np.mean(alld) > 0 and pb[0] > 0)}
            e = STATE['evidence']
            lines += ['', f'**Pooled confirmation alt − baseline win rate: {e["pooled_diff"]:+.3f}** over {e["pairs_complete"]} complete paired particles '
                      f'({e["states"]} states, {e["paths"]} paths); path-cluster bootstrap 95% {pb[0]:+.3f} to {pb[1]:+.3f}; state-cluster {sb[0]:+.3f} to {sb[1]:+.3f}. '
                      f'Particle-level alt-better/base-better/tie {tw}/{tl}/{tt}. Predeclared evidence rule (mean>0 and path-cluster lower bound>0): '
                      f'**{"MET" if e["improving_alternatives_found"] else "NOT MET"}**. Bootstrap over <=12 path clusters is approximate.', '']
            sel_gap = np.mean([p['screen_gap'] / 8 for p in STATE['confirm_pairs']])
            lines += [f'Winner\'s curse: mean selected screen gap {sel_gap:+.3f} vs confirmation {e["pooled_diff"]:+.3f} (shrinkage {sel_gap - e["pooled_diff"]:+.3f}).', '']
    if STATE.get('screen'):
        lines += ['## Screen (8 shared root particles per state; all outcomes retained)', '', '| state | path | turn | legal | input/task | baseline wins | candidates action:source:wins (gap) |', '|---|---|---|---|---|---|---|']
        tdiff = []
        for st in STATE['screen']:
            cs = ' '.join(f'{c["action"]}:{"+".join(s[:3] for s in c["source"])}:{c.get("screen", {}).get("alt_wins", st["screen_base_wins"])}'
                          + (f'({c["screen"]["gap"]:+d})' if c.get('screen') and c['screen']['gap'] is not None else '') for c in st['candidates'])
            lines.append(f'| {st["state"]} | p{st["path"]:02d} | {st["turn"]} | {st["legal_count"]} | {st["input"]}/{st["task"]} | {st["screen_base_wins"]}/8 | {cs} |')
            for c in st['candidates'][1:]:
                if 'teacher_20k' in c['source'] and c['screen']['gap'] is not None: tdiff.append(c['screen']['gap'] / 8)
        allc = [c for st in STATE['screen'] for c in st['candidates'][1:] if c.get('screen') and c['screen']['gap'] is not None]
        lines += ['', f'Unselected screen estimates: teacher≠baseline at {len(tdiff)} states, mean teacher − baseline {np.mean(tdiff) if tdiff else float("nan"):+.3f}; '
                  f'all {len(allc)} alternatives mean {np.mean([c["screen"]["gap"] / 8 for c in allc]) if allc else float("nan"):+.3f}; '
                  f'baseline screen wins {sum(st["screen_base_wins"] for st in STATE["screen"])}/{8 * len(STATE["screen"])}. '
                  f'States where teacher == baseline: {sum("teacher_20k" in st["candidates"][0]["source"] for st in STATE["screen"])}.', '']
    lines += ['## Stages', '', '| stage | start | finish | detail |', '|---|---|---|---|']
    lines += [f'| {s["stage"]} | {s["start"]} | {s["finish"]} | {json.dumps({k: v for k, v in s.items() if k not in ("stage", "start", "finish")})} |' for s in STATE['stages']]
    if (OUT / 'ledger.jsonl').exists():
        recs = [json.loads(l) for l in (OUT / 'ledger.jsonl').read_text().splitlines()]
        tim = {}
        for r in recs:
            t = tim.setdefault(r['stage'], {'dispatches': 0, 'completed': 0, 'capped': 0, 'error': 0, 'cpu_s': 0.0, 'first': r['started'], 'last': r['finished']})
            t['dispatches'] += 1; t[r['status']] += 1; t['cpu_s'] += r['seconds']; t['first'] = min(t['first'], r['started']); t['last'] = max(t['last'], r['finished'])
        lines += ['', '| ledger stage | dispatches | completed | capped | error | CPU s | wall s |', '|---|---|---|---|---|---|---|']
        lines += [f'| {k} | {v["dispatches"]} | {v["completed"]} | {v["capped"]} | {v["error"]} | {v["cpu_s"]:.0f} | {v["last"] - v["first"]:.0f} |' for k, v in tim.items()]
    lines += ['', f'Main wall seconds: {STATE.get("main_wall_seconds")}; exit {STATE.get("exit")}; orphans after: {STATE.get("orphans_after")}; manifest sha256 {STATE.get("manifest_sha256")}.']
    if STATE.get('failure'): lines += ['', '## Failure', '', '```', STATE.get('traceback', ''), '```']
    lines += ['', 'Artifacts: out/{ledger.jsonl,states.json,candidates.json,state.json,replays/,frozen/}. No training. Query features are features only (no labels).', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)); (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str))


def on_term(*_):
    for p in list(ACTIVE): kill_group(p)
    raise Stop('SIGTERM (hard wall bound or external stop)')


def orphans():
    o = subprocess.run(['pgrep', '-af', 'pv_continuation'], capture_output=True, text=True).stdout
    return [l for l in o.splitlines() if 'pgrep' not in l]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['main', 'report']); a = ap.parse_args()
    if a.command == 'report':
        STATE.update(json.loads((OUT / 'state.json').read_text())); analysis(); return
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    if (OUT / 'ledger.jsonl').exists(): raise SystemExit('refusing rerun: ledger exists')
    code = 0
    try:
        main_run()
    except BaseException as e:
        STATE['failure'] = f'{type(e).__name__}: {e}'[:2000]; STATE['traceback'] = traceback.format_exc()[-4000:]; code = 2
        for p in list(ACTIVE): kill_group(p)
    finally:
        time.sleep(1); STATE['orphans_after'] = orphans(); STATE['finish'] = now(); STATE['exit'] = code
        try: analysis()
        except Exception:
            (OUT / 'REPORT.md').write_text(f'# Phase 3 REPORT (analysis failed)\n\nfailure: {STATE.get("failure")}\n\n```\n{traceback.format_exc()}\n```\n')
            (OUT / 'state.json').write_text(json.dumps(STATE, indent=1, default=str)); code = code or 3
        log('REPORT written', OUT / 'REPORT.md', 'exit', code)
    sys.exit(code)


if __name__ == '__main__': main()
