"""Teacher continuations from learner decision states (Demon Form) — machinery, gates, frozen sample, labels.

Uses a SNAPSHOT of the existing champ_session sandbox binary (apps/champ_viewer/session.cpp), never the shared
build. A state is rebuilt as start + learner action prefix ({"act": bits} ops, legality checked natively).
`playout SIMS FROM N` plays N complete fair (non-oracle) MCTS-teacher fights (guided rollout, 8 belief particles
per decision, fresh search per decision, forced moves unsearched) from public-belief particles FROM..FROM+N-1 of the
state: hidden draw order + all RNG streams resampled (draw pile reshuffled uniformly; known Headbutt top cards are
NOT preserved), search salt = particle index + 1. Label = the fight's actual PLAYER_VICTORY flag.

Stages (each separately approved; this file only implements them):
  snapshot : copy binary + relevant sources, hashes, git heads/dirty status
  gates    : A replay/encoding parity, B teacher-search parity, C terminal replay, D playout determinism (no labels)
  sample   : freeze eligible pool + seeded sample (no labels)
  smoke    : <= 12 continuations
  label    : 4 continuations per sampled state
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import correction_absorption as C  # noqa: E402
import decoupled_policy as DP  # noqa: E402

EXP = 'demon-form-continuation-v1'
OUT = C.DATE / f'id={EXP}/out'
BINARY = Path('build/champ-viewer/champ_session')
LS = Path('../sts_lightspeed')
SOURCES = ['apps/champ_viewer/session.cpp', 'agents/combat/search/teacher_search.cpp', 'agents/combat/search/teacher_search.hpp',
           'agents/combat/search/teacher_leaves.cpp', 'agents/combat/search/teacher_leaves.hpp',
           'environments/combat/environment.cpp', 'agents/combat/pv/evaluator.cpp', 'agents/combat/pv/search.cpp',
           'apps/pv/worker.cpp']
LS_SOURCES = ['src/sim/search/PublicBeliefCombatSearch.cpp', 'include/sim/search/PublicBeliefCombatSearch.h']
ROUNDS = range(6, 16)
SIMS = 20000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True).stdout.strip()


def cmd_snapshot(out):
    d = out / 'frozen'
    if d.exists(): raise FileExistsError('snapshot exists')
    (d / 'src').mkdir(parents=True)
    shutil.copy2(BINARY, d / 'champ_session')
    for s in SOURCES: (d / 'src' / s).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(s, d / 'src' / s)
    for s in LS_SOURCES: (d / 'src/sts_lightspeed' / s).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(LS / s, d / 'src/sts_lightspeed' / s)
    meta = dict(binary=str(BINARY), binary_sha=sha(d / 'champ_session'), binary_mtime=Path(BINARY).stat().st_mtime,
                sources={s: sha(s) for s in SOURCES}, sts_lightspeed={s: sha(LS / s) for s in LS_SOURCES},
                head=git('.', 'rev-parse', 'HEAD'), dirty=git('.', 'status', '--short', '--', *SOURCES),
                sts_lightspeed_head=git(LS, 'rev-parse', 'HEAD'), sts_lightspeed_dirty=git(LS, 'status', '--short'),
                note='Binary built 2026-10-05 18:06 from the then-current tree; sources copied now (may differ from what '
                     'was compiled). Behavioural parity is established by gates, not by source hashes.')
    (d / 'snapshot.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))


def session(requests, out, model=None, timeout=600):
    """One fresh snapshot-binary process per call; returns parsed responses (errors recorded, never retried)."""
    cmd = [str(out / 'frozen/champ_session')] + ([str(model)] if model else [])
    proc = subprocess.run(cmd, input=''.join(json.dumps(r) + '\n' for r in requests), text=True,
                          capture_output=True, timeout=timeout)
    if proc.returncode: raise RuntimeError(f'champ_session exit {proc.returncode}: {proc.stderr[-500:]}')
    return [json.loads(l) for l in proc.stdout.splitlines()]


# ---------------- learner data ----------------

def learner_fights():
    """fight_id -> dict(start, actions, won, round) for active replay rounds 6–15 (TRAINING starts only)."""
    starts, out = {}, {}
    for i in ROUNDS:
        d = C.SRC / f'iter{i:03d}'
        starts.update({r['fight_id']: r for r in pq.read_table(d / 'starts.parquet').to_pylist()})
        for r in pq.read_table(d / 'collect/fights-0.parquet', columns=['fight_id', 'actions', 'won']).to_pylist():
            out[r['fight_id']] = dict(actions=r['actions'], won=r['won'], round=i)
    ids = set(json.loads((C.SRC / 'iter015/train-fights.json').read_text()))
    split = json.loads((C.SRC / 'split.json').read_text())
    for fid in out: out[fid]['start'] = starts[fid]['start']
    return {fid: v for fid, v in out.items() if fid in ids and split.get(fid) == 'train'}


def forbidden_seeds():
    s = set()
    for f in ['monitor.parquet', 'final-reserved.parquet']:
        s.update(r['start']['seed'] for r in pq.read_table(C.SRC / f, columns=['start']).to_pylist())
    s.update(r['start']['seed'] for r in pq.read_table(DP.C.DATE / 'id=demon-form-decoupled-policy-v1/out/main/starts.parquet', columns=['start']).to_pylist())
    return s


# ---------------- gates ----------------

def gate_rows(n=30, seed=1):
    """Deterministic mixed set of training decision rows: plays, Dual Wield/Headbutt selections, forced steps."""
    rng = random.Random(seed); picks = []
    for i in [6, 10, 15]:
        rows = C.Rows(C.SRC / f'iter{i:03d}/rows/rows.parquet')
        groups = collections.defaultdict(list)
        for k, r in enumerate(rows.meta):
            groups['forced' if not r['has_policy'] else r['stratum']].append(k)
        for g in ['dual_wield_select', 'headbutt_select', 'setup_legal', 'other_play', 'forced']:
            if groups[g]: picks += [(rows, k) for k in rng.sample(groups[g], min(2, len(groups[g])))]
    return picks[:n]


def cmd_gates(out):
    if sha(out / 'frozen/champ_session') != json.loads((out / 'frozen/snapshot.json').read_text())['binary_sha']:
        raise ValueError('snapshot binary changed')
    fights = learner_fights(); res = {}
    # A: rebuilt state's legal menu (order) and update15 value/priors == frozen worker on the training row
    picks = gate_rows(); reqs = []; native_lines = []
    for rows, k in picks:
        r = rows.meta[k]; f = fights[r['fight_id']]
        reqs.append(dict(start=f['start'], ops=[{'act': a} for a in f['actions'][:r['step']]], query='pv'))
        native_lines.append([DP.state_json(rows.shard, rows.shard.rows[k])])
    resp = session(reqs, out, DP.A_DIR / 'model.onnx')
    nat = DP.native(DP.A_DIR / 'model.onnx', native_lines)
    a = []
    for (rows, k), s, p in zip(picks, resp, nat):
        r = rows.meta[k]
        legal = [m['bits'] for m in s['view']['legal']]
        if 'pv' not in s:  # pv needs >0 moves; record
            a.append(dict(fight_id=r['fight_id'], step=r['step'], error=s.get('query_error') or s.get('error'))); continue
        z = np.asarray(p['logits'], np.float64); q = np.exp(z - z.max()); q /= q.sum()
        pri = [s['pv']['priors'][str(b)] for b in r['moves']]
        a.append(dict(fight_id=r['fight_id'], step=r['step'], stratum=r['stratum'], forced=not r['has_policy'],
                      legal_order_equal=legal == list(r['moves']),
                      value_bitwise=s['pv']['value'] == float(np.float32(p['value'])) / 100.0,
                      max_prior_diff=float(np.abs(np.asarray(pri) - q).max())))
    res['A_replay_encoding'] = dict(states=len(a), legal_order_equal=sum(x.get('legal_order_equal', False) for x in a),
                                    value_bitwise=sum(x.get('value_bitwise', False) for x in a),
                                    max_prior_diff=max(x.get('max_prior_diff', 1) for x in a), detail=a)
    # B: teacher search parity at teacher-play states (training starts): salt-0 search == recorded frozen teacher
    tp = C.TC / 'teacher-play'
    rec = {r['fight_id']: r for r in map(json.loads, (tp / 'results.jsonl').read_text().splitlines())}
    reqs, keys = [], []
    for fid in sorted(rec)[:4]:
        r = rec[fid]
        for s in r['search'][:3]:
            reqs.append(dict(start=r['fight']['start'], ops=[{'act': x} for x in r['fight']['actions'][:s['step']]],
                             query='search', sims=SIMS, salt=0)); keys.append((fid, s))
    resp = session(reqs, out)
    b = []
    for (fid, s), o in zip(keys, resp):
        mine = {int(k): v['visits'] for k, v in o['search']['moves'].items()}
        theirs = {c['action']: c['visits'] for c in s['children']}
        b.append(dict(fight_id=fid, step=s['step'], visits_equal=mine == theirs, root_value_equal=o['search']['root_value'] == s['root_value'],
                      simulations_equal=o['search']['simulations'] == s['simulations']))
    res['B_teacher_search'] = dict(states=len(b), all_equal=sum(x['visits_equal'] and x['root_value_equal'] and x['simulations_equal'] for x in b), detail=b)
    # C: terminal replay — full learner action list reproduces the recorded outcome; playout on it is refused
    ids = sorted(fights)[:6]
    resp = session([dict(start=fights[i]['start'], ops=[{'act': x} for x in fights[i]['actions']], query='playout', sims=SIMS, n=1) for i in ids], out)
    res['C_terminal'] = [dict(fight_id=i, recorded_won=fights[i]['won'], kind=o['view'].get('kind'), won=o['view'].get('won'),
                              legal_after_end=len(o['view'].get('legal', [])), query_error=o.get('query_error'),
                              match=o['view'].get('won') == fights[i]['won'] and o.get('query_error') == 'the fight is over')
                         for i, o in zip(ids, resp)]
    # D (playout determinism/independence) is deferred to the approved smoke stage: it consumes continuations.
    (out / 'gates.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'detail'} if isinstance(v, dict) else v) for k, v in res.items()}, indent=1, default=str))

SAMPLE_SEED = 20261007
N_STATES = 300
K = 4
SAMPLER = 'current-public-particles: uniform draw reshuffle, known Headbutt top card NOT preserved'


def cmd_sample(out):
    """Freeze the eligible pool and a seeded sample: 300 of the 1,000 active-replay training fights uniformly, then one
    searched decision state (>1 legal move) uniformly within each. Inclusion prob = 0.3 / n_decisions(fight)."""
    dest = out / 'sample.jsonl'
    if dest.exists(): raise FileExistsError('sample already frozen')
    import torch
    from agents.combat.pv.model import NAMES
    fights = learner_fights(); bad = forbidden_seeds()
    if any(f['start']['seed'] in bad for f in fights.values()): raise ValueError('training/eval seed overlap')
    states = collections.defaultdict(list); rowsets = {}
    for i in ROUNDS:
        rows = C.Rows(C.SRC / f'iter{i:03d}/rows/rows.parquet'); rowsets[i] = rows
        for k, r in enumerate(rows.meta):
            if r['fight_id'] in fights and r['has_policy']: states[r['fight_id']].append((i, k))
    pool = sorted(fid for fid in fights if states[fid])
    rng = random.Random(SAMPLE_SEED)
    chosen = sorted(rng.sample(pool, N_STATES))
    net = C.load_net(C.MODELS['update15']); out_rows = []
    for fid in chosen:
        i, k = rng.choice(sorted(states[fid], key=lambda x: rowsets[x[0]].meta[x[1]]['step']))
        rows = rowsets[i]; r = rows.meta[k]; f = fights[fid]
        prefix = f['actions'][:r['step']]
        with torch.no_grad():
            b = rows.shard.batch(rows.shard.rows[[k]]); v, _ = net(*(b[0][n] for n in NAMES))
        # known-top-card caveat: a Headbutt selection earlier in the same turn (belief reshuffles that card)
        same_turn = [m for m in rows.meta if m['fight_id'] == fid and m['turn'] == r['turn'] and m['step'] < r['step']]
        hb = any(m['stratum'] == 'headbutt_select' for m in same_turn)
        out_rows.append(dict(fight_id=fid, round=i, step=r['step'], turn=r['turn'], stratum=r['stratum'], learner_won=f['won'],
                             n_decisions=len(states[fid]), inclusion_prob=N_STATES / len(pool) / len(states[fid]),
                             update15_value=float(v[0]), after_headbutt_same_turn=hb, start=f['start'], prefix=prefix,
                             moves=r['moves']))
    dest.write_text(''.join(json.dumps(x) + '\n' for x in out_rows))
    meta = dict(seed=SAMPLE_SEED, pool_fights=len(pool), pool_states=sum(len(v) for v in states.values()), sampled=len(out_rows),
                sample_sha=sha(dest), learner_wins=sum(x['learner_won'] for x in out_rows),
                strata=dict(collections.Counter(x['stratum'] for x in out_rows)),
                after_headbutt_same_turn=sum(x['after_headbutt_same_turn'] for x in out_rows))
    (out / 'sample.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))


def playout_job(args):
    out, x, first, n, timeout = args
    req = dict(start=x['start'], ops=[{'act': a} for a in x['prefix']], query='playout', sims=SIMS, **{'from': first}, n=n)
    import time
    t = time.time()
    try:
        r = session([req], out, timeout=timeout)[0]
        status = 'error' if 'error' in r or 'query_error' in r else 'completed'
        legal = [m['bits'] for m in r.get('view', {}).get('legal', [])]
        return dict(fight_id=x['fight_id'], step=x['step'], first=first, n=n, status=status, seconds=time.time() - t, sampler=SAMPLER,
                    menu_matches_row=legal == x['moves'], playouts=r.get('playouts'), error=r.get('error') or r.get('query_error'))
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        return dict(fight_id=x['fight_id'], step=x['step'], first=first, n=n, status='error', seconds=time.time() - t, error=repr(e))


def run_jobs(out, jobs, name, workers=10):
    from concurrent.futures import ThreadPoolExecutor
    dest = out / f'{name}.jsonl'
    if dest.exists(): raise FileExistsError(f'{dest} exists; no resume/retry')
    with ThreadPoolExecutor(workers) as pool, dest.open('w') as f:
        for r in pool.map(playout_job, jobs):
            f.write(json.dumps(r) + '\n'); f.flush()
            print(r['fight_id'], r['step'], r['status'], round(r['seconds'], 1), flush=True)


def load_sample(out):
    meta = json.loads((out / 'sample.json').read_text())
    if sha(out / 'sample.jsonl') != meta['sample_sha']: raise ValueError('sample changed')
    return [json.loads(l) for l in (out / 'sample.jsonl').read_text().splitlines()]


def cmd_smoke(out):
    """12 continuations: states 0 and 1 x particles 0-3, then state 0 particles 0-3 again (determinism)."""
    xs = load_sample(out)
    run_jobs(out, [(out, xs[0], 0, 4, 900), (out, xs[1], 0, 4, 900), (out, xs[0], 0, 4, 900)], 'smoke', workers=3)


def cmd_eligible(out):
    """Freeze the eligible manifest (k = 0 known draw-order cards) with exclusion reasons, BEFORE any bulk label.
    Fails closed unless known_prefix.jsonl exists and covers exactly the frozen sample."""
    dest = out / 'eligible.json'
    if dest.exists(): raise FileExistsError('eligible manifest already frozen')
    xs = load_sample(out); kp = out / 'known_prefix.jsonl'
    if not kp.exists(): raise FileNotFoundError('known_prefix.jsonl missing: run the audit first')
    audit = {(r['fight_id'], r['step']): r for r in map(json.loads, kp.read_text().splitlines())}
    if set(audit) != {(x['fight_id'], x['step']) for x in xs} or len(audit) != len(xs):
        raise ValueError('known-prefix audit does not match the frozen sample')
    eligible, excluded = [], []
    for x in xs:
        a = audit[(x['fight_id'], x['step'])]
        if a['anomalies']: excluded.append(dict(fight_id=x['fight_id'], step=x['step'], reason='audit anomaly'))
        elif a['known_prefix'] > 0: excluded.append(dict(fight_id=x['fight_id'], step=x['step'], reason=f'known draw-order prefix k={a["known_prefix"]}'))
        else: eligible.append([x['fight_id'], x['step']])
    man = dict(sample_sha=sha(out / 'sample.jsonl'), known_prefix_sha=sha(kp), eligible=eligible, excluded=excluded,
               n_eligible=len(eligible), n_excluded=len(excluded),
               estimand='sampled states with no identified known draw-order information (k=0), current public-particle sampler')
    dest.write_text(json.dumps(man, indent=1)); print(json.dumps({k: v for k, v in man.items() if k != 'eligible'}, indent=1))


def eligible_states(out):
    man = json.loads((out / 'eligible.json').read_text())
    if sha(out / 'sample.jsonl') != man['sample_sha'] or sha(out / 'known_prefix.jsonl') != man['known_prefix_sha']:
        raise ValueError('sample or audit changed since eligibility freeze')
    keep = {tuple(e) for e in man['eligible']}
    return [x for x in load_sample(out) if (x['fight_id'], x['step']) in keep]


def cmd_label(out):
    """4 unique particle indices (0-3) per eligible state; smoke results reused for identical-protocol states."""
    from concurrent.futures import ThreadPoolExecutor
    xs = eligible_states(out); dest = out / 'labels.jsonl'
    if dest.exists(): raise FileExistsError(f'{dest} exists; no resume/retry')
    smoke = {}
    for r in map(json.loads, (out / 'smoke.jsonl').read_text().splitlines()):
        key = (r['fight_id'], r['step'])
        if key not in smoke and r['first'] == 0 and r['n'] == K and r['status'] == 'completed': smoke[key] = r
    reused = [dict(smoke[(x['fight_id'], x['step'])], reused_from='smoke') for x in xs if (x['fight_id'], x['step']) in smoke]
    todo = [x for x in xs if (x['fight_id'], x['step']) not in smoke]
    total, done, statuses = len(xs), 0, collections.Counter()
    with dest.open('w') as f:
        for r in reused:
            f.write(json.dumps(r) + '\n'); done += 1; statuses[r['status']] += 1
        print(f'reused {len(reused)} smoke states; dispatching {len(todo)} states x {K} = {len(todo) * K} continuations', flush=True)
        with ThreadPoolExecutor(10) as pool:
            for r in pool.map(playout_job, [(out, x, 0, K, 300) for x in todo]):
                f.write(json.dumps(r) + '\n'); f.flush(); done += 1; statuses[r['status']] += 1
                wins = sum(p['won'] for p in r['playouts']) if r.get('playouts') else None
                print(f'{done}/{total} {r["fight_id"]} step {r["step"]} {r["status"]} wins={wins} {r["seconds"]:.1f}s '
                      f'statuses={dict(statuses)}', flush=True)
    print('done', dict(statuses), flush=True)


def debiased_sq_error(v, wins, k):
    """Unbiased per-state estimate of (v - T)^2 from k Bernoulli(T) continuation outcomes (k >= 2):
    (v - Ybar)^2 - Ybar(1 - Ybar)/(k - 1), since E[(v-Ybar)^2] = (v-T)^2 + T(1-T)/k and
    E[Ybar(1-Ybar)] = T(1-T)(k-1)/k. Individual values can be negative; only averages are meaningful."""
    y = wins / k
    return (v - y) ** 2 - y * (1 - y) / (k - 1)


def summarize_labels(xs, labels, weights=None, k=K):
    """Aggregate estimands over sampled states (one per fight => independent units). v in [0,1]."""
    v = np.array([x['update15_value'] / 100 for x in xs]); lw = np.array([float(x['learner_won']) for x in xs])
    wins = np.array([sum(p['won'] for p in lab['playouts']) for lab in labels], float); y = wins / k
    w = np.ones(len(xs)) if weights is None else np.asarray(weights, float); w = w / w.sum()
    def mean_se(a):
        m = float((w * a).sum()); n = len(a)
        se = float(np.sqrt((w ** 2 * (a - m) ** 2).sum() * n / (n - 1))) if n > 1 else float('nan')
        return dict(mean=m, se=se, ci95=[m - 1.959964 * se, m + 1.959964 * se])
    return dict(n=len(xs), teacher_continuation=mean_se(y), learner_outcome=mean_se(lw), update15_value=mean_se(v),
                teacher_minus_learner=mean_se(y - lw), teacher_minus_value=mean_se(y - v),
                debiased_mse_value_vs_teacher=mean_se(debiased_sq_error(v, wins, k)),
                brier_value_vs_learner_outcome=mean_se((v - lw) ** 2))


def infer_known_prefix(views):
    """Known-top-of-draw count after each step from consecutive public views (deck-specific: only Headbutt puts a
    card on top of the draw pile; draws consume from the top; a reshuffle only happens once the pile is empty).
    views[j] = state before action j. Returns (k per state, anomalies)."""
    k, ks, anomalies = 0, [0], []
    for j in range(len(views) - 1):
        a, b = views[j], views[j + 1]
        delta = len(b['draw']) - len(a['draw'])
        if a['kind'] == 'headbutt':
            if delta != 1: anomalies.append((j, 'headbutt delta', delta))
            k += 1
        elif delta < 0:
            k = max(0, k + delta)
        elif delta > 0:  # reshuffle: only after the pile (including any known prefix) was drawn empty
            k = 0
        ks.append(k)
    return ks, anomalies


def cmd_known_prefix(out):
    """Audit (no continuations): known-top-card count at each sampled state, inferred from public views."""
    xs = load_sample(out); res = []
    for x in xs:
        reqs = [dict(start=x['start'], ops=[{'act': a} for a in x['prefix'][:j]], query='view') for j in range(len(x['prefix']) + 1)]
        views = [r['view'] for r in session(reqs, out)]
        ks, anomalies = infer_known_prefix(views)
        res.append(dict(fight_id=x['fight_id'], step=x['step'], turn=x['turn'], known_prefix=ks[-1],
                        any_known_before=any(ks), anomalies=anomalies, headbutt_same_turn=x['after_headbutt_same_turn']))
    (out / 'known_prefix.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in res))
    summary = dict(states=len(res), known_prefix_positive=sum(r['known_prefix'] > 0 for r in res),
                   distribution=dict(collections.Counter(r['known_prefix'] for r in res)),
                   positive_not_same_turn=sum(r['known_prefix'] > 0 and not r['headbutt_same_turn'] for r in res),
                   same_turn_flag_but_zero=sum(r['known_prefix'] == 0 and r['headbutt_same_turn'] for r in res),
                   anomalies=sum(bool(r['anomalies']) for r in res))
    (out / 'known_prefix.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


def cmd_analyze(out):
    """Aggregate descriptive analysis of labels (no per-state claims)."""
    xs = {(x['fight_id'], x['step']): x for x in eligible_states(out)}
    labels = [json.loads(l) for l in (out / 'labels.jsonl').read_text().splitlines()]
    ok = [l for l in labels if l['status'] == 'completed' and l['playouts'] and len(l['playouts']) == K
          and sorted(p['particle'] for p in l['playouts']) == list(range(K))]
    res = dict(eligible=len(xs), label_records=len(labels), statuses=dict(collections.Counter(l['status'] for l in labels)),
               complete_k_states=len(ok), menu_mismatches=sum(not l.get('menu_matches_row', True) for l in labels),
               missing_note='states without exactly K completed continuations are excluded from all estimates')
    sel = [xs[(l['fight_id'], l['step'])] for l in ok]
    pool = json.loads((out / 'sample.json').read_text())['pool_fights']
    w_state = np.array([1 / x['inclusion_prob'] for x in sel])
    res['fight_uniform'] = summarize_labels(sel, ok)
    res['state_weighted'] = summarize_labels(sel, ok, w_state)
    res['state_weighted']['effective_n'] = float(w_state.sum() ** 2 / (w_state ** 2).sum())
    wins = [sum(p['won'] for p in l['playouts']) for l in ok]
    res['wins_distribution'] = dict(sorted(collections.Counter(wins).items()))
    def table(key, edges=None):
        groups = collections.defaultdict(list)
        for x, l, w in zip(sel, ok, wins):
            g = key(x); groups[g].append((x, w))
        rows = {}
        for g, items in sorted(groups.items()):
            v = np.array([x['update15_value'] / 100 for x, _ in items]); y = np.array([w / K for _, w in items])
            lw = np.array([float(x['learner_won']) for x, _ in items])
            n = len(items)
            rows[str(g)] = dict(states=n, fights=len({x['fight_id'] for x, _ in items}), mean_V=float(v.mean()),
                                mean_T=float(y.mean()), se_T=float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                                mean_learner_outcome=float(lw.mean()),
                                debiased_mse=float(debiased_sq_error(v, np.array([w for _, w in items], float), K).mean()))
        return rows
    vb = lambda x: ('[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,.9)', '[.9,1]')[sum(x['update15_value'] / 100 >= e for e in (.2, .4, .6, .8, .9))]
    res['by_value_bin'] = table(vb)
    res['by_turn'] = table(lambda x: min(x['turn'], 6))
    res['by_stratum'] = table(lambda x: x['stratum'])
    res['by_learner_outcome'] = table(lambda x: 'learner_won' if x['learner_won'] else 'learner_lost')
    res['notes'] = ('T = deployed MCTS teacher continuation win frequency from current public-particle sampler roots (k=0, '
                    'no identified known-order violation); learner outcome = historical exploratory behaviour return; V = update15 '
                    'value. Differences are descriptive cross-policy discrepancies, not value errors or teacher advantage over '
                    'update15. Uncertainty: conditional on fixed models, frozen sample design and Monte Carlo; debiased MSE can be negative.')
    (out / 'analysis.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['snapshot', 'gates', 'sample', 'smoke', 'label', 'known_prefix', 'eligible', 'analyze'])
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args(); (a.out / 'logs').mkdir(parents=True, exist_ok=True)
    dict(snapshot=cmd_snapshot, gates=cmd_gates, sample=cmd_sample, smoke=cmd_smoke, label=cmd_label, known_prefix=cmd_known_prefix, eligible=cmd_eligible, analyze=cmd_analyze)[a.stage](a.out)


if __name__ == '__main__':
    main()
