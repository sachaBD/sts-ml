#!/usr/bin/env python3
"""Phase 3 smoke (<=6 adapter games, persistent ledger) + zero-game sampler/force/query tests. Run from repo root."""
import json, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branch as B  # noqa: E402

SM = B.OUT / 'smoke'; RES = {'started': B.now(), 'checks': {}, 'queries': 0}
END_TURN = 2147483648


def check(name, ok, detail=None, hard=True):
    RES['checks'][name] = {'ok': bool(ok), 'detail': detail}
    print(B.now(), 'CHECK', name, 'OK' if ok else 'FAIL', json.dumps(detail, default=str)[:700], flush=True)
    if hard and not ok: raise SystemExit(f'smoke check failed: {name}')


def game(binary, model, job):
    led = SM / 'smoke-ledger.jsonl'; prior = len(led.read_text().splitlines()) if led.exists() else 0
    key = {k: v for k, v in job.items() if k not in ('start', 'ops', 'chain')}
    for l in (led.read_text().splitlines() if led.exists() else []):  # reuse a passed game of an earlier attempt, never re-dispatch
        x = json.loads(l)
        if x.get('job') == key and x.get('status') in ('completed', 'capped'):
            print(B.now(), 'REUSED smoke game', json.dumps(key)[:200], flush=True); return x
    if prior + 1 > 6: raise SystemExit('phase3 smoke game budget (6) exhausted')
    rec = B.continuation(binary, model, job)
    with open(led, 'a') as f: f.write(json.dumps(rec) + '\n')
    return rec


def raw(binary, model, reqs):
    resp, rec = B.call(binary, model, reqs, timeout=600)
    if resp is None or any(r.get('error') for r in resp): raise SystemExit(f'raw call failed {rec} {resp}')
    return resp


def freeze():
    fz = B.OUT / 'frozen'
    if fz.exists(): raise SystemExit('frozen exists')
    fz.mkdir(parents=True); src = Path('experiments/combat-agent-rescue/counterfactual-rescue/phase3')
    files = {'pv_continuation': B.BUILD_BIN, 'continuation.cpp': Path('apps/continuation/continuation.cpp'), 'branch.py': src / 'branch.py',
             'smoke.py': src / 'smoke.py'}
    for n, p in files.items(): shutil.copy2(p, fz / n)
    (fz / 'sha256.json').write_text(json.dumps({n: B.sha(fz / n) for n in files}, indent=1))
    return fz / 'pv_continuation'


def main():
    SM.mkdir(parents=True, exist_ok=True); binary = freeze(); P = B.paths()
    # 1. default semantics parity with the new binary: dev:0 rows vs recorded encoding
    dev = {r['fight_id']: r for r in pq.read_table(B.DEV / 'starts.parquet').to_pylist()}; fid = 'demon-form-decoupled-policy-v1:dev:0'
    orig = B.D6 / 'id=single-deck-demon-form-v1/out/iter015/model/model.onnx'
    sp = SM / 'parity.parquet'
    # true_state request (continuation() samples a particle; parity needs the true state) -> direct call
    led = SM / 'smoke-ledger.jsonl'; prior = len(led.read_text().splitlines()) if led.exists() else 0
    old = [json.loads(l) for l in (led.read_text().splitlines() if led.exists() else []) if json.loads(l).get('kind') == 'parity']
    if old and old[0]['response'] and sp.exists(): resp = old[0]['response']; print('REUSED parity game', flush=True)
    else:
      if prior + 1 > 6: raise SystemExit('budget')
      resp, crec = B.call(binary, orig, [{'start': dev[fid]['start'], 'ops': [], 'true_state': True, 'mode': 'learner', 'sims': 2000,
                                        'fight_id': fid, 'shard_path': str(sp)}])
      with open(led, 'a') as f: f.write(json.dumps({'kind': 'parity', **crec, 'response': resp}) + '\n')
    ref = [r for r in B.rows(B.DEV / 'baseline/encoded/rows.parquet') if r['fight_id'] == fid]
    check('default_parity_dev0_rows', resp and B.rows(sp) == ref, {'rows': len(ref)})
    # 2. replay determinism of the first pool episode (exact rows/outcome)
    eps = B.pool_episodes(); ep = eps[0]
    rec = game(binary, ep['model'], B.replay_job(ep, SM / 'replay.parquet', P))
    m = B.check_replay(rec, ep) if rec['status'] != 'error' else rec
    check('replay_exact', m is None, {'episode': ep['fight_id'], 'mismatch': m})
    el, anomalies = B.eligible(rec)
    check('eligible_states', el and not anomalies, {'eligible': el, 'anomalies': anomalies})
    i = el[0]; chain = [{'particle': ep['particle'], 'ops': rec['actions'][:i]}]
    base = {'start': P[ep['path']]['start'], 'ops': P[ep['path']]['ops']['early'], 'chain': chain}
    # 3. zero-game: reconstruction, sampler canonicalization (leak test), query invariance
    r0, r1 = raw(binary, None, [dict(base, mode='roots', sims=0, particles=list(range(8))),
                                dict(base, mode='roots', sims=0, particles=list(range(8)), perturb=True)])
    row = B.rows(rec['job']['shard_path'])[i]
    check('chain_reconstructs_menu', r0['legal_actions'] == row['moves'], {'legal': r0['legal_actions']})
    sem = [a['root_semantic'] == b['root_semantic'] for a, b in zip(r0['results'], r1['results'])]
    exact = [a['root_fingerprint'] == b['root_fingerprint'] for a, b in zip(r0['results'], r1['results'])]
    check('particles_invariant_to_hidden_order_and_rng', all(sem), {'semantic_equal': sem, 'bitwise_equal_incl_uniqueId': exact,
          'true_differs': r0['true_fingerprint'] != r1['true_fingerprint']})
    q0, q1 = raw(binary, B.U3, [dict(base, mode='query', sims=2000), dict(base, mode='query', sims=2000, perturb=True)]); RES['queries'] += 2
    feats = dict(zip(('context', 'cards', 'monsters', 'potions', 'relics', 'actions'), q0['query']['features']))
    check('query_features_equal_replayed_row', all(list(np.float32(feats[k])) == list(np.float32(row[k])) for k in feats))
    check('query_invariant_to_hidden_order_and_rng', q0['query'] == q1['query'],
          {'learner': [q0['query']['learner']['action'], q1['query']['learner']['action']],
           'teacher': [q0['query']['teacher']['action'], q1['query']['teacher']['action']]})
    # 4. root-before-action: forced end turn inside particles (state with END_TURN legal)
    j_end = next((k for k in el if END_TURN in B.rows(rec['job']['shard_path'])[k]['moves']), None)
    check('end_turn_state_found', j_end is not None, {'step': j_end})
    base_e = dict(base, chain=[{'particle': ep['particle'], 'ops': rec['actions'][:j_end]}])
    f0, f1 = raw(binary, None, [dict(base_e, mode='roots', sims=0, particles=[0, 1, 2, 3], force=END_TURN)] * 2)
    fps = [x['after_force_fingerprint'] for x in f0['results']]; hands = [x['after_force_hand'] for x in f0['results']]
    strip = lambda r: [{k: v for k, v in x.items() if k != 'seconds'} for x in r['results']]
    check('force_after_root_sampling', len(set(fps)) == 4 and strip(f0) == strip(f1) and len({json.dumps(h) for h in hands}) > 1,
          {'distinct_after': len(set(fps)), 'hands': hands, 'repeat_identical': strip(f0) == strip(f1)})
    bad, _ = B.call(binary, None, [dict(base_e, mode='roots', sims=0, particles=[0], force=999999)])
    check('illegal_force_hard_error', bad and 'illegal' in (bad[0].get('error') or ''), bad)
    # 5. forced end-turn continuation game
    g = game(binary, B.U3, dict(base_e, particle=900, force=END_TURN, kind='smoke-endturn'))
    check('forced_endturn_continuation', g['status'] in ('completed', 'capped') and g['forced'] == END_TURN, {k: g.get(k) for k in ('status', 'won', 'continuation_actions')})
    # 6. <=3 candidate continuations at the same sampled root (baseline, teacher, other)
    cands = B.candidates({'episode': ep['fight_id']}, q0['query'])[:3]
    gs = [game(binary, B.U3, dict(base, particle=901, force=c['action'], kind='smoke-candidate')) for c in cands]
    check('candidates_same_root', all(x['status'] in ('completed', 'capped') for x in gs) and len({x['root_fingerprint'] for x in gs}) == 1,
          {'candidates': [[c['action'], c['source'], x['status'], x.get('won')] for c, x in zip(cands, gs)]})
    RES['finished'] = B.now(); RES['games'] = len((SM / 'smoke-ledger.jsonl').read_text().splitlines())
    (B.OUT / 'smoke-passed.json').write_text(json.dumps(RES, indent=1, default=str)); print('SMOKE PASSED', RES['games'], flush=True)


if __name__ == '__main__': main()
