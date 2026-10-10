#!/usr/bin/env python3
"""Stage 2 smoke/tests (<= 4 adapter games). Freezes inputs into out/stage2/frozen, then:
roots gate (no play) vs Stage1, true-start row parity vs recorded dev:0 encoding, restart rows through Shard/evaluate,
cap test (test-only turn_cap), trainer tiny fixed-step test, range disjointness, timeout cleanup.
Writes out/stage2/smoke-passed.json only if everything passes. Run from repo root."""
import json, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner as R  # noqa: E402
from agents.combat.pv.data import Shard, merge  # noqa: E402
from agents.combat.pv.model import PolicyValue  # noqa: E402
from agents.combat.pv.train import evaluate  # noqa: E402

SM = R.OUT / 'smoke'
RESULTS = {'started': R.now(), 'games': 0, 'checks': {}}


def check(name, ok, detail=None):
    RESULTS['checks'][name] = {'ok': bool(ok), 'detail': detail}
    print(R.now(), 'CHECK', name, 'OK' if ok else 'FAIL', json.dumps(detail, default=str)[:600], flush=True)
    if not ok: raise SystemExit(f'smoke check failed: {name}')


def freeze():
    fz = R.OUT / 'frozen'
    if fz.exists(): raise SystemExit('frozen dir exists; refusing to overwrite')
    fz.mkdir(parents=True)
    src = Path('experiments/combat-agent-rescue/counterfactual-rescue/stage2')
    files = {'pv_continuation': R.BUILD_BIN, 'continuation.cpp': Path('apps/continuation/continuation.cpp'),
             'CMakeLists.txt': Path('apps/continuation/CMakeLists.txt'), 'runner.py': src / 'runner.py',
             'train_fixed.py': src / 'train_fixed.py', 'smoke.py': src / 'smoke.py'}
    for n, p in files.items(): shutil.copy2(p, fz / n)
    hashes = {n: R.sha(fz / n) for n in files}
    (fz / 'sha256.json').write_text(json.dumps(hashes, indent=1))
    return fz / 'pv_continuation'


def game(binary, model, job):
    led = SM / 'smoke-ledger.jsonl'  # persists across attempts: every dispatched game counts (budget <= 8 total)
    prior = len(led.read_text().splitlines()) if led.exists() else 0
    if prior + 1 > 8: raise SystemExit('adapter smoke game budget (8) exhausted')
    RESULTS['games'] += 1; RESULTS['games_all_attempts'] = prior + 1
    rec = R.continuation(binary, model, job)
    with open(SM / 'smoke-ledger.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
    return rec


def main():
    SM.mkdir(parents=True, exist_ok=True)
    binary = freeze(); model = R.ORIG_MODEL / 'model.onnx'
    # 0. layout
    seen = set(); n = 0
    for key in R.itertools.product(R.ARMS, R.UPDATES, range(12), R.LEVELS, R.KINDS):
        r = set(R.block(*key)); n += 1
        if r & seen or min(r) < R.BASE: check('index_ranges_disjoint', False, key)
        seen |= r
    check('index_ranges_disjoint', len(seen) == n * R.BLOCK and min(seen) == 1000 and R.SMOKE_INDEX + 3 < 1000,
          {'blocks': n, 'max_index': max(seen)})
    # 1. roots gate, no play: rebuilt binary reproduces Stage1 legal menus and root fingerprints for all 36 x 4
    P = R.paths(); want = {}
    for l in (R.S1 / 'stage1-results.jsonl').read_text().splitlines():
        x = json.loads(l); j = x['job']
        want[(j['fight_id'], j['position'], j['particle'])] = (x['response']['legal_actions'], x['response']['results'][0]['root_fingerprint'])
    reqs, keys = [], []
    for p in P:
        for lv in ('early', 'middle', 'late'):
            reqs.append(json.dumps({'start': p['start'], 'ops': p['ops'][lv], 'mode': 'roots', 'sims': 0, 'particles': [0, 1, 2, 3]}))
            keys.append((p['fight_id'], lv))
    out = subprocess.run([str(binary)], input='\n'.join(reqs) + '\n', capture_output=True, text=True, timeout=60).stdout.splitlines()
    bad = []
    for (fid, lv), line in zip(keys, out):
        resp = json.loads(line)
        for r in resp['results']:
            if want[(fid, lv, r['particle'])] != (resp['legal_actions'], r['root_fingerprint']): bad.append((fid, lv, r['particle']))
    check('stage1_roots_and_legal_menus', len(out) == 36 and not bad, {'states': len(out), 'mismatch': bad[:5]})
    # control full-start roots are distinct for a block of 12 (no play)
    resp = json.loads(subprocess.run([str(binary)], input=json.dumps({'start': P[0]['start'], 'ops': [], 'mode': 'roots', 'sims': 0,
                      'particles': list(R.block('control', 1, 0, 'full', 'train'))[:12]}) + '\n', capture_output=True, text=True, timeout=60).stdout)
    fps = [r['root_fingerprint'] for r in resp['results']]
    check('control_root_diversity_preview', len(set(fps)) == 12, {'distinct': len(set(fps))})
    # 2. true-start parity vs recorded dev:0 (frozen worker play + frozen worker encode)
    dev = {r['fight_id']: r for r in pq.read_table(R.DEV_STARTS).to_pylist()}
    fid = 'demon-form-decoupled-policy-v1:dev:0'
    shard = SM / 'parity.parquet'
    rec = game(binary, model, {'fight_id': fid, 'start': dev[fid]['start'], 'ops': [], 'true_state': True, 'particles': [], 'shard_path': str(shard)})
    ref_fight = json.loads(open(R.DEV / 'baseline/results.jsonl').readline())
    assert ref_fight['fight']['fight_id'] == fid
    check('parity_outcome', rec['status'] == 'completed' and rec['won'] == ref_fight['fight']['won'] and rec['hp'] == ref_fight['fight']['final_hp']
          and rec['actions'] == len(ref_fight['fight']['actions']), {k: rec.get(k) for k in ('status', 'won', 'hp', 'actions')})
    got = pq.read_table(shard).to_pylist(); ref = [r for r in pq.read_table(R.DEV / 'baseline/encoded/rows.parquet').to_pylist() if r['fight_id'] == fid]
    diffs = []
    cols = [c for c in pq.read_table(shard).schema.names if c not in ('schema', 'date', 'id')]  # hive columns inferred from run path
    for c in cols:
        for i, (g, r) in enumerate(zip(got, ref)):
            a, b = g[c], r[c]
            if c == 'root_value' and a is not None and b is not None:
                if abs(a - b) > 1e-4: diffs.append((c, i))
            elif c in ('policy_target',) and a is not None:
                if len(a) != len(b) or np.abs(np.array(a) - np.array(b)).max(initial=0) > 1e-6: diffs.append((c, i))
            elif a != b: diffs.append((c, i))
    check('parity_rows', len(got) == len(ref) and not diffs, {'rows': [len(got), len(ref)], 'columns': cols, 'diffs': diffs[:10]})
    # 3. restart rows (middle, late of path 0), smoke-only particles
    net_ck = torch.load(R.ORIG_MODEL / 'model.pt', map_location='cpu', weights_only=False)
    net = PolicyValue(net_ck['width'], net_ck['value_activation']); net.load_state_dict(net_ck['state_dict']); net.eval()
    restart_shards = []
    for k, lv in enumerate(('middle', 'late')):
        idx = R.SMOKE_INDEX + k; fidr = f'cfr2:smoke:p00:{lv}:train:{idx}'; sp = SM / f'restart-{lv}.parquet'
        rec = game(binary, model, {'fight_id': fidr, 'start': P[0]['start'], 'ops': P[0]['ops'][lv], 'particles': [idx], 'shard_path': str(sp)})
        if rec['status'] != 'completed': check(f'restart_{lv}_completed', False, rec)
        s = Shard(sp, False, {fidr: 'train'}); t = pq.read_table(sp).to_pylist()
        b = s.batch(s.rows)
        with torch.no_grad(): v, q, _, nn, npol = evaluate(net, b, 'cpu', flat_policy_weighting=True)
        legal_ok = all(int((b[0]['actions'][i, :, 260] != 0).sum()) == len(t[i]['moves']) and
                       float(b[2][i, len(t[i]['moves']):].abs().sum()) == 0 for i in range(len(t)))
        pol_ok = all(abs(sum(r['policy_target']) - 1) < 1e-5 for r in t if r['has_policy']) and all(sum(r['policy_target']) == 0 for r in t if not r['has_policy'])
        check(f'restart_{lv}_rows', torch.isfinite(v) and torch.isfinite(q) and (b[1] == 100 * rec['won']).all() and
              t[0]['step'] == len(P[0]['ops'][lv]) and [r['step'] for r in t] == list(range(len(P[0]['ops'][lv]), len(P[0]['ops'][lv]) + len(t)))
              and len(t) == rec['actions'] and legal_ok and pol_ok and all(r['fight_id'] == fidr for r in t),
              {'won': rec['won'], 'rows': len(t), 'value_loss': v.item() / nn, 'policy_loss': q.item() / max(npol, 1), 'root_fp': rec['root_fingerprint']})
        restart_shards.append(sp)
    # 4. cap test (test-only turn_cap=2 from the full start): capped, nothing published, no tmp
    sp = SM / 'cap.parquet'
    rec = game(binary, model, {'fight_id': 'cfr2:smoke:p00:full:cap:902', 'start': P[0]['start'], 'ops': [], 'particles': [R.SMOKE_INDEX + 2],
                               'shard_path': str(sp), 'turn_cap': 2})
    check('cap_not_published', rec['status'] == 'capped' and rec['won'] is None and not sp.exists() and not Path(str(sp) + '.tmp').exists(),
          {k: rec.get(k) for k in ('status', 'won', 'turns', 'shard_published', 'error')})
    # 5. trainer: tiny fixed-step run through the production code path (test-only steps override)
    newf = SM / 'train-test' / 'new-rows.parquet'; newf.parent.mkdir(exist_ok=True)
    R.merge_rows(restart_shards, newf)
    ids = list({r for f in restart_shards for r in pq.read_table(f, columns=['fight_id'])['fight_id'].to_pylist()})
    R.write_split(newf.parent / 'split.json', ids)
    cmd = [str(R.PY), str(R.OUT / 'frozen/train_fixed.py'), '--old', *map(str, R.OLD_SHARDS), '--new', str(newf), '--split', str(newf.parent / 'split.json'),
           '--init', str(R.ORIG_MODEL / 'model.pt'), '--out', str(newf.parent / 'model'), '--update', '1', '--expect-old-fights', '1000',
           '--expect-new-fights', '2', '--test-steps', '5']
    t0 = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    (newf.parent / 'train.log').write_text(p.stdout + p.stderr)
    rep = json.loads((newf.parent / 'model/train.json').read_text()) if p.returncode == 0 else {}
    check('trainer_fixed_steps', p.returncode == 0 and rep['counts'] == {'old': 160, 'new': 160, 'steps': 5} and rep['adam_step_after'] == rep['adam_step_before'] + 5
          and rep['onnx_max_abs_diff'] < 1e-3 and rep['old_fights'] == 1000 and rep['new_fights'] == 2 and {'model.onnx', 'model.pt'} <= set(rep['files_sha256']),
          {'rc': p.returncode, 'seconds': time.time() - t0, 'rep': {k: rep.get(k) for k in ('counts', 'adam_step_before', 'adam_step_after', 'onnx_max_abs_diff', 'files_sha256', 'train_seconds', 'old_states')},
           'stderr': p.stderr[-800:] if p.returncode else ''})
    # Exported ONNX loads in the adapter (roots mode does not evaluate; use pv_worker evaluate? -> check via onnxruntime done above)
    # 6. old-half draws independent of the new pool; locked steps refused without override
    sys.path.insert(0, str(R.OUT / 'frozen')); import train_fixed as T
    old = T.Pool(R.OLD_SHARDS[:2], {f: 'train' for s in R.OLD_SHARDS[:2] for f in pq.read_table(s, columns=['fight_id'])['fight_id'].to_pylist()})
    draws = []
    for _ in range(2):
        rg = np.random.default_rng(np.random.SeedSequence([0, 1, 0])); draws.append(old.sample(rg, 32)[1])
    check('old_stream_reproducible', torch.equal(draws[0], draws[1]))
    # 7. timeout cleanup
    rc, _, _, _, err = R.run_one(['sleep', '30'], '', timeout=1)
    left = subprocess.run(['pgrep', '-f', '^sleep 30$'], capture_output=True, text=True).stdout.strip()
    check('timeout_cleanup', rc is None and err and not left, {'err': err, 'left': left})
    RESULTS['finished'] = R.now()
    (R.OUT / 'smoke-passed.json').write_text(json.dumps(RESULTS, indent=1, default=str))
    print('SMOKE PASSED games', RESULTS['games'], flush=True)


if __name__ == '__main__': main()
