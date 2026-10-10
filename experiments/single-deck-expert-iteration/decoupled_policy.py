"""Evaluator-decoupled policy intervention (Demon Form): update15 VALUE everywhere + fitted unweighted-epoch30 POLICY.

One composite ONNX graph returns (value from network A, policy_logits from network B) for the same six inputs, so
the unchanged frozen native worker uses A's value and B's logits at every root and every leaf (one Evaluator, one
session). Search settings are untouched (only the model path differs). Stages, each separately approved:

  export  : standalone B ONNX, composite(A,A), composite(A,B); fingerprints of every file incl. external data.
  gate0   : native `pv_worker evaluate` bitwise parity (no gameplay) + fail-closed checks.
  gates12 : 5 consumed monitor seeds: original A reproduces recorded iter015/eval exactly (G1); composite(A,A) = G1 (G2).
  seeds   : freeze 100 fresh development seeds (manifest + sha) before any main game.
  main    : 100 paired games per arm (baseline original A ONNX vs composite(A,B)), 10 workers, no retries.
  report  : paired summary (gap, 95% interval, discordance, timing, Dual Wield / setup descriptives).

A = runs/.../single-deck-demon-form-v1/out/iter015/model (update15); B = demon-form-correction-fit-v1
unweighted/epoch030.pt (fixed pre-selected endpoint). Fixed 2k simulations; not an equal-wall-time claim.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np
import onnx
import pyarrow.parquet as pq
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import correction_absorption as C  # noqa: E402
from agents.combat.pv.model import CONTRACT, NAMES, WIDTHS, PolicyValue  # noqa: E402

EXP = 'demon-form-decoupled-policy-v1'
RUNS = Path('runs/schema=combat_v4')
SRC = C.SRC
A_DIR = SRC / 'iter015/model'
B_PT = C.DATE / 'id=demon-form-correction-fit-v1/out/unweighted/epoch030.pt'
WORKER = SRC / 'frozen/pv_worker'
WORKER_SHA = '58c5c4ab8d3dbbd6868a22bb86b467b08d3686fd8e0f368206b9b5368d83a6b0'
A_PT_SHA = '97059033e330c47e9d6228315197e91608776d870c65e49a925253aafe8614c9'
SIMS, WORKERS, N_MAIN, N_GATE = 2000, 10, 100, 5
STARTS_NAMES = {'starts.parquet', 'new-starts.parquet', 'final-reserved.parquet', 'monitor.parquet', 'bootstrap.parquet',
                'teacher-starts.parquet', 'pool.parquet'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fingerprint(directory):
    return {p.name: sha(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = PolicyValue(ck['width'], ck.get('value_activation', 'softplus')); net.load_state_dict(ck['state_dict'])
    return net.eval()


class Composite(nn.Module):
    """value from `value_net`, policy logits from `policy_net`; both in eval mode (the model has no dropout/BN)."""

    def __init__(self, value_net, policy_net):
        super().__init__()
        self.value_net, self.policy_net = value_net.eval(), policy_net.eval()

    def forward(self, context, cards, monsters, potions, relics, actions):
        value, _ = self.value_net(context, cards, monsters, potions, relics, actions)
        _, logits = self.policy_net(context, cards, monsters, potions, relics, actions)
        return value, logits


def export(module, example, path, props):
    """Same exporter call/dynamic dims as PolicyValue.export; metadata pv_contract plus provenance props."""
    module.eval(); path.parent.mkdir(parents=True, exist_ok=True)
    dims = {n: {0: torch.export.Dim('batch')} for n in NAMES}
    for n in NAMES[1:]: dims[n][1] = torch.export.Dim(n + '_count')
    torch.onnx.export(module, tuple(example[n] for n in NAMES), str(path), dynamo=True, input_names=list(NAMES),
                      output_names=['value', 'policy_logits'], dynamic_shapes=dims)
    graph = onnx.load(path)
    onnx.helper.set_model_props(graph, {'pv_contract': CONTRACT, **props})
    onnx.save(graph, path)


def check_onnx(path):
    """Fail closed unless the graph has the native worker's exact input/output schema and contract metadata."""
    m = onnx.load(path, load_external_data=False)
    props = {p.key: p.value for p in m.metadata_props}
    if props.get('pv_contract') != CONTRACT: raise ValueError(f'{path}: pv_contract mismatch')
    ins = [(i.name, len(i.type.tensor_type.shape.dim), i.type.tensor_type.shape.dim[-1].dim_value,
            i.type.tensor_type.elem_type) for i in m.graph.input]
    want = [(n, 2 if k == 0 else 3, w, onnx.TensorProto.FLOAT) for k, (n, w) in enumerate(zip(NAMES, WIDTHS))]
    if ins != want: raise ValueError(f'{path}: input schema {ins} != {want}')
    if [o.name for o in m.graph.output] != ['value', 'policy_logits']: raise ValueError(f'{path}: output names')
    return props


def cmd_export(out):
    a, b = load(A_DIR / 'model.pt'), load(B_PT)
    if sha(A_DIR / 'model.pt') != A_PT_SHA: raise ValueError('update15 checkpoint changed')
    rows = C.Rows(C.TEACHER); example = rows.shard.batch(rows.shard.rows[:2])[0]
    m = out / 'models'
    if m.exists(): raise FileExistsError(f'{m} exists; refuse to overwrite exported models')
    src = dict(value_source=str(A_DIR / 'model.pt'), value_sha=A_PT_SHA, policy_source=str(B_PT), policy_sha=sha(B_PT))
    export(b, example, m / 'policy_b/model.onnx', {'width': str(b.width), 'value_activation': b.value_activation,
                                                   'source': str(B_PT), 'source_sha': src['policy_sha']})
    export(Composite(a, load(A_DIR / 'model.pt')), example, m / 'composite_aa/model.onnx',
           {'composite': 'value=A policy=A', 'value_sha': A_PT_SHA, 'policy_sha': A_PT_SHA})
    export(Composite(a, b), example, m / 'composite_ab/model.onnx',
           {'composite': 'value=A policy=B', 'value_sha': A_PT_SHA, 'policy_sha': src['policy_sha']})
    manifest = dict(sources=src, original_a_onnx=fingerprint(A_DIR),
                    exported={d.name: fingerprint(d) for d in sorted(m.iterdir())},
                    metadata={d.name: check_onnx(d / 'model.onnx') for d in sorted(m.iterdir())},
                    script_sha=sha(__file__))
    (out / 'export.json').write_text(json.dumps(manifest, indent=1)); print(json.dumps(manifest, indent=1))


# ---------------- gate 0: native evaluation parity ----------------

BATCH_SIZES = (1, 2, 3, 7, 32, 64, 5, 17)


def state_json(shard, r):
    out = {}
    for n, w in zip(NAMES, WIDTHS):
        v, o = shard.flat[n]; x = v[o[r]:o[r + 1]]
        out[n] = x.tolist() if n == 'context' else x.reshape(-1, w).tolist()
    return out


def batches(shards):
    """Fixed sequence of mixed-size batches over teacher + monitor rows (varying action/token counts)."""
    states = [state_json(s, r) for s in shards for r in s.rows]
    out, i, k = [], 0, 0
    while i < len(states):
        n = BATCH_SIZES[k % len(BATCH_SIZES)]; out.append(states[i:i + n]); i += n; k += 1
    return out


def native(model, lines):
    text = lines if isinstance(lines, str) else ''.join(json.dumps(b) + '\n' for b in lines)
    proc = subprocess.run([str(WORKER), 'evaluate', str(model)], input=text,
                          text=True, capture_output=True, timeout=600)
    if proc.returncode: raise RuntimeError(f'native evaluate failed: {proc.stderr[-800:]}')
    return [p for line in proc.stdout.splitlines() for p in json.loads(line)]


def f32(x):
    return np.asarray(x, np.float64).astype(np.float32)


def cmd_gate0(out):
    if sha(WORKER) != WORKER_SHA: raise ValueError('frozen worker changed')
    m = out / 'models'; ex = json.loads((out / 'export.json').read_text())
    for d, files in ex['exported'].items():
        if fingerprint(m / d) != files: raise ValueError(f'exported model {d} changed since export')
    shards = [C.Rows(C.TEACHER).shard, C.Rows(C.VAL).shard]
    lines = batches(shards); n = sum(map(len, lines))
    alt = batches(shards[::-1])  # different batch composition/order for a composition-invariance check
    res = dict(states=n, batches=len(lines), batch_sizes=list(BATCH_SIZES))
    lines, alt = ''.join(json.dumps(b) + '\n' for b in lines), ''.join(json.dumps(b) + '\n' for b in alt)
    A = native(A_DIR / 'model.onnx', lines); AA = native(m / 'composite_aa/model.onnx', lines)
    AB = native(m / 'composite_ab/model.onnx', lines); B = native(m / 'policy_b/model.onnx', lines)
    A_alt = native(A_DIR / 'model.onnx', alt)

    def bitwise(x, y, key):
        bad = sum(not np.array_equal(f32(p[key]), f32(q[key])) for p, q in zip(x, y))
        lens = sum(len(np.atleast_1d(p[key])) != len(np.atleast_1d(q[key])) for p, q in zip(x, y))
        return dict(mismatched_states=int(bad), length_mismatches=int(lens), compared=len(x))
    res['a_vs_composite_aa'] = dict(value=bitwise(A, AA, 'value'), logits=bitwise(A, AA, 'logits'))
    res['composite_ab_value_vs_a'] = bitwise(A, AB, 'value')
    res['composite_ab_logits_vs_b'] = bitwise(B, AB, 'logits')
    res['policy_differs_from_a'] = int(sum(not np.array_equal(f32(p['logits']), f32(q['logits'])) for p, q in zip(A, AB)))
    # composition invariance (informative): map alt order back
    k = len(shards[1].rows); A_alt = A_alt[k:] + A_alt[:k]
    res['a_batch_composition_invariance'] = dict(value=bitwise(A, A_alt, 'value'), logits=bitwise(A, A_alt, 'logits'))
    # native B vs torch B
    b = load(B_PT); worst_v = worst_l = 0.0; i = 0
    with torch.no_grad():
        for s in shards:
            for start in range(0, len(s.rows), 256):
                idx = s.rows[start:start + 256]; bt = s.batch(idx)[0]; v, l = b(*(bt[nm] for nm in NAMES))
                for j in range(len(idx)):
                    p = B[i]; i += 1; nl = len(p['logits'])
                    if nl != int(s.counts['actions'][idx[j]]): raise ValueError('legal action count mismatch')
                    worst_v = max(worst_v, abs(float(v[j]) - p['value']))
                    worst_l = max(worst_l, float(np.abs(l[j, :nl].numpy() - f32(p['logits'])).max()))
    res['native_b_vs_torch_b'] = dict(max_value_diff=worst_v, max_logit_diff=worst_l)
    # fail closed: tampered metadata must be rejected by the worker and by check_onnx
    bad = out / 'gate0-tampered'; shutil.rmtree(bad, ignore_errors=True); shutil.copytree(m / 'composite_ab', bad)
    g = onnx.load(bad / 'model.onnx', load_external_data=False)
    for p in g.metadata_props:
        if p.key == 'pv_contract': p.value = 'wrong'
    onnx.save(g, bad / 'model.onnx')
    try:
        native(bad / 'model.onnx', lines.split('\n', 1)[0] + '\n'); res['tampered_rejected_by_worker'] = False
    except RuntimeError:
        res['tampered_rejected_by_worker'] = True
    try:
        check_onnx(bad / 'model.onnx'); res['tampered_rejected_by_check'] = False
    except ValueError:
        res['tampered_rejected_by_check'] = True
    shutil.rmtree(bad)
    ok = (all(v['mismatched_states'] == 0 and v['length_mismatches'] == 0 for v in
              [*res['a_vs_composite_aa'].values(), res['composite_ab_value_vs_a'], res['composite_ab_logits_vs_b']])
          and res['native_b_vs_torch_b']['max_logit_diff'] < 1e-4 and res['native_b_vs_torch_b']['max_value_diff'] < 1e-3
          and res['tampered_rejected_by_worker'] and res['tampered_rejected_by_check'] and res['policy_differs_from_a'] > 0)
    res['pass'] = bool(ok)
    (out / 'gate0.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    if not ok: raise SystemExit('GATE 0 FAILED: stop')


# ---------------- gameplay stages (bench play, frozen worker, no retries) ----------------

def bench(model, starts, dest, log):
    if dest.exists(): raise FileExistsError(f'{dest} exists; no resume/retry')
    cmd = [sys.executable, '-m', 'apps.human_champ.bench', 'play', '--starts', str(starts), '--agent', 'pv',
           '--model', str(model), '--sims', str(SIMS), '--workers', str(WORKERS), '--worker', str(WORKER),
           '--out', str(dest)]
    with open(log, 'w') as f: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    return dest


def results(d):
    return {r['fight_id']: r for r in map(json.loads, (Path(d) / 'results.jsonl').read_text().splitlines())}


def visits(r):
    return [(s['step'], sorted((c['action'], c['visits']) for c in s['children'])) for s in r.get('search', [])]


def cmd_gates12(out):
    from apps.run_rl.combat_loop import write_starts
    if not json.loads((out / 'gate0.json').read_text())['pass']: raise SystemExit('gate0 not passed')
    recorded = results(SRC / 'iter015/eval')
    ids = sorted(recorded)[:N_GATE]
    rows = [r for r in pq.read_table(SRC / 'monitor.parquet').to_pylist() if r['fight_id'] in ids]
    assert len(rows) == N_GATE
    g = out / 'gates12'; g.mkdir(parents=True, exist_ok=False); (out / 'logs').mkdir(exist_ok=True)
    write_starts(rows, g / 'starts.parquet')
    g1 = bench(A_DIR / 'model.onnx', g / 'starts.parquet', g / 'g1', out / 'logs/g1.log')
    r1 = results(g1)
    res = dict(seeds=ids, g1=[], g2=[])
    for fid in ids:
        res['g1'].append(dict(fight_id=fid, status=r1[fid]['status'],
                              actions_equal=r1[fid].get('fight', {}).get('actions') == recorded[fid]['fight']['actions'],
                              visits_equal=visits(r1[fid]) == visits(recorded[fid])))
    if not all(x['status'] == 'completed' and x['actions_equal'] and x['visits_equal'] for x in res['g1']):
        res['pass'] = False; (out / 'gates12.json').write_text(json.dumps(res, indent=1))
        raise SystemExit('GATE 1 FAILED: stop')
    g2 = bench(out / 'models/composite_aa/model.onnx', g / 'starts.parquet', g / 'g2', out / 'logs/g2.log')
    r2 = results(g2)
    for fid in ids:
        res['g2'].append(dict(fight_id=fid, status=r2[fid]['status'],
                              actions_equal=r2[fid].get('fight', {}).get('actions') == r1[fid]['fight']['actions'],
                              visits_equal=visits(r2[fid]) == visits(r1[fid]),
                              root_values_equal=[s['root_value'] for s in r2[fid].get('search', [])] ==
                              [s['root_value'] for s in r1[fid]['search']]))
    res['pass'] = all(x['status'] == 'completed' and x['actions_equal'] and x['visits_equal'] and x['root_values_equal']
                      for x in res['g2'])
    (out / 'gates12.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    if not res['pass']: raise SystemExit('GATE 2 FAILED: stop')


def used_seeds():
    """Every recorded start seed in starts-like files under runs/schema=combat_v4 (all dates)."""
    used, files, skipped = set(), [], []
    for p in sorted(RUNS.rglob('*.parquet')):
        if p.name not in STARTS_NAMES: continue
        try:
            t = pq.read_table(p, columns=['start'])
            used.update(int(s['seed']) for s in t.column('start').to_pylist()); files.append(str(p))
        except Exception as e:  # noqa: BLE001 - recorded, not silently ignored
            skipped.append([str(p), type(e).__name__])
    return used, files, skipped


def cmd_seeds(out):
    from apps.run_rl.combat_loop import write_starts
    from apps.run_rl.single_deck import generate
    dest = out / 'main/starts.parquet'
    if dest.exists(): raise FileExistsError('seed manifest already frozen')
    used, files, skipped = used_seeds()
    required = [SRC / f for f in ['monitor.parquet', 'final-reserved.parquet', 'bootstrap.parquet']] + \
               [SRC / f'iter{i:03d}/starts.parquet' for i in range(1, 16)]
    missing = [str(p) for p in required if str(p) not in files]
    if missing: raise ValueError(f'required seed sources not read: {missing}')
    base = copy.deepcopy(pq.read_table(SRC / 'monitor.parquet').to_pylist()[0])
    before = set(used)
    rows = generate(base, EXP, 'dev', N_MAIN, used)
    seeds = [r['start']['seed'] for r in rows]
    assert len(set(seeds)) == N_MAIN and not set(seeds) & before
    dest.parent.mkdir(parents=True); write_starts(rows, dest)
    manifest = dict(namespace=EXP, split='dev', n=N_MAIN, starts_sha=sha(dest), seeds=seeds,
                    excluded_seed_count=len(before), seed_files_read=len(files), seed_files_skipped=skipped,
                    base_loadout_from=str(SRC / 'monitor.parquet'), note='development set, not final confirmation')
    (out / 'main/manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'seeds'}, indent=1))


def cmd_main(out):
    if not json.loads((out / 'gates12.json').read_text())['pass']: raise SystemExit('gates not passed')
    man = json.loads((out / 'main/manifest.json').read_text())
    if sha(out / 'main/starts.parquet') != man['starts_sha']: raise ValueError('frozen starts changed')
    ex = json.loads((out / 'export.json').read_text())
    if fingerprint(out / 'models/composite_ab') != ex['exported']['composite_ab']: raise ValueError('composite changed')
    if fingerprint(A_DIR) != ex['original_a_onnx']: raise ValueError('update15 ONNX changed')
    bench(A_DIR / 'model.onnx', out / 'main/starts.parquet', out / 'main/baseline', out / 'logs/baseline.log')
    bench(out / 'models/composite_ab/model.onnx', out / 'main/starts.parquet', out / 'main/intervention',
          out / 'logs/intervention.log')
    cmd_report(out)


def describe(out, arm):
    """Dual Wield selections and first Demon Form / Corruption play turns, from canonical re-encoding (replay only)."""
    from agents.combat.pv.data import collect
    d = out / 'main' / arm; enc = d / 'encoded'
    if not (enc / 'rows.parquet').exists():
        collect([d / 'fights-0.parquet'], [d / 'search-0.parquet'], [39], enc, WORKER)
    rows = pq.read_table(enc / 'rows.parquet', columns=['fight_id', 'step', 'turn', 'moves', 'actions']).to_pylist()
    fights = {r['fight_id']: r['actions'] for r in pq.read_table(d / 'fights-0.parquet', columns=['fight_id', 'actions']).to_pylist()}
    dw = collections.Counter(); first = collections.defaultdict(dict)
    for r in rows:
        t = np.asarray(r['actions'], np.float32).reshape(-1, 261)[r['moves'].index(fights[r['fight_id']][r['step']])]
        if t[0] == C.KIND_SELECT and t[1] == C.TASK_DUAL_WIELD: dw[C.label(t)] += 1
        if t[0] == C.KIND_CARD and int(t[8]) in (107, 83):
            first[r['fight_id']].setdefault(C.CARD[int(t[8])], r['turn'])
    turns = {c: [f[c] for f in first.values() if c in f] for c in ['Demon Form', 'Corruption']}
    return dict(dual_wield_selections=dict(dw.most_common()),
                first_play_turn={c: dict(fights_played=len(v), mean=statistics.mean(v) if v else None) for c, v in turns.items()},
                fights=len(fights))


def cmd_report(out):
    man = json.loads((out / 'main/manifest.json').read_text())
    ids = [f'{EXP}:dev:{i}' for i in range(N_MAIN)]
    b, t = results(out / 'main/baseline'), results(out / 'main/intervention')
    status = {arm: collections.Counter(r.get(i, {}).get('status', 'missing') for i in ids) for arm, r in [('baseline', b), ('intervention', t)]}
    pairs = [(b[i]['fight']['won'], t[i]['fight']['won']) for i in ids
             if b.get(i, {}).get('status') == 'completed' and t.get(i, {}).get('status') == 'completed']
    n = len(pairs); d = [int(y) - int(x) for x, y in pairs]
    gap = sum(d) / n if n else float('nan')
    se = math.sqrt(sum((x - gap) ** 2 for x in d) / (n * (n - 1))) if n > 1 else float('nan')
    disc = dict(both_won=sum(x and y for x, y in pairs), baseline_only=sum(x and not y for x, y in pairs),
                intervention_only=sum(y and not x for x, y in pairs), both_lost=sum(not x and not y for x, y in pairs))
    timing = {}
    for arm, r in [('baseline', b), ('intervention', t)]:
        s = [r[i]['seconds'] for i in ids if i in r]
        dec = [sum(x.get('seconds', 0) for x in r[i].get('stats', [])) for i in ids if r.get(i, {}).get('status') == 'completed']
        timing[arm] = dict(n=len(s), wall_per_game_mean=statistics.mean(s), wall_per_game_median=statistics.median(s),
                           decision_seconds_mean=statistics.mean(dec) if dec else None,
                           decision_seconds_median=statistics.median(dec) if dec else None)
    errors = {arm: [dict(fight_id=i, status=r[i]['status'], error=r[i].get('error')) for i in ids
                    if i in r and r[i]['status'] != 'completed'] for arm, r in [('baseline', b), ('intervention', t)]}
    rep = dict(intended=N_MAIN, starts_sha=man['starts_sha'], statuses={k: dict(v) for k, v in status.items()},
               completed_pairs=n, wins=dict(baseline=sum(x for x, _ in pairs), intervention=sum(y for _, y in pairs)),
               gap_intervention_minus_baseline=gap, paired_se=se,
               approx_95=[gap - 1.959964 * se, gap + 1.959964 * se], discordance=disc, timing=timing, errors=errors,
               note='Normal-approximation interval over the 100 seed pairs, conditional on the one fitted policy model; '
                    'development seeds, fixed 2k simulations, not an equal-wall-time or final-test claim.')
    try:
        rep['descriptive'] = {arm: describe(out, arm) for arm in ['baseline', 'intervention']}
    except Exception as e:  # noqa: BLE001 - descriptives must not hide the primary result
        rep['descriptive_error'] = repr(e)
    (out / 'main/report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['export', 'gate0', 'gates12', 'seeds', 'main', 'report'])
    ap.add_argument('--out', type=Path, default=C.DATE / f'id={EXP}/out')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    dict(export=cmd_export, gate0=cmd_gate0, gates12=cmd_gates12, seeds=cmd_seeds, main=cmd_main, report=cmd_report)[a.stage](a.out)


if __name__ == '__main__':
    main()
