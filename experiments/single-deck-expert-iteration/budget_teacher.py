"""Matched teacher gap and learned-search budget response on the 100 decoupled-dev starts (Demon Form).

Arms (new games): frozen rollout MCTS teacher at 20k simulations (deployed objective, `pv_worker teacher 20000`) and
original update15 policy+value at 10k learned simulations (`pv_worker play MODEL 10000`, all other settings as the
2k baseline). The cached 2k baseline (and composite intervention) come from demon-form-decoupled-policy-v1.
Same frozen worker, starts copied byte-identically (sha checked), no retries/substitutions, 10 workers.

  setup  : copy starts + fingerprints (no games)
  play   : 100 teacher + 100 learned-10k games (explicit timestamps)
  report : paired gaps / 95% intervals / discordance, timing, depth, root vs raw value, flips
"""
import argparse
import collections
import datetime
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent))
import correction_absorption as C  # noqa: E402
import decoupled_policy as DP  # noqa: E402
from agents.combat.pv.model import NAMES  # noqa: E402

EXP = 'demon-form-budget-teacher-v1'
DEC = C.DATE / 'id=demon-form-decoupled-policy-v1/out'
IDS = [f'{DP.EXP}:dev:{i}' for i in range(100)]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def cmd_setup(out):
    if (out / 'starts.parquet').exists(): raise FileExistsError('already set up')
    man = json.loads((DEC / 'main/manifest.json').read_text())
    if DP.sha(DEC / 'main/starts.parquet') != man['starts_sha']: raise ValueError('decoupled starts changed')
    shutil.copy2(DEC / 'main/starts.parquet', out / 'starts.parquet')
    if DP.sha(out / 'starts.parquet') != man['starts_sha']: raise ValueError('copy mismatch')
    cfg = dict(starts_sha=man['starts_sha'], starts_source=str(DEC / 'main/starts.parquet'),
               worker=str(DP.WORKER), worker_sha=DP.sha(DP.WORKER), update15_onnx=DP.fingerprint(DP.A_DIR),
               baseline_2k=str(DEC / 'main/baseline'), baseline_2k_settings=json.loads((DEC / 'main/baseline/settings.json').read_text()),
               arms=dict(teacher=dict(agent='teacher', sims=20000), learned_10k=dict(agent='pv', sims=10000)),
               workers=10, script_sha=DP.sha(__file__))
    if cfg['worker_sha'] != DP.WORKER_SHA: raise ValueError('worker changed')
    (out / 'config.json').write_text(json.dumps(cfg, indent=1)); print(json.dumps(cfg, indent=1))


def run(agent, sims, dest, starts):
    if dest.exists(): raise FileExistsError(f'{dest} exists; no resume/retry')
    cmd = [sys.executable, '-m', 'apps.human_champ.bench', 'play', '--starts', str(starts), '--agent', agent,
           '--sims', str(sims), '--workers', '10', '--worker', str(DP.WORKER), '--out', str(dest)]
    if agent == 'pv': cmd += ['--model', str(DP.A_DIR / 'model.onnx')]
    return cmd


def cmd_play(out):
    cfg = json.loads((out / 'config.json').read_text())
    if DP.sha(out / 'starts.parquet') != cfg['starts_sha'] or DP.sha(DP.WORKER) != cfg['worker_sha'] \
            or DP.fingerprint(DP.A_DIR) != cfg['update15_onnx']:
        raise ValueError('fingerprint changed since setup')
    times = {}
    for arm, (agent, sims) in [('teacher', ('teacher', 20000)), ('learned_10k', ('pv', 10000))]:
        cmd = run(agent, sims, out / arm, out / 'starts.parquet')
        times[arm] = dict(start=now())
        with open(out / f'logs/{arm}.log', 'w') as f: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        times[arm]['finish'] = now()
        (out / 'timestamps.json').write_text(json.dumps(times, indent=1)); print(arm, times[arm], flush=True)
    cmd_report(out)


def results(d):
    return {r['fight_id']: r for r in map(json.loads, (Path(d) / 'results.jsonl').read_text().splitlines())}


def paired(a, b, na='first', nb='second'):
    """b minus a over completed pairs; discordance keys name the arms explicitly."""
    pairs = [(a[i]['fight']['won'], b[i]['fight']['won']) for i in IDS
             if a.get(i, {}).get('status') == 'completed' and b.get(i, {}).get('status') == 'completed']
    n = len(pairs); d = [int(y) - int(x) for x, y in pairs]; gap = sum(d) / n
    se = math.sqrt(sum((x - gap) ** 2 for x in d) / (n * (n - 1)))
    k = sum(x and not y for x, y in pairs); m = sum(y and not x for x, y in pairs)
    p = min(1.0, 2 * sum(math.comb(k + m, i) for i in range(min(k, m) + 1)) / 2 ** (k + m)) if k + m else 1.0
    return dict(completed_pairs=n, intended=100, gap=gap, se=se, approx_95=[gap - 1.959964 * se, gap + 1.959964 * se],
                discordance={'both_won': sum(x and y for x, y in pairs), f'{na}_only_won': k, f'{nb}_only_won': m,
                             'both_lost': sum(not x and not y for x, y in pairs)}, exact_mcnemar_p=p)


def timing(r):
    out = {}
    s = [r[i]['seconds'] for i in IDS if i in r]
    out.update(n=len(s), wall_mean=statistics.mean(s), wall_median=statistics.median(s))
    for key in ['mean_depth', 'mean_turns', 'max_depth', 'max_turns']:
        v = [x[key] for i in IDS if r.get(i, {}).get('status') == 'completed' for x in r[i]['stats'] if x.get(key) is not None]
        if v: out[key] = dict(decisions=len(v), mean=statistics.mean(v), median=statistics.median(v))
    return out


def first_root(r, i):
    s = r[i].get('search') or []
    return s[0]['root_value'] if s else None


def raw_start_values():
    """Raw update15 value at each start's first decision state (same physical state for all arms)."""
    rows = C.Rows(DEC / 'main/baseline/encoded/rows.parquet')
    first = {}
    for k, r in enumerate(rows.meta):
        if r['fight_id'] not in first or r['step'] < rows.meta[first[r['fight_id']]]['step']: first[r['fight_id']] = k
    net = C.load_net(C.MODELS['update15']); ks = sorted(first.values())
    with torch.no_grad():
        b = rows.shard.batch(rows.shard.rows[ks]); v, _ = net(*(b[0][n] for n in NAMES))
    return {rows.meta[k]['fight_id']: float(x) for k, x in zip(ks, v)}


def cmd_report(out):
    arms = dict(baseline_2k=results(DEC / 'main/baseline'), learned_10k=results(out / 'learned_10k'),
                teacher=results(out / 'teacher'), intervention_2k=results(DEC / 'main/intervention'))
    rep = dict(timestamps=json.loads((out / 'timestamps.json').read_text()),
               statuses={k: dict(collections.Counter(v.get(i, {}).get('status', 'missing') for i in IDS)) for k, v in arms.items()},
               wins={k: sum(v[i]['fight']['won'] for i in IDS if v.get(i, {}).get('status') == 'completed') for k, v in arms.items()},
               errors={k: [dict(fight_id=i, status=v[i]['status'], error=v[i].get('error')) for i in IDS
                           if i in v and v[i]['status'] != 'completed'] for k, v in arms.items()})
    rep['paired'] = {f'{b}-minus-{a}': paired(arms[a], arms[b], a, b) for a, b in
                     [('baseline_2k', 'learned_10k'), ('baseline_2k', 'teacher'), ('learned_10k', 'teacher'),
                      ('intervention_2k', 'teacher')]}
    rep['timing'] = {k: timing(v) for k, v in arms.items()}
    # Budget flips and confidence of losses (descriptive; root values are search estimates in win-points).
    b2, b10 = arms['baseline_2k'], arms['learned_10k']
    try:
        raw = raw_start_values()
    except Exception as e:  # noqa: BLE001
        raw = {}; rep['raw_value_error'] = repr(e)
    rows = []
    for i in IDS:
        if not all(a.get(i, {}).get('status') == 'completed' for a in (b2, b10, arms['teacher'])): continue
        rows.append(dict(fight_id=i, won_2k=b2[i]['fight']['won'], won_10k=b10[i]['fight']['won'],
                         won_teacher=arms['teacher'][i]['fight']['won'], raw_start=raw.get(i),
                         root_2k=first_root(b2, i), root_10k=first_root(b10, i),
                         max_root_2k=max(s['root_value'] for s in b2[i]['search']),
                         max_root_10k=max(s['root_value'] for s in b10[i]['search'])))
    (out / 'per_fight.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    buckets = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r['raw_start'] is None: continue
        b = '<50' if r['raw_start'] < 50 else '50-70' if r['raw_start'] < 70 else '70-85' if r['raw_start'] < 85 else '>=85'
        buckets[b]['n'] += 1; buckets[b]['won_2k'] += r['won_2k']; buckets[b]['won_10k'] += r['won_10k']; buckets[b]['won_teacher'] += r['won_teacher']
    losses2k = [r for r in rows if not r['won_2k']]
    rep['calibration_descriptive'] = dict(
        by_raw_start_value={k: dict(v) for k, v in sorted(buckets.items())},
        mean_raw_start=statistics.mean(r['raw_start'] for r in rows if r['raw_start'] is not None) if raw else None,
        mean_first_root_2k=statistics.mean(r['root_2k'] for r in rows), mean_first_root_10k=statistics.mean(r['root_10k'] for r in rows),
        losses_2k=len(losses2k),
        losses_2k_max_root_ge_80=sum(r['max_root_2k'] >= 80 for r in losses2k),
        losses_2k_max_root_ge_80_won_10k=sum(r['max_root_2k'] >= 80 and r['won_10k'] for r in losses2k),
        losses_2k_won_by_teacher=sum(r['won_teacher'] for r in losses2k),
        note='Aggregate calibration does not establish causal value error.')
    (out / 'report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['setup', 'play', 'report'])
    ap.add_argument('--out', type=Path, default=C.DATE / f'id={EXP}/out')
    a = ap.parse_args(); (a.out / 'logs').mkdir(parents=True, exist_ok=True)
    dict(setup=cmd_setup, play=cmd_play, report=cmd_report)[a.stage](a.out)


if __name__ == '__main__':
    main()
