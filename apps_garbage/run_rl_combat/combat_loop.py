"""Bounded human-deck combat expert iteration; reuses human_champ play and PV encode/train.

Run under runs.run combat_v4 --no-compact. See COMBAT.md. No overworld code or native
search changes. All source descendants share an explicit train/val membership.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from apps.human_champ import bench

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def fingerprint(row):
    """Group deck/relic families regardless of battle seed, HP and cyclic counters."""
    s = row['start']
    family = dict(deck=sorted((c['id'], c['upgraded'], c['misc']) for c in s['deck']),
                  relics=sorted(r['id'] for r in s['relics']), ascension=s['ascension'], encounter=s['encounter'])
    return hashlib.sha256(json.dumps(family, sort_keys=True).encode()).hexdigest()


def partition(rows, heldout, val_fraction=.1, sample_seed=0):
    excluded_ids = {r['deck_id'] for r in heldout}
    excluded_families = {fingerprint(r) for r in heldout}
    # One historical-seed start per human source deck: no gameplay augmentation in v1.
    pool, seen = [], set()
    for row in rows:
        if row['seed_kind'] != 'human' or row['deck_id'] in seen:
            continue
        seen.add(row['deck_id'])
        if row['deck_id'] in excluded_ids or fingerprint(row) in excluded_families:
            continue
        pool.append(row)
    families = sorted({fingerprint(r) for r in pool},
                      key=lambda key: hashlib.sha256(f'{sample_seed}:{key}'.encode()).hexdigest())
    if len(families) < 2:
        raise ValueError('need at least two independent training/validation deck families')
    nval = min(len(families) - 1, max(1, round(len(families) * val_fraction)))
    val = set(families[:nval])
    mapping = {r['fight_id']: 'val' if fingerprint(r) in val else 'train' for r in pool}
    return pool, mapping


def read_results(directory):
    path = Path(directory) / 'results.jsonl'
    return {r['fight_id']: r for r in map(json.loads, path.read_text().splitlines())}


def comparison(starts, baseline, candidate):
    groups = collections.defaultdict(list)
    for r in starts:
        groups[r['deck_id']].append(r)
    a, b = read_results(baseline), read_results(candidate)
    pairs = []
    for pid, rows in groups.items():
        if any(d.get(r['fight_id'], {}).get('status') != 'completed' for d in (a, b) for r in rows):
            continue
        av = sum(a[r['fight_id']]['fight']['won'] for r in rows) / len(rows)
        bv = sum(b[r['fight_id']]['fight']['won'] for r in rows) / len(rows)
        pairs.append((fingerprint(rows[0]), av, bv))
    if not pairs:
        raise ValueError('no completed paired decks')
    # Cluster SE by deck family. Each family contributes its summed paired difference.
    n = len(pairs)
    amean, bmean = (sum(r[k] for r in pairs) / n for k in (1, 2))
    diff = bmean - amean
    clusters = collections.defaultdict(float)
    for family, av, bv in pairs:
        clusters[family] += bv - av - diff
    g = len(clusters)
    se = math.sqrt(g / (g - 1) * sum(x*x for x in clusters.values()) / n**2) if g > 1 else None
    return dict(selected_decks=len(groups), paired_decks=n, families=g, baseline_win_rate=amean,
                candidate_win_rate=bmean, difference=diff, se=se,
                baseline_statuses=dict(collections.Counter(r['status'] for r in a.values())),
                candidate_statuses=dict(collections.Counter(r['status'] for r in b.values())))


def choose_teacher(rows, results, limit):
    # Full teacher recovery trajectories, NOT teacher annotation of learner intermediate states.
    # Keep some ordinary coverage as well as low-performing learner starts.
    completed = [r for r in rows if results.get(r['fight_id'], {}).get('status') == 'completed']
    rank = lambda r: hashlib.sha256(r['fight_id'].encode()).hexdigest()
    completed.sort(key=rank)
    coverage = completed[:min(len(completed), limit // 2)]
    used = {r['fight_id'] for r in coverage}
    hard = [r for r in completed if r['fight_id'] not in used and not results[r['fight_id']]['fight']['won']]
    rest = [r for r in completed if r['fight_id'] not in used and results[r['fight_id']]['fight']['won']]
    return [dict(r, fight_id=r['fight_id'] + ':teacher') for r in (coverage + hard + rest)[:limit]]


def write_starts(rows, path):
    pq.write_table(pa.Table.from_pylist(rows, bench.load('human_champ_bench_v1').STARTS), path, compression='zstd')


def active_play_workers():
    count = 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / 'cmdline').read_bytes().split(b'\0')
            if argv and b'pv_worker' in argv[0] and len(argv) > 1 and argv[1] in (b'play', b'teacher'):
                count += 1
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return count


def wait_for_cpu(out, max_wait=7200):
    start = time.monotonic()
    while active_play_workers():
        if time.monotonic() - start > max_wait:
            raise RuntimeError('other PV gameplay still running after resource wait; retry later')
        print('Waiting for other PV gameplay (not stopping it)', flush=True)
        time.sleep(15)


def stage(out, name, command, cpu=False):
    marker = out / (name + '.done')
    if marker.exists():
        return
    if cpu:
        wait_for_cpu(out)
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / (name + '.log')
    print(f'{datetime.now(timezone.utc).isoformat()} {name}: runtime unknown; may exceed one minute; log={log}', flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('a') as stream:
        subprocess.run([str(c) for c in command], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    marker.touch()


def encode(out, name, source, worker):
    # Use canonical replay and the existing feature-cache contract, not new encoders.
    stage(out, name, [PY, '-m', 'agents.combat.pv.data', '--fights', source / 'fights-0.parquet',
                     '--search', source / 'search-0.parquet', '--encounters', 39, '--worker', worker,
                     '--out', out / name])
    return out / name / 'rows.parquet'


def play(out, name, starts, model, worker, sims, workers, explore=False, teacher=False):
    command = [PY, '-m', 'apps.human_champ.bench', 'play', '--starts', starts, '--agent',
               'teacher' if teacher else 'pv', '--sims', sims, '--workers', workers, '--worker', worker,
               '--out', out / name]
    if not teacher:
        command += ['--model', model / 'model.onnx']
    if explore:
        command += ['--explore', '--sample-turns']
    stage(out, name, command, cpu=True)
    return out / name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='additional megacrit pull run directories')
    p.add_argument('--regression', type=Path, required=True, help='existing benchmark excluded from training')
    p.add_argument('--init', type=Path, required=True, help='directory containing model.pt/onnx/external data')
    p.add_argument('--worker', type=Path, default=ROOT / 'build/pv/agents/combat/pv/pv_worker')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--sims', type=int, default=2000)
    p.add_argument('--teacher-sims', type=int, default=20000)
    p.add_argument('--teacher-decks', type=int, default=128, help='maximum recovery/coverage teacher trajectories per round')
    p.add_argument('--max-train-decks', type=int, default=800)
    p.add_argument('--regression-decks', type=int, default=100, help='fixed deterministic subset, not training data')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--device', default='cuda')
    p.add_argument('--val-fraction', type=float, default=.1)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--window', type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.workers <= 8 or min(a.rounds, a.sims, a.teacher_sims, a.epochs, a.max_train_decks,
                                    a.regression_decks, a.window) < 1 or a.teacher_decks < 0:
        p.error('positive budgets, nonnegative teacher-decks, and 1–8 workers required')
    if not 0 < a.val_fraction < 1:
        p.error('val-fraction must be in (0,1)')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {k: [str(x.resolve()) for x in v] if isinstance(v, list) else str(v.resolve()) if isinstance(v, Path) else v
              for k, v in vars(a).items()}
    config.update(worker_sha=sha(a.worker), model_sha=sha(a.init / 'model.pt'), onnx_sha=sha(a.init / 'model.onnx'),
                  onnx_data_sha=sha(a.init / 'model.onnx.data') if (a.init / 'model.onnx.data').exists() else None,
                  regression_sha=sha(a.regression))
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('resume configuration differs; use a fresh output directory')
    config_path.write_text(json.dumps(config, indent=2))
    worker = out / 'frozen/pv_worker'
    if not worker.exists():
        worker.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.worker, worker)
    if sha(worker) != config['worker_sha']:
        raise ValueError('frozen worker checksum mismatch')
    incumbent = out / 'frozen/model'
    if not incumbent.exists():
        shutil.copytree(a.init, incumbent)
    if sha(incumbent / 'model.pt') != config['model_sha'] or sha(incumbent / 'model.onnx') != config['onnx_sha']:
        raise ValueError('frozen initial model checksum mismatch')
    if config['onnx_data_sha'] is not None and sha(incumbent / 'model.onnx.data') != config['onnx_data_sha']:
        raise ValueError('frozen ONNX external data checksum mismatch')
    stage(out, 'prepare', [PY, '-m', 'apps.human_champ.bench', 'prepare', '--runs', *a.runs,
                          '--limit', 100000, '--sample-seed', a.sample_seed, '--out', out / 'source'])
    original = pq.read_table(out / 'source/starts.parquet').to_pylist()
    regression_all = pq.read_table(a.regression).to_pylist()
    pool, membership = partition(original, regression_all, a.val_fraction, a.sample_seed)
    train = sorted([r for r in pool if membership[r['fight_id']] == 'train'],
                   key=lambda r: hashlib.sha256(f'{a.sample_seed}:{r["deck_id"]}'.encode()).hexdigest())
    val = sorted([r for r in pool if membership[r['fight_id']] == 'val'], key=lambda r: r['deck_id'])
    train = train[:a.max_train_decks * a.rounds]
    planned_rounds = min(a.rounds, math.ceil(len(train) / a.max_train_decks))
    if not train or not val:
        raise ValueError('empty train or validation cohort')
    # Deterministic deck sample, not a seed sample: retain both regression seeds.
    regression_ids = sorted({r['deck_id'] for r in regression_all},
                            key=lambda pid: hashlib.sha256(f'{a.sample_seed}:{pid}'.encode()).hexdigest())[:a.regression_decks]
    regression = [r for r in regression_all if r['deck_id'] in set(regression_ids)]
    write_starts(train, out / 'train.parquet')
    write_starts(val, out / 'val.parquet')
    write_starts(regression, out / 'regression.parquet')
    split_summary = dict(source_decks=len(original)//2, independent_pool_decks=len(pool), train_decks=len(train),
                         validation_decks=len(val), regression_decks=len(regression_ids),
                         excluded_decks=len(original)//2-len(pool), augmentation='none')
    (out / 'split-summary.json').write_text(json.dumps(split_summary, indent=2))
    print(json.dumps(split_summary), flush=True)
    mapping = {r['fight_id']: membership[r['fight_id']] for r in train + val}
    split_path = out / 'split.json'
    baseline_val = play(out, 'baseline-val', out / 'val.parquet', incumbent, worker, a.sims, a.workers)
    baseline_regression = play(out, 'baseline-regression', out / 'regression.parquet', incumbent, worker, a.sims, a.workers)
    validation_rows = encode(out, 'validation-rows', baseline_val, worker)
    train_shards = []
    curves = []
    for iteration in range(planned_rounds):
        current = out / f'iter{iteration:03d}'
        current.mkdir(exist_ok=True)
        batch = train[iteration * a.max_train_decks:(iteration + 1) * a.max_train_decks]
        write_starts(batch, current / 'train-starts.parquet')
        learner = play(current, 'collect', current / 'train-starts.parquet', incumbent, worker, a.sims, a.workers, explore=True)
        shards = [encode(current, 'learner-rows', learner, worker)]
        teacher_rows = choose_teacher(batch, read_results(learner), a.teacher_decks)
        if teacher_rows:
            teacher_path = current / 'teacher-starts.parquet'
            write_starts(teacher_rows, teacher_path)
            mapping.update({r['fight_id']: 'train' for r in teacher_rows})
            teacher = play(current, 'teacher', teacher_path, incumbent, worker, a.teacher_sims, a.workers, teacher=True)
            shards.append(encode(current, 'teacher-rows', teacher, worker))
        train_shards.append(shards)
        split_path.write_text(json.dumps(mapping, indent=2))
        window = [s for group in train_shards[-a.window:] for s in group]
        stage(current, 'train', [PY, '-m', 'agents.combat.pv.train', '--data', *window, validation_rows,
                               '--split-manifest', split_path, '--flat-policy-weighting', '--init', incumbent / 'model.pt',
                               '--epochs', a.epochs, '--lr', a.lr, '--grad-clip', 1, '--device', a.device,
                               '--stream', '--out', current / 'model'])
        candidate = current / 'model'
        cv = play(current, 'eval-val', out / 'val.parquet', candidate, worker, a.sims, a.workers)
        cr = play(current, 'eval-regression', out / 'regression.parquet', candidate, worker, a.sims, a.workers)
        v = comparison(val, baseline_val, cv)
        r = comparison(regression, baseline_regression, cr)
        enough = v['paired_decks'] >= max(20, math.ceil(.95 * len(val))) and r['paired_decks'] >= math.ceil(.95 * len(regression_ids))
        promote = (enough and v['se'] is not None and v['difference'] > 1.96 * v['se']
                   and r['se'] is not None and r['difference'] >= -1.96 * r['se'])
        result = dict(iteration=iteration, validation=v, regression=r, promoted=promote,
                      candidate=str(candidate), incumbent=str(incumbent))
        curves.append(result)
        (current / 'comparison.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if promote:
            incumbent, baseline_val, baseline_regression = candidate, cv, cr
    summary = dict(**split_summary, rounds=curves, selected_model=str(incumbent))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Human-deck combat expert iteration', '', json.dumps(split_summary), '',
             'No gameplay augmentation; D5 initialization; action-level public-belief search.',
             'Rates are paired deck means; uncertainty is 1 SE clustered by deck family. Promotion is a conservative heuristic, not a multiple-testing-adjusted guarantee.', '']
    for curve in curves:
        lines += [f'## Round {curve["iteration"]}', '', '| set | paired decks | incumbent | candidate | difference ± SE |', '|---|---:|---:|---:|---:|']
        for name in ('validation', 'regression'):
            x = curve[name]
            se = f'{100*x["se"]:.1f}' if x['se'] is not None else 'unknown'
            lines.append(f'| {name} | {x["paired_decks"]} | {100*x["baseline_win_rate"]:.1f}% | {100*x["candidate_win_rate"]:.1f}% | {100*x["difference"]:+.1f} ± {se} pts |')
        lines += ['', f'Promoted: {curve["promoted"]}.', '']
    lines += [f'Selected model: `{incumbent}`.', '', 'Historical balance, no-potion, reconstruction and reset-counter caveats from human_champ v1 still apply.',
              'Validation outcome labels come from frozen baseline play; training losses are diagnostics, not an oracle evaluation of optimal win probability.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
