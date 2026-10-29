"""Run progress and recent results; --json provides the same data for agents/web UIs."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time
import tomllib

from apps.combat_expert_iteration.expert_iteration import deck_of, wilson


def jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # live writer may be midway through its last line
    return rows


def running(training):
    if not (training / '.lock').exists():
        return False
    with (training / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def mean_bounds(values):
    mean = sum(values) / len(values)
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) * (len(values) - 1))) if len(values) > 1 else None
    return mean, [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None


def score(results, baselines):
    n, wins = len(results), sum(results.values())
    comparisons = {}
    for name, reference in baselines.items():
        keys = results.keys() & reference.keys()
        if not keys:
            continue
        differences = [int(results[k]) - int(reference[k]) for k in keys]
        m = len(keys)
        gap = sum(differences) / m
        se = math.sqrt(sum((d - gap) ** 2 for d in differences) / (m * (m - 1))) if m > 1 else None
        grouped = {}
        for k in keys:
            grouped.setdefault(deck_of(k), []).append(int(results[k]) - int(reference[k]))
        macro, cluster_ci = mean_bounds([sum(v) / len(v) for v in grouped.values()])
        comparisons[name] = dict(n=m, wins=sum(results[k] for k in keys), baseline_wins=sum(reference[k] for k in keys),
                                 gap=gap, gap_95=[gap - 1.96 * se, gap + 1.96 * se] if se is not None else None,
                                 families=len(grouped), macro_gap=macro, family_gap_95=cluster_ci)
    grouped = {}
    for k, won in results.items():
        grouped.setdefault(deck_of(k), []).append(won)
    macro, cluster_ci = mean_bounds([sum(v) / len(v) for v in grouped.values()])
    return dict(n=n, wins=wins, win_rate=wins / n, wilson_95=wilson(wins, n), comparisons=comparisons,
                families=len(grouped), macro_win_rate=macro, family_95=cluster_ci)


def snapshot(out):
    training = out / 'training'
    config = tomllib.loads((out / 'config.toml').read_text())
    path = training / 'status.json'
    progress = json.loads(path.read_text()) if path.exists() else {}
    record_path = out.parent / 'run.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    if running(training):
        state = 'stopping' if (training / 'STOP').exists() else 'running'
    elif progress.get('phase') in {'stopped', 'done'}:
        state = progress['phase']
    else:
        state = 'interrupted' if progress or record.get('status') == 'failed' else 'preparing'
    recovery_path = out / 'recovery.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    if state == 'interrupted' and recovery and time.time() - recovery['time'] < 60:
        command = Path(f'/proc/{recovery.get("supervisor_pid", 0)}/cmdline')
        try:
            if b'gumbel_nn_combat_expert_iteration' in command.read_bytes():
                state = 'recovering'
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    rows = jsonl(training / 'log.jsonl')
    completed = [r for r in rows if 'iteration' in r]
    latest = max((r['iteration'] for r in completed), default=None)
    checkpoint = training / f'iter{latest:03d}/model.pt' if latest is not None else None
    evaluations = []
    baseline_path = out / 'inputs/baselines.json'
    baselines = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    reference_path = out / 'inputs/reference.json'
    if not baselines and reference_path.exists():
        baselines['mcts20k'] = json.loads(reference_path.read_text())
    tested = sorted({r['test'] for r in rows if 'test' in r})
    if 0 in tested:
        baselines['initial'] = {r['fight_id']: r['won'] for r in jsonl(training / 'test/iter000/results.jsonl')}
        baselines['initial'].update({r['fight_id']: r['won'] for r in jsonl(training / 'test/iter000/in-sample/results.jsonl')})
    cohort_path = out / 'inputs/cohorts.json'
    cohorts = json.loads(cohort_path.read_text()) if cohort_path.exists() else {}
    for it in tested[-3:]:
        results = {r['fight_id']: r['won'] for r in jsonl(training / f'test/iter{it:03d}/results.jsonl')}
        if not results:
            continue
        combined = dict(results)
        combined.update({r['fight_id']: r['won'] for r in jsonl(training / f'test/iter{it:03d}/in-sample/results.jsonl')})
        decks = {name: score({k: v for k, v in combined.items() if deck_of(k) == name}, baselines)
                 for name in sorted({deck_of(k) for k in combined})}
        groups = {group: score({k: v for k, v in combined.items() if cohorts.get(deck_of(k)) == group}, baselines)
                  for group in sorted(set(cohorts.values())) if any(cohorts.get(deck_of(k)) == group for k in combined)}
        evaluations.append(dict(iteration=it, all=score(results, baselines), decks=decks, cohorts=groups))
    return dict(version=1, state=state, config=config, progress=progress, recovery=recovery,
                progress_age_seconds=time.time() - progress['time'] if progress else None,
                last_completed_iteration=latest,
                checkpoint=str(checkpoint) if checkpoint and checkpoint.exists() else None,
                updates=[r for r in completed if r['iteration'] > 0][-10:], evaluations=evaluations,
                validation_wall_fraction=sum(r['seconds'] for r in rows if 'test' in r) /
                    max(1, sum(r.get('play_seconds', 0) + r.get('train_seconds', 0) + r.get('seconds', 0) for r in rows)),
                disk_free_bytes=shutil.disk_usage(out).free, logs=str(out.parent / 'logs'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    out = a.run / 'out' if (a.run / 'out').is_dir() else a.run
    if not (out / 'config.toml').exists():
        record = out.parent / 'run.json'
        failed = record.exists() and json.loads(record.read_text()).get('status') == 'failed'
        state = 'interrupted' if failed else 'preparing'
        print(json.dumps(dict(state=state)) if a.json else state)
        return
    data = snapshot(out.resolve())
    if a.json:
        print(json.dumps(data))
        return
    config, status = data['config'], data['progress']
    r, settings = config['run'], config['loop']
    print(f'{r["id"]}   {time.strftime("%H:%M:%S")}   {data["state"].upper()}')
    print(f'{len(config["deck"])} decks; {r["games_per_deck"] * len(config["deck"]):,} games/update; '
          f'{settings["sims"]} sims; NN {settings["dim"]}x{settings["layers"]}; '
          f'{settings["steps"]} train steps x {settings["batch"]} rows')
    if status:
        progress = f'{status["done"]}/{status["total"]}' if 'total' in status else status.get('detail', '')
        print(f'{"Now" if data["state"] in {"running", "stopping"} else "Last progress"}: '
              f'{status["stage"]} / {status["phase"]} {progress} (updated {data["progress_age_seconds"]:.0f}s ago)')
    if data['state'] == 'stopping': print('Finishing current update and scheduled evaluation before stopping.')
    print(f'Last completed: {data["last_completed_iteration"]}; disk free: {data["disk_free_bytes"] / 1024**3:.1f} GiB')
    print('\niter  collection wins  selected rows   play s  train/export s')
    for row in data['updates']:
        print(f'{row["iteration"]:4}  {row["wins"]:5}/{row["games"]:<5}  {row["states"]:11,} '
              f'{row["play_seconds"]:8} {row["train_seconds"]:15}')
    print(f'\nValidation wall share: {100 * data["validation_wall_fraction"]:.1f}% (includes initial baseline)')
    print('Evaluation: family-clustered approximate 95% bounds; per-deck Wilson bounds. One training run, unadjusted.')
    print('MCTS20k/PUCT are historical matched-start references; policy names and simulator rules differ.')
    for row in data['evaluations']:
        s = row['all']
        ci = s['family_95'] or s['wilson_95']
        print(f'iter{row["iteration"]:03d}: {s["wins"]}/{s["n"]} = {100*s["win_rate"]:.1f}% '
              f'[{100*ci[0]:.1f}, {100*ci[1]:.1f}], {s["families"]} families')
        if row is not data['evaluations'][-1]:
            continue
        for name, s in [*row['cohorts'].items(), *row['decks'].items()]:
            ci = s['family_95'] if s['families'] > 1 else s['wilson_95']
            print(f'  {name:23} {s["wins"]:4}/{s["n"]} [{100*ci[0]:.1f}, {100*ci[1]:.1f}]%')
            if name in row['cohorts'] and 'initial' in s['comparisons']:
                c = s['comparisons']['initial']; bounds = c['family_gap_95']
                if bounds:
                    print(f'    vs initial: {100*c["macro_gap"]:+.1f} pp [{100*bounds[0]:+.1f}, {100*bounds[1]:+.1f}], '
                          f'family-clustered n={c["families"]}')
        for name, c in row['all']['comparisons'].items():
            ci = c['family_gap_95'] or c['gap_95']
            gap = c['macro_gap'] if c['family_gap_95'] else c['gap']
            print(f'  vs {name:12} {100*gap:+.1f} pp '
                  f'[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}], paired n={c["n"]}' if ci else f'  vs {name}: insufficient pairs')
    print(f'\nCheckpoint: {data["checkpoint"]}\nLogs: {data["logs"]}')


if __name__ == '__main__':
    main()
