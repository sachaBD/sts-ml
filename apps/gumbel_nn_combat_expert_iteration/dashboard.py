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
        comparisons[name] = dict(n=m, wins=sum(results[k] for k in keys), baseline_wins=sum(reference[k] for k in keys),
                                 gap=gap, gap_95=[gap - 1.96 * se, gap + 1.96 * se] if se is not None else None)
    return dict(n=n, wins=wins, win_rate=wins / n, wilson_95=wilson(wins, n), comparisons=comparisons)


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
    cohort_path = out / 'inputs/cohorts.json'
    cohorts = json.loads(cohort_path.read_text()) if cohort_path.exists() else {}
    for it in tested[-3:]:
        results = {r['fight_id']: r['won'] for r in jsonl(training / f'test/iter{it:03d}/results.jsonl')}
        if not results:
            continue
        decks = {name: score({k: v for k, v in results.items() if deck_of(k) == name}, baselines)
                 for name in sorted({deck_of(k) for k in results})}
        groups = {group: score({k: v for k, v in results.items() if cohorts.get(deck_of(k)) == group}, baselines)
                  for group in sorted(set(cohorts.values())) if any(cohorts.get(deck_of(k)) == group for k in results)}
        evaluations.append(dict(iteration=it, all=score(results, baselines), decks=decks, cohorts=groups))
    return dict(version=1, state=state, config=config, progress=progress,
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
    print('\niter  collection wins    loaded rows   play s  train/export s')
    for row in data['updates']:
        print(f'{row["iteration"]:4}  {row["wins"]:5}/{row["games"]:<5}  {row["states"]:11,} '
              f'{row["play_seconds"]:8} {row["train_seconds"]:15}')
    print(f'\nValidation wall share: {100 * data["validation_wall_fraction"]:.1f}% (includes initial baseline)')
    print('Evaluation: Wilson win-rate / paired-difference 95% bounds; conditional on one run.')
    print('MCTS20k/PUCT are historical, available only on original monitoring seeds; rules differ.')
    for row in data['evaluations']:
        s = row['all']
        print(f'iter{row["iteration"]:03d}: {s["wins"]}/{s["n"]} = {100*s["win_rate"]:.1f}% '
              f'[{100*s["wilson_95"][0]:.1f}, {100*s["wilson_95"][1]:.1f}]')
        if row is not data['evaluations'][-1]:
            continue
        for name, s in [*row['cohorts'].items(), *row['decks'].items()]:
            print(f'  {name:23} {s["wins"]:4}/{s["n"]} [{100*s["wilson_95"][0]:.1f}, {100*s["wilson_95"][1]:.1f}]%')
        for name, c in row['all']['comparisons'].items():
            ci = c['gap_95']
            print(f'  vs {name:12} {100*c["gap"]:+.1f} pp '
                  f'[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}], paired n={c["n"]}' if ci else f'  vs {name}: insufficient pairs')
    print(f'\nCheckpoint: {data["checkpoint"]}\nLogs: {data["logs"]}')


if __name__ == '__main__':
    main()
