"""Run progress and recent results; --json provides the same data for agents/web UIs."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import tomllib

from .reporting import baselines, deck_of, jsonl, outcomes, score


def running(training):
    if not (training / '.lock').exists():
        return False
    with (training / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


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
    references = baselines(out)
    tested = sorted({r['test'] for r in rows if 'test' in r})
    if 0 in tested:
        references['initial'] = outcomes(training / 'test/iter000/results.jsonl')
        references['initial'].update(outcomes(training / 'test/iter000/in-sample/results.jsonl'))
    cohort_path = out / 'inputs/cohorts.json'
    cohorts = json.loads(cohort_path.read_text()) if cohort_path.exists() else {}
    for it in tested[-3:]:
        results = outcomes(training / f'test/iter{it:03d}/results.jsonl')
        if not results:
            continue
        combined = dict(results)
        combined.update(outcomes(training / f'test/iter{it:03d}/in-sample/results.jsonl'))
        decks = {name: score({k: v for k, v in combined.items() if deck_of(k) == name}, references)
                 for name in sorted({deck_of(k) for k in combined})}
        groups = {group: score({k: v for k, v in combined.items() if cohorts.get(deck_of(k)) == group}, references)
                  for group in sorted(set(cohorts.values())) if any(cohorts.get(deck_of(k)) == group for k in combined)}
        evaluations.append(dict(iteration=it, all=score(results, references), decks=decks, cohorts=groups))
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
    r, model, training = config['run'], config['model'], config['training']
    print(f'{r["id"]}   {time.strftime("%H:%M:%S")}   {data["state"].upper()}')
    decks = len(json.loads((out / 'inputs/starts.json').read_text()))
    print(f'{decks} decks; {r["games_per_deck"] * decks:,} games/update; '
          f'{config["search"]["sims"]} sims ({config["evaluation"]["sims"]} evaluation); '
          f'NN {model["dim"]}x{model["layers"]}; {training["steps"]} train steps x {training["batch"]} rows')
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
