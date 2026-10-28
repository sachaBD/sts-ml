"""Run progress and recent results; --json provides the same data for agents/web UIs."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import tomllib

from apps.combat_expert_iteration.expert_iteration import deck_of


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


def score(reference, results):
    return dict(n=len(reference), wins=sum(results[k] for k in reference), teacher_wins=sum(reference.values()))


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
    reference_path = out / 'inputs/reference.json'
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        for it in sorted({r['test'] for r in rows if 'test' in r})[-3:]:
            results = {r['fight_id']: r['won'] for r in jsonl(training / f'test/iter{it:03d}/results.jsonl')}
            if set(results) != set(reference):
                continue
            decks = {name: score({k: v for k, v in reference.items() if deck_of(k) == name}, results)
                     for name in sorted({deck_of(k) for k in reference})}
            evaluations.append(dict(iteration=it, all=score(reference, results), decks=decks))
    return dict(version=1, state=state, config=config, progress=progress,
                progress_age_seconds=time.time() - progress['time'] if progress else None,
                last_completed_iteration=latest,
                checkpoint=str(checkpoint) if checkpoint and checkpoint.exists() else None,
                updates=[r for r in completed if r['iteration'] > 0][-10:], evaluations=evaluations,
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
    print('\nEvaluation: fixed monitoring seeds; historical teacher rules differ. One training run.')
    for row in data['evaluations']:
        s = row['all']
        print(f'iter{row["iteration"]:03d}: {s["wins"]}/{s["n"]}; teacher {s["teacher_wins"]}/{s["n"]}')
        for name, s in row['decks'].items():
            print(f'  {name:20} {s["wins"]:3}/{s["n"]} (teacher {s["teacher_wins"]})')
    print(f'\nCheckpoint: {data["checkpoint"]}\nLogs: {data["logs"]}')


if __name__ == '__main__':
    main()
