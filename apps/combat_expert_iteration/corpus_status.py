"""Read-only dashboard for the four-round corpus pilot."""
import argparse
import json
from pathlib import Path
import time
import os


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def render(run):
    out = run / 'out'
    status = read(out / 'status.json')
    lines = [f'CORPUS PILOT: {run.name}',
             f'Status: {status.get("status", "preparing implementation / manifests")}',
             f'Stage: {status.get("stage", "preflight")}',
             'Budget: 4 rounds; 5,500 collection + 1,600 development games; final set untouched.']
    if status.get('updated'):
        lines.append(f'Last progress: {time.time()-status["updated"]:.0f}s ago')
    if status.get('pid'):
        try:
            os.kill(status['pid'], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        lines.append(f'Controller PID {status["pid"]}: {"alive" if alive else "not running"}')
    progress = status.get('progress')
    if progress:
        lines.append(f'Games: {progress.get("done", 0)}/{progress.get("total", "?")} | wins {progress.get("wins", 0)} | statuses {progress.get("statuses", {})}')
    if status.get('error'):
        lines.append('ERROR: ' + status['error'])
    lines.append('\nCompleted stages:')
    for p in sorted(out.glob('*/done.json')):
        meta = read(p)
        lines.append(f'  {p.parent.name}: {meta.get("count", "?")} games')
    for p in sorted(out.glob('round*/model/complete.json')):
        meta = read(p)
        lines.append(f'  {p.parent.parent.name} training: {meta.get("steps")} steps, ONNX error {meta.get("onnx_max_error")}')
    curve = read(out / 'curve.json')
    if curve:
        lines.append('\nDevelopment (same 20 families ×20 seeds; descriptive):')
        for row in curve.get('checks', []):
            lo, hi = row['family_ci95']
            lines.append(f'  round {row["round"]}: {row["wins"]}/{row["n"]}; MCTS {row["teacher_wins"]}/{row["n"]}; gap {100*row["gap"]:+.1f}pp, family 95% [{100*lo:+.1f}, {100*hi:+.1f}]')
    log = run / 'logs' / 'controller.log'
    lines.append(f'\nLog: {log}')
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-2500))
            tail = stream.read().decode(errors='replace').splitlines()[-6:]
        lines.extend(tail)
    for p in sorted(out.glob('round*/train.log'))[-1:]:
        with p.open('rb') as stream:
            stream.seek(max(0, p.stat().st_size-2000))
            tail = stream.read().decode(errors='replace').splitlines()[-2:]
        lines.append('\nLatest training:')
        lines.extend(tail)
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(render(parser.parse_args().run))
