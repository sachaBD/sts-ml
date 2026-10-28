"""Bounded recovery for recognised CUDA failures. No configuration changes or silent game filtering."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from apps.common.app import write_json
from .dashboard import jsonl, running

CUDA_FAILURES = ('CUDA: unknown error', 'CUDA: unspecified launch failure',
                 'CUDA: an illegal memory access', 'CUDA failure', 'cudaErrorUnknown',
                 'CUDA error: unknown error', 'CUDA error: unspecified launch failure',
                 'CUDA error: an illegal memory access')


def resume_command(out, original):
    training = out / 'training'
    if running(training):
        raise RuntimeError('loop is still running; refusing overlap')
    if (training / 'STOP').exists():
        raise RuntimeError('STOP exists; remove it explicitly before continuing')
    # A killed Python parent may leave a native process alive. Never overlap it.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if Path(args[0]).name == 'nn_play' and any(str(training.resolve()) in a for a in args):
            raise RuntimeError(f'orphan native worker PID{path.parent.name}; inspect before recovery')
    completed = [r['iteration'] for r in jsonl(training / 'log.jsonl') if 'iteration' in r]
    if not completed:
        raise RuntimeError('no committed checkpoint to recover')
    latest = max(completed)
    if not (training / f'iter{latest:03d}/model.pt').exists():
        raise FileNotFoundError('committed checkpoint is missing')
    partial = training / f'iter{latest + 1:03d}'
    if partial.exists():
        archive = training / f'.interrupted-{time.time_ns()}-{partial.name}'
        partial.rename(archive)
        print(f'supervisor: preserved interrupted stage at {archive}', flush=True)
    return original + ['--resume-from', str(latest)]


def launch(command, out, retries=2):
    """Fresh CUDA process on failure; retry allowance is persisted across supervisor sessions."""
    out = Path(out).resolve()
    journal = out / 'recovery.jsonl'
    used = len(jsonl(journal))
    while True:
        errors = out.parent / 'logs' / f'learner-{time.time_ns()}.stderr.log'
        errors.parent.mkdir(parents=True, exist_ok=True)
        with errors.open('w') as stream:
            result = subprocess.run(command, stderr=stream)
        if result.returncode == 0:
            return
        status_path = out / 'training/status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        worker_log = Path(status.get('directory', str(out / 'training'))) / 'worker.log'
        text = worker_log.read_text(errors='replace') if worker_log.exists() else ''
        text += errors.read_text(errors='replace')
        print(f'supervisor: learner exited {result.returncode}; diagnostics: {errors}', flush=True)
        if (out / 'training/STOP').exists() or result.returncode != 1 or not any(s in text for s in CUDA_FAILURES) or used >= retries:
            raise subprocess.CalledProcessError(result.returncode, command)
        original = json.loads((out / 'command.json').read_text())
        command = resume_command(out, original)
        used += 1
        event = dict(time=time.time(), retry=used, supervisor_pid=os.getpid(), reason='recognised CUDA failure',
                     error=next((line[-2000:] for line in text.splitlines() if any(s in line for s in CUDA_FAILURES)), ''),
                     worker_log=str(worker_log), stderr=str(errors), resume_from=int(command[-1]), cooldown_seconds=15)
        with journal.open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        write_json(out / 'recovery.json', event)
        print('supervisor: ' + json.dumps(event), flush=True)
        time.sleep(15)
        if (out / 'training/STOP').exists():
            raise RuntimeError('stop requested during recovery; not restarting')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--retries', type=int, default=2)
    a = p.parse_args()
    if a.retries < 0:
        p.error('retries must be nonnegative')
    out = (a.run / 'out' if (a.run / 'out').is_dir() else a.run).resolve()
    if len(jsonl(out / 'recovery.jsonl')) >= a.retries and a.retries > 0:
        raise RuntimeError('automatic retry budget exhausted; inspect before explicitly increasing it')
    original = json.loads((out / 'command.json').read_text())
    command = resume_command(out, original)
    launch(command, out, retries=a.retries)


if __name__ == '__main__':
    main()
