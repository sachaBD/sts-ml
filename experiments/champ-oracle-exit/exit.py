#!/usr/bin/env python3
"""Resumable Champ oracle expert iteration. Starts → oracle play → encode → train → three benches.

Example: exit.py --tag a --init runs/.../iteration-0 --rounds 3 --n 3000 --sims 800 --workers 9
Outputs: runs/schema=combat_v4/date=YYYY-MM-DD/id=champ-ox-TAG-rNN/{starts,play,rows,model,...}.
Each stage has a log, PID file and success marker. Failed/incomplete stages are rerun; complete stages are skipped.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / '.venv/bin/python'
RUNBOOK = ROOT / 'experiments/champ-oracle-exit/RUNBOOK.md'
BENCH = ROOT / 'runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet'
TEACHER = ROOT / 'runs/schema=combat_v4/date=2026-10-03/id=champ-bench-teacher20k/out'
SOURCE = 'runs/schema=overworld_v1/date=2026-10-0*/id=ah-*/out/combat/fights-*.parquet'


def log(message):
    with RUNBOOK.open('a') as f:
        f.write(f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {message}\n')


def stage(run, name, command):
    marker = run / f'{name}.done'
    if marker.exists():
        return
    output = run / f'{name}.log'
    pid = run / f'{name}.pid'
    if pid.exists():
        previous = int(pid.read_text())
        try:
            os.kill(previous, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'{name}: PID {previous} still exists; verify it before resuming')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    print(f'{run.name} {name}: runtime unknown (may exceed one minute); log={output}', flush=True)
    with output.open('w') as stream:
        proc = subprocess.Popen([str(x) for x in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
        pid.write_text(str(proc.pid))
        code = proc.wait()
    pid.unlink(missing_ok=True)
    if code:
        log(f'FAIL {run.name} {name} exit={code} log={output}')
        raise subprocess.CalledProcessError(code, command)
    marker.touch()
    log(f'DONE {run.name} {name} log={output}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True)
    p.add_argument('--init', type=Path, required=True, help='new-contract model directory with model.pt/model.onnx')
    p.add_argument('--rounds', type=int, required=True)
    p.add_argument('--first-round', type=int, default=1)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--sims', type=int, required=True)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--real-every', type=int, default=1, help='run real bench and compare every K rounds')
    p.add_argument('--train-args', default='--epochs 2')
    p.add_argument('--selfplay-flags', default='', help='shlex-split extra flags for oracle self-play')
    p.add_argument('--oracle-bench-flags', default='', help='shlex-split extra flags for oracle bench')
    p.add_argument('--real-flags', default='', help='shlex-split extra flags for real bench (base sims remains 2000)')
    p.add_argument('--worker', type=Path, default=ROOT / 'build/pv/agents/combat/pv/pv_worker')
    p.add_argument('--date', default=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    # Accept quoted single-option values too: argparse otherwise treats "--turn-search" as an option.
    argv = []
    words = iter(sys.argv[1:])
    for word in words:
        if word in {'--selfplay-flags', '--oracle-bench-flags', '--real-flags'}:
            value = next(words, None)
            if value is None:
                p.error(f'{word} requires a flag string')
            word += '=' + value
        argv.append(word)
    a = p.parse_args(argv)
    try:
        selfplay_flags = shlex.split(a.selfplay_flags)
        oracle_bench_flags = shlex.split(a.oracle_bench_flags)
        real_flags = shlex.split(a.real_flags)
    except ValueError as error:
        p.error(f'invalid play flags: {error}')
    if not 1 <= a.workers <= 9 or min(a.n, a.sims, a.rounds, a.first_round) < 1:
        p.error('workers must be 1–9; n, sims, rounds and first-round must be positive')
    if a.real_every < 1:
        p.error('real-every must be positive')
    if a.n > 1_000_000:
        p.error('n exceeds the per-iteration seed block')
    base = ROOT / f'runs/schema=combat_v4/date={a.date}'
    run_for = lambda r: base / f'id=champ-ox-{a.tag}-r{r:02d}'
    # One immutable snapshot per driver launch, shared by its sequential stages.
    frozen = ROOT / 'build/frozen' / f'pv_worker.ox-{a.tag}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.worker, frozen)
    model = a.init.resolve()
    if a.first_round > 1:
        model = run_for(a.first_round - 1) / 'model'
        if not (model / 'model.pt').exists():
            p.error('previous iteration model missing; resume from an earlier round')
    for r in range(a.first_round, a.first_round + a.rounds):
        run = run_for(r)
        run.mkdir(parents=True, exist_ok=True)
        starts = run / 'starts.parquet'
        # Stable disjoint million-seed blocks across iterations (and practical tag separation).
        seed0 = 995_000_000_000 + (zlib.crc32(a.tag.encode()) % 10000) * 1_000_000_000 + r * 1_000_000
        stage(run, 'starts', [PY, 'apps/pv/starts.py', 'generate', '--source', SOURCE, '--encounter', 39,
                             '--n', a.n, '--seed0', seed0, '--exclude-seed-min', 981_000_000_000,
                             '--exclude-seed-max', 982_000_000_000, '--rng', seed0, '--out', starts])

        def play(name, source, sims, flags):
            stage(run, name, [PY, 'apps/pv/play.py', '--starts', source, '--agent', 'pv', '--model', model / 'model.onnx',
                              '--sims', sims, '--workers', a.workers, '--worker', frozen, '--out', run / name, *flags])

        play('play', starts, a.sims, ['--oracle', '--explore', '--sample-turns', *selfplay_flags])
        stage(run, 'encode', [PY, '-m', 'agents.combat.pv.data', '--fights', *sorted((run / 'play').glob('fights-*.parquet')),
                             '--search', *sorted((run / 'play').glob('search-*.parquet')), '--encounters', 39,
                             '--worker', frozen, '--out', run / 'rows'])
        rows = [run_for(k) / 'rows/rows.parquet' for k in range(max(1, r - 2), r + 1)]
        if not all(path.exists() for path in rows):
            raise RuntimeError(f'replay window missing rows: {rows}')
        stage(run, 'train', [PY, '-m', 'agents.combat.pv.train', '--data', *rows, '--out', run / 'model',
                            '--init', model / 'model.pt', '--device', 'cuda', '--stream', *shlex.split(a.train_args)])
        model = run / 'model'
        play('bench-oracle', BENCH, a.sims, ['--oracle', *oracle_bench_flags])
        if r % a.real_every == 0:
            play('bench-real', BENCH, 2000, real_flags)
        play('bench-policy', BENCH, 1, ['--policy-only'])
        if r % a.real_every == 0:
            stage(run, 'compare', [PY, 'apps/pv/compare.py', TEACHER, run / 'bench-real'])


if __name__ == '__main__':
    main()
