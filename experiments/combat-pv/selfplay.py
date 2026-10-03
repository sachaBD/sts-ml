#!/usr/bin/env python3
"""Champ self-play loop (resumable; each stage skipped when its output exists).

  selfplay.py --tag sp1 --init MODEL_DIR --rounds 6 --n 3000 --sims 800 --eval-sims 2000 --workers 8 [--window 3]
              [--extra-rows rows.parquet ...] [--train-args "--epochs 4"]

Round r: generate n Champ starts (apps/pv/starts.py, fresh seeds 992e9 + 1e6·(round id)) → PV self-play with root
noise (play.py --explore) using the previous model → encode → train from the previous checkpoint on the last
`window` rounds' rows (+ extra rows) → greedy bench on the 453 held-out Champ starts. Results: RUNBOOK.md lines.
Runs one CPU stage at a time; `workers` play processes (+1 core for the trainer when training).
"""
import argparse
import json
import shlex
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / '.venv/bin/python')
FIGHTS = 'runs/schema=overworld_v1/date=2026-10-0*/id=ah-*/out/combat/fights-*.parquet'
BENCH = 'runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet'
RUNBOOK = ROOT / 'experiments/combat-pv/RUNBOOK.md'


def log(msg):
    with RUNBOOK.open('a') as f:
        f.write(f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}\n')


def sh(argv, **kw):
    subprocess.run([str(x) for x in argv], cwd=ROOT, check=True, **kw)


def find(schema, id_):
    hits = sorted(ROOT.glob(f'runs/schema={schema}/date=*/id={id_}'))
    return hits[-1] if hits else None


def done(run):
    return run is not None and json.loads((run / 'run.json').read_text()).get('status') == 'done'


def play(id_, starts, model, sims, workers, worker, explore):
    run = find('combat_v4', id_)
    if done(run):
        return run / 'out'
    log(f'START play {id_} model={model} sims={sims} explore={explore}')
    argv = [PY, '-m', 'runs.run', 'combat_v4', id_, '--no-compact', '--', PY, 'apps/pv/play.py', '--starts', starts,
            '--agent', 'pv', '--model', model, '--sims', sims, '--workers', workers, '--worker', worker, '--out', '{out}']
    sh(argv + (['--explore'] if explore else []), stdout=subprocess.DEVNULL)
    run = find('combat_v4', id_)
    log(f'DONE play {id_} {(run / "out/summary.json").read_text().strip()[:400]}'.replace('\n', ' '))
    return run / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--init', type=Path, required=True, help='model dir with model.pt / model.onnx')
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--first-round', type=int, default=1)
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--sims', type=int, default=800)
    ap.add_argument('--eval-sims', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--window', type=int, default=3)
    ap.add_argument('--extra-rows', nargs='*', default=[])
    ap.add_argument('--train-args', default='--epochs 4')
    ap.add_argument('--worker', default='build/frozen/pv_worker')
    a = ap.parse_args()
    model = a.init
    rows = []
    for r in range(a.first_round, a.first_round + a.rounds):
        rid = f'champ-{a.tag}-r{r:02d}'
        # 1. starts
        sdir = ROOT / f'runs/schema=pv_starts_v1/id={rid}'
        starts = sdir / 'starts.parquet'
        if not starts.exists():
            sdir.mkdir(parents=True, exist_ok=True)
            seed0 = 992_000_000_000 + 1_000_000 * (zlib.crc32(a.tag.encode()) % 1000) + 10_000 * r
            sh([PY, 'apps/pv/starts.py', 'generate', '--source', FIGHTS, '--encounter', 39, '--n', a.n,
                '--seed0', seed0, '--exclude-seed-min', 981_000_000_000, '--exclude-seed-max', 982_000_000_000,
                '--rng', seed0, '--out', starts])
        # 2. self-play
        sp = play(f'{rid}-play', starts, model / 'model.onnx', a.sims, a.workers, a.worker, True)
        # 3. encode
        rdir = ROOT / f'runs/schema=pv_rows_v1/id={rid}'
        if not (rdir / 'rows.parquet').exists():
            sh([PY, '-m', 'agents.combat.pv.data', '--fights', *sorted(sp.glob('fights-*.parquet')),
                '--search', *sorted(sp.glob('search-*.parquet')), '--encounters', 39, '--out', rdir])
        rows.append(rdir / 'rows.parquet')
        # 4. train
        mdir = find('pv_model_v1', rid)
        if mdir is None or not (mdir / 'out/model.onnx').exists():
            data = [*rows[-a.window:], *a.extra_rows]
            sh(['experiments/combat-pv/train.sh', rid, *data, '--', '--init', model / 'model.pt',
                *shlex.split(a.train_args)])
            mdir = find('pv_model_v1', rid)
        model = mdir / 'out'
        # 5. bench
        bench = play(f'{rid}-bench', BENCH, model / 'model.onnx', a.eval_sims, a.workers + 1, a.worker, False)
        teacher = find('combat_v4', 'champ-bench-teacher20k')
        out = subprocess.run([PY, 'apps/pv/compare.py', ROOT / 'runs/schema=combat_v4/date=2026-10-04/id=champ-bench/out',
                              *([teacher / 'out'] if teacher else []), bench], cwd=ROOT, capture_output=True, text=True, env={'PYTHONPATH': str(ROOT)})
        (bench.parent / 'compare.md').write_text(out.stdout + out.stderr)
        log(f'BENCH {rid}: {bench.parent / "compare.md"}')


if __name__ == '__main__':
    main()
