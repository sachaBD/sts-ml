"""Progress of run_panache.sh: stage, update N/total and its current step, timing/ETA, learning curve.

  watch -n 30 .venv/bin/python experiments/25-10-26/a-gumbel/status.py
"""
import json
import subprocess
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
D = REPO / 'runs/schema=combat_v4/date=2026-10-25'
RUNS = [('gumbel v1 (stopped)', 'a-gumbel-panache-v1'), ('gumbel v2', 'a-gumbel-panache-v2')]
TEST = D / 'id=a-gumbel-panache-test-v1/out'
TEST_ARMS = ['gumbel_v1_u06_g256', 'gumbel_v2_u06_g256']
LOG = REPO / 'experiments/25-10-26/a-gumbel/logs/run_panache.log'


def lines(p):
    return sum(1 for _ in p.open()) if p.exists() else 0


def step(u):
    """Current step inside update directory u."""
    if (u / 'monitor/results.jsonl').exists():
        return f'monitor {lines(u / "monitor/results.jsonl")}/50'
    if (u / 'model/model.onnx').exists():
        return 'monitor starting'
    if (u / 'train.log').exists():
        epochs = sum('"epoch"' in l for l in (u / 'train.log').open())
        return f'train epoch {epochs + 1}/3'
    if (u / 'play/rows').exists():
        return 'encode'
    return f'play {lines(u / "play/results.jsonl")}/{BATCH}'


def run_status(stage, rid):
    out = D / f'id={rid}/out'
    if not out.exists():
        return print(f'{stage:18s} {rid}: not started')
    global BATCH
    cfg = tomllib.loads((out / 'config.toml').read_text())['run']
    total, BATCH = cfg['updates'], cfg['batch_fights']
    ups = sorted(out.glob('update[0-9][0-9][0-9]'))
    curve = json.loads((out / 'curve.json').read_text()) if (out / 'curve.json').exists() else []
    done = len(curve) - 1 if curve else 0
    t0 = (out / 'bootstrap/model').stat().st_mtime if (out / 'bootstrap/model').exists() else None
    finished = (out / 'summary.json').exists() or rid.endswith('panache-v1') and 'gumbel' in rid
    now = time.time()
    if finished:
        head = 'DONE'
    else:
        head = f'update {len(ups)}/{total}: {step(ups[-1])}' if ups else 'bootstrap'
    print(f'{stage:18s} {rid}: {head}')
    stamps = [(u / 'monitor/results.jsonl').stat().st_mtime for u in ups if (u / 'monitor/results.jsonl').exists()
              and int(u.name[-3:]) <= done]
    if len(stamps) >= 2 and not finished:
        recent = stamps[-4:]
        per = (recent[-1] - recent[0]) / (len(recent) - 1)
        print(f'{"":18s} last updates {per / 60:.1f} min each; ETA this stage ~{(total - done) * per / 60:.0f} min '
              f'(finish ~{time.strftime("%H:%M", time.localtime(now + (total - done) * per))}); '
              f'elapsed {(now - t0) / 60:.0f} min')
    if curve:
        ref = curve[0]['decks']['panache']['teacher_wins']
        print(f'{"":18s} monitor (same 50 seeds each update, greedy; MCTS20k {ref}/50):')
        row = '  '.join(f'u{c["update"]}:{c["decks"]["panache"]["wins"]}' for c in curve)
        print(f'{"":18s} {row}')
        coll = []
        for u in ups:
            p = u / 'play/results.jsonl'
            if p.exists() and lines(p) == BATCH:
                rs = [json.loads(l) for l in p.open()]
                coll.append(f'u{int(u.name[-3:])}:{sum(r["status"] == "completed" and r["fight"]["won"] for r in rs)}')
        print(f'{"":18s} collection wins/{BATCH} (exploring): {"  ".join(coll)}')


def main():
    print(time.strftime('%H:%M:%S'), '| run_panache.sh:', (LOG.read_text().strip().splitlines() or ['?'])[-1][:110])
    for stage, rid in RUNS:
        run_status(stage, rid)
    if TEST.exists():
        parts = []
        for a in TEST_ARMS:
            n = lines(TEST / a / 'results.jsonl')
            parts.append(f'{a} {n}/300' if n else f'{a} -')
        print(f'{"2/2 tests":18s} ' + ', '.join(parts))
        if (TEST / 'report.json').exists():
            r = json.loads((TEST / 'report.json').read_text())
            print(f'{"":18s} stored: ten-deck PUCT2k {r["stored"]["ten"]}/300, MCTS20k {r["stored"]["mcts"]}/300')
            for k, v in r['arms'].items():
                lo, hi = v['vs_ten_puct2k']['ci95_pp']
                print(f'{"":18s} {k}: {v["wins"]}/300, vs PUCT2k {v["vs_ten_puct2k"]["gap_pp"]:+.1f} pp [{lo:+.1f}, {hi:+.1f}]')
    else:
        print(f'{"2/2 tests":18s} not started')
    workers = subprocess.run(['pgrep', '-c', '-f', 'pv_worker play'], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'], capture_output=True,
                         text=True).stdout.strip()
    print(f'pv_worker processes: {workers or 0} | GPU: {gpu}')


if __name__ == '__main__':
    main()
