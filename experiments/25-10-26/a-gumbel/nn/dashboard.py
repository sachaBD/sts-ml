"""Dashboard for nn-gumbel-panache-v1 (RUN.md).

  watch -n 10 .venv/bin/python experiments/25-10-26/a-gumbel/nn/dashboard.py [RUN_DIR]
"""
import json
import math
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel-panache-v1'
STORED = REPO / 'runs/schema=combat_v4/date=2026-10-09/id=champ-ten-test-u6-v1/out'
REFERENCES = (('2k-sim network', 'ten', 174), ('MCTS20k', 'mcts', 124))


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def paired(ours, theirs):
    ids = [i for i in ours if i in theirs]
    d = [int(ours[i]) - int(theirs[i]) for i in ids]
    if len(d) < 2:
        return 'insufficient matching seeds'
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return f'{100 * m:+.1f} pp [{100 * (m - 1.96 * se):+.1f}, {100 * (m + 1.96 * se):+.1f}]'


def worker_metrics(directory):
    path = directory / 'worker.log'
    text = path.read_text(errors='replace') if path.exists() else ''
    def number(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None
    stats = re.findall(r'gpu_stats: (\{[^\n]+\})', text)
    exact = {}
    for raw in reversed(stats):
        try:
            exact = json.loads(raw)
            break
        except json.JSONDecodeError:
            continue  # a live log read may catch the last line halfway through a write
    return dict(setup=number(r'timing: evaluator setup ([\d.]+) s'),
                gameplay=number(r'timing: gameplay ([\d.]+) s'),
                leaves=number(r'gpu: \d+ batches, mean ([\d.]+) leaves'),
                fill=exact['leaves'] / exact['capacity'] if exact.get('capacity') else None,
                busy=number(r'leaves, ([\d.]+) s busy'))


def main():
    if not (RUN / 'status.json').exists():
        return print(f'{RUN.name}: not started')
    args = json.loads((RUN / 'args.json').read_text())
    status = json.loads((RUN / 'status.json').read_text())
    log = jsonl(RUN / 'log.jsonl')
    iterations = [r for r in log if 'iteration' in r and r['iteration'] > 0]
    total = int(args['iterations'])
    now = time.time()

    print(f'{RUN.name}   {time.strftime("%H:%M:%S")}')
    print(f'network {args.get("dim", 64)} wide / {args.get("layers", 2)} layers / {args.get("heads", 2)} heads; '
          f'{args["sims"]} sims; {args["games"]} games/update; {args["workers"]} CPU workers; '
          f'GPU batch {args.get("gpu_batch", 1024)} × {args.get("gpu_tokens", 48)} tokens')
    progress = f'{status.get("done")}/{status.get("total")}' if status.get('total') else status.get('detail', '')
    print(f'now: {status["stage"]} — {status["phase"]} {progress}   (updated {now - status["time"]:.0f} s ago)')
    if iterations:
        timing_rows = [r for r in iterations if r['games'] == int(args['games'])] or iterations
        timing_rows = timing_rows[-3:]
        per = (sum(r['play_seconds'] for r in timing_rows) / sum(r['games'] for r in timing_rows) * int(args['games'])
               + sum(r['train_seconds'] for r in timing_rows) / len(timing_rows))
        left = (total - len(iterations)) * per
        print(f'iterations done {len(iterations)}/{total}; projected {per / 60:.1f} min/update at current game count; '
              f'ETA ~{left / 60:.0f} min (+ tests), finish ~{time.strftime("%H:%M", time.localtime(now + left))}')
    print()
    print('iter  self-play wins (exploring)   states   play s   train s   gameplay g/h   GPU fill')
    for r in log:
        if r.get('iteration', -1) == 0:
            print(f'   0  bootstrap (MCTS games)        {r["states"]:>6}        -   {r["train_seconds"]:>7}')
        elif 'iteration' in r:
            metrics = worker_metrics(RUN / f'iter{r["iteration"]:03d}')
            seconds = metrics['gameplay']
            rate = f'{r["games"] * 3600 / seconds:,.0f}' if seconds else '—'
            fill_fraction = metrics['fill']
            if fill_fraction is None and metrics['leaves'] is not None:
                fill_fraction = metrics['leaves'] / int(args.get('gpu_batch', 1024))
            fill = f'{100 * fill_fraction:.0f}%' if fill_fraction is not None else '—'
            print(f'{r["iteration"]:>4}  {r["wins"]:>4}/{r["games"]:<4} ({100 * r["wins"] / r["games"]:.0f}%)'
                  f'           {r["states"]:>6}   {r["play_seconds"]:>6}   {r["train_seconds"]:>7}'
                  f'   {rate:>12}   {fill:>8}')
    print('play s includes engine setup; gameplay g/h excludes setup; GPU fill = real leaves / computed row capacity.')
    print('Timing/strength varies with policy and game length; individual updates are not independent replicates.')
    print()
    print(f'test: 300 fixed panache seeds, greedy Gumbel {args["sims"]} (stored: 2k-sim network 174/300, MCTS20k 124/300)')
    print('Differences: paired 95% normal-approximation CIs, conditional on one trained network per arm.')
    stored = {k: {r['fight_id']: r['status'] == 'completed' and r['fight']['won'] for r in jsonl(STORED / k / 'results.jsonl')}
              for _, k, _ in REFERENCES}
    for d in sorted((RUN / 'test').glob('iter*')) if (RUN / 'test').exists() else []:
        ours = {r['fight_id']: r['won'] for r in jsonl(d / 'results.jsonl')}
        if len(ours) < 300:
            print(f'  {d.name}: running {len(ours)}/300'); continue
        print(f'  {d.name}: {sum(ours.values())}/300   ' +
              '   '.join(f'vs {name} {paired(ours, stored[k])}' for name, k, _ in REFERENCES))
        previous = REPO / 'runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel-panache-v1/test' / d.name / 'results.jsonl'
        if previous.exists() and previous.resolve() != (d / 'results.jsonl').resolve():
            baseline = {r['fight_id']: r['won'] for r in jsonl(previous)}
            print(f'    vs previous NN256 ({sum(baseline.values())}/{len(baseline)}): {paired(ours, baseline)}')


if __name__ == '__main__':
    main()
