"""Dashboard for nn-gumbel-panache-v1 (RUN.md).

  watch -n 10 .venv/bin/python experiments/25-10-26/a-gumbel/nn/dashboard.py [RUN_DIR]
"""
import json
import math
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
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return f'{100 * m:+.1f} pp [{100 * (m - 1.96 * se):+.1f}, {100 * (m + 1.96 * se):+.1f}]'


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
    progress = f'{status.get("done")}/{status.get("total")}' if status.get('total') else status.get('detail', '')
    print(f'now: {status["stage"]} — {status["phase"]} {progress}   (updated {now - status["time"]:.0f} s ago)')
    if iterations:
        per = sum(r['play_seconds'] + r['train_seconds'] for r in iterations) / len(iterations)
        left = (total - len(iterations)) * per
        print(f'iterations done {len(iterations)}/{total}; {per / 60:.1f} min each; '
              f'ETA ~{left / 60:.0f} min (+ tests), finish ~{time.strftime("%H:%M", time.localtime(now + left))}')
    print()
    print('iter  self-play wins (exploring)   states   play s   train s')
    for r in log:
        if r.get('iteration', -1) == 0:
            print(f'   0  bootstrap (MCTS games)        {r["states"]:>6}        -   {r["train_seconds"]:>7}')
        elif 'iteration' in r:
            print(f'{r["iteration"]:>4}  {r["wins"]:>4}/{r["games"]:<4} ({100 * r["wins"] / r["games"]:.0f}%)'
                  f'           {r["states"]:>6}   {r["play_seconds"]:>6}   {r["train_seconds"]:>7}')
    print()
    print('test: 300 fixed panache seeds, greedy Gumbel 256 (stored: 2k-sim network 174/300, MCTS20k 124/300)')
    stored = {k: {r['fight_id']: r['status'] == 'completed' and r['fight']['won'] for r in jsonl(STORED / k / 'results.jsonl')}
              for _, k, _ in REFERENCES}
    for d in sorted((RUN / 'test').glob('iter*')) if (RUN / 'test').exists() else []:
        ours = {r['fight_id']: r['won'] for r in jsonl(d / 'results.jsonl')}
        if len(ours) < 300:
            print(f'  {d.name}: running {len(ours)}/300'); continue
        print(f'  {d.name}: {sum(ours.values())}/300   ' +
              '   '.join(f'vs {name} {paired(ours, stored[k])}' for name, k, _ in REFERENCES))


if __name__ == '__main__':
    main()
