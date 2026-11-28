"""Play paired full fights. Only the measurement executor knows the simulator or the network."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from itertools import islice
from pathlib import Path
import subprocess

import numpy as np

from agents.combat.player.play import gpu_environment
from agents.combat.search.voi_gym.model import Belief, Measurement, buy, rank_measurements


class CombatDecision:
    def __init__(self, root):
        logits = np.array(root['logits'])
        self.actions = np.argsort(-logits, kind='stable')[:4]
        v = root['value']
        means = v + .03*(logits[self.actions] - logits[self.actions[0]])
        self.reaches = np.eye(len(means))  # choose a root action, then fixed continuation
        prior_sd = min(.06, v, 1-v)
        self.belief = Belief(means, self.reaches * prior_sd**2)
        self.probes = [Measurement(str(i), self.reaches[i]-self.reaches[0], .08**2, 1)
                       for i in range(1, len(means))]

    def select(self, price, remaining):
        if not remaining:
            return None
        key = buy(rank_measurements(self.belief, self.reaches, self.probes, price, remaining))
        return None if key is None else self.probes[int(key)-1]

    def pair(self, m):
        return [int(self.actions[int(m.name)]), int(self.actions[0])]

    def move(self):
        return int(self.actions[self.belief.mean.argmax()])


def exchange(proc, command):
    proc.stdin.write(json.dumps(command) + '\n')
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f'bridge exited ({proc.poll()})')
    return json.loads(line)


def play(proc, start, price, cap):
    state = exchange(proc, start)
    decisions = switches = probes = probe_requests = 0
    while 'win' not in state:
        d = CombatDecision(state)
        remaining = cap if price is not None else 0
        while (m := d.select(price, remaining)) is not None:
            result = exchange(proc, {'pair': d.pair(m)})
            d.belief.observe(m, result['difference'])
            probes += 1
            probe_requests += result['requests']
            remaining -= 1
        move = d.move()
        switches += int(move != d.actions[0])
        decisions += 1
        state = exchange(proc, {'play': move})
    return dict(price=price, decisions=decisions, switches=switches, probes=probes,
                probe_requests=probe_requests, **state)


def shard(task):
    args, starts = task
    results = []
    env = gpu_environment() if args.gpu else None
    if args.gpu and Path('/usr/lib/wsl/lib').is_dir():
        env['LD_LIBRARY_PATH'] += ':/usr/lib/wsl/lib'
    command = [args.bridge, args.model] + (['--gpu'] if args.gpu else [])
    with subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        try:
            for row in starts:
                arms = [play(proc, row['start'], p, args.cap) for p in [None, *args.prices]]
                results.append(dict(fight_id=row['fight_id'], encounter=row['start']['encounter'], arms=arms))
        finally:
            proc.stdin.close()
        if proc.wait():
            raise RuntimeError('bridge failed')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['bridge', 'model', 'starts', 'out']:
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--prices', nargs='+', type=float, default=[.015, .006])
    parser.add_argument('--cap', type=int, default=3, help='max paired probes per decision')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', help='TensorRT with CPU overflow; use the full checkpoint')
    args = parser.parse_args()
    if args.gpu and args.jobs != 1:
        parser.error('use one GPU bridge; independent workers would duplicate the engines')
    if args.cap < 0 or min(args.prices) < 0 or args.jobs < 1:
        parser.error('invalid budget, price or jobs')
    with args.starts.open() as f:
        rows = [json.loads(x) for x in islice(f, args.limit)]
    if not rows or len({r['fight_id'] for r in rows}) != len(rows):
        parser.error('expected distinct, nonempty fight starts')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out, ProcessPoolExecutor(args.jobs) as pool:
        chunk = len(rows) if args.gpu else 8
        for batch in pool.map(shard, ((args, rows[i:i+chunk]) for i in range(0,len(rows),chunk))):
            for result in batch:
                print(json.dumps(result), file=out)
            out.flush()


if __name__ == '__main__':
    main()
