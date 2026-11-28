"""Paired gym benchmark: prices share matchups and per-measurement random streams."""
import argparse
import json
from statistics import mean, stdev

from .session import Session


def benchmark(n, budget, price, adaptive):
    gains, costs, values = [], [], []
    for seed in range(n):
        s = Session(seed, budget, price, adaptive=adaptive)
        while s.step() is not None:
            pass
        values.append(s.grade())
        gains.append(values[-1]['search'] - values[-1]['no_search'])
        costs.append(s.spent)
    gain = mean(gains)
    half = 1.96*stdev(gains)/n**.5
    return dict(seeds=n, budget=budget, price=price, adaptive=adaptive, mean_cost=mean(costs),
                **{k: mean(v[k] for v in values) for k in values[0]},
                paired_gain=gain, paired_normal_95=[gain-half, gain+half])


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--seeds', type=int, default=128)
    p.add_argument('--budget', type=int, default=16)
    p.add_argument('--prices', type=float, nargs='+', default=[.005, .001])
    p.add_argument('--fixed-scale', action='store_true')
    args = p.parse_args()
    if args.seeds < 2:
        p.error('at least two seeds needed for a paired interval')
    for price in args.prices:
        print(json.dumps(benchmark(args.seeds, args.budget, price, not args.fixed_scale)), flush=True)
