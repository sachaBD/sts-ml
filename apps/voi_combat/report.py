"""Summarize complete paired fights; resample decks within each boss."""
import json
import sys

import numpy as np


def summarize(rows):
    ids = [r['fight_id'] for r in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('expected distinct, nonempty fights')
    prices = [a['price'] for a in rows[0]['arms']]
    if prices[0] is not None or any([a['price'] for a in r['arms']] != prices for r in rows):
        raise ValueError('unpaired operating points')
    wins = np.array([[a['win'] for a in r['arms']] for r in rows], dtype=int)
    boss = np.array([r['encounter'] for r in rows])
    rng = np.random.default_rng(120)
    counts = np.zeros((4000, len(prices)))
    for k in np.unique(boss):
        w = wins[boss == k]
        counts += w[rng.integers(len(w), size=(len(counts), len(w)))].sum(axis=1)
    delta = counts - counts[:,[0]]
    for j, p in enumerate(prices):
        arms = [r['arms'][j] for r in rows]
        yield dict(price=p, fights=len(ids), wins=int(wins[:,j].sum()),
                   win_rate=wins[:,j].mean(), gain=(wins[:,j] - wins[:,0]).mean(),
                   paired_95=np.quantile(delta[:,j]/len(ids), [.025,.975]).tolist(),
                   mean_requests=np.mean([a['requests'] for a in arms]),
                   mean_pairs=np.mean([a['probes'] for a in arms]), capped=sum(a['capped'] for a in arms))


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        for result in summarize([json.loads(line) for line in f]):
            print(json.dumps(result))
