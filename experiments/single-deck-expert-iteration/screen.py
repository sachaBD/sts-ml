"""Small fixed-loadout MCTS selection screen; no training or HP augmentation."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq

from apps.human_champ import bench


def wilson(wins, n):
    if not n:
        return None
    z = 1.959963984540054
    p = wins / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [center-half, center+half]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--deck-id', required=True)
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--sims', type=int, default=20000)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.n < 1 or a.sims < 1 or not 1 <= a.workers <= 10:
        p.error('positive n/sims and 1–10 workers required')
    source = [r for r in pq.read_table(a.source).to_pylist() if r['deck_id'] == a.deck_id]
    if not source:
        raise ValueError('deck not found')
    base = next(r for r in source if r['seed_kind'] == 'human')
    used = {r['start']['seed'] for r in source}
    rows = []
    for i in range(a.n):
        seed = int.from_bytes(hashlib.sha256(f'single-deck-selection-v1:{a.deck_id}:{i}'.encode()).digest()[:8], 'little')
        while seed in used:
            seed = (seed+1) & bench.MASK
        used.add(seed)
        row = copy.deepcopy(base)
        row.update(fight_id=f'single-deck-selection:{a.deck_id}:{i}', seed_kind='selection')
        row['start'].update(seed=seed, misc_rng=bench.rng_state(seed), potion_rng=bench.rng_state(seed))
        rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    starts = a.out / 'starts.parquet'
    pq.write_table(bench.pa.Table.from_pylist(rows, bench.load('human_champ_bench_v1').STARTS), starts, compression='zstd')
    print(f'MCTS{a.sims}: {a.n} fresh seeds, HP {base["start"]["hp"]}/{base["start"]["max_hp"]}, fixed loadout', flush=True)
    bench.play(SimpleNamespace(starts=starts, agent='teacher', sims=a.sims, workers=a.workers,
                               worker=a.worker, model=None, out=a.out / 'play', timeout=600, limit=None))
    summary = json.loads((a.out / 'play/summary.json').read_text())
    summary.update(deck_id=a.deck_id, hp=base['start']['hp'], max_hp=base['start']['max_hp'], simulations=a.sims,
                   confidence_95_wilson=wilson(summary['wins'], summary['completed']))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
