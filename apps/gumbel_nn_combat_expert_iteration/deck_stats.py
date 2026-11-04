"""Aggregate the newest experiment's latest N self-play iterations.

    .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.deck_stats 3
"""
import argparse
import json

import pyarrow as pa

from runs.run import RUNS


def newest_runs(recent_most: int) -> pa.Table:
    """Sum N latest saved self-play iterations of the newest experiment (not scratch).

    Subtract cumulative snapshots; iteration zero is initialization, not self-play.
    Fewer than N available iterations means all available iterations.
    UUIDs or legacy deck names are preserved exactly as stored.
    """
    if recent_most < 1:
        raise ValueError('recent_most must be positive')
    runs = [(json.loads(p.read_text())['started'], p.parent)
            for p in RUNS.glob('schema=gumbel_nn_combat_v1/date=*/id=*/run.json')]
    counts = {}
    if runs:
        training = max(runs)[1] / 'out' / 'training'
        snapshots = {int(p.parent.name[4:]): p for p in training.glob('iter[0-9]*/decks.json')}
        if snapshots:
            latest = max(snapshots)
            baseline = max(0, latest - recent_most)
            before = json.loads(snapshots[baseline].read_text())
            after = json.loads(snapshots[latest].read_text())
            for deck, stats in after.items():
                old = before[deck]
                counts[deck] = stats['fights'] - old['fights'], stats['wins'] - old['wins']
    return pa.table({
        'deck_id': pa.array(sorted(counts), type=pa.string()),
        'attempts': pa.array([counts[d][0] for d in sorted(counts)], type=pa.int64()),
        'wins': pa.array([counts[d][1] for d in sorted(counts)], type=pa.int64()),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recent_most', type=int)
    args = parser.parse_args()
    print(newest_runs(args.recent_most))
