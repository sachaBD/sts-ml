"""Aggregate the newest experiment's latest N self-play iterations.

    .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.deck_stats 3
"""
import argparse
import json

import pyarrow as pa

from runs.run import RUNS


def iteration_counts(training, latest: int, recent_most: int):
    """Per-deck counts over an explicit window ending at latest (zero is initialization)."""
    if recent_most < 1 or latest < 0:
        raise ValueError('positive recent_most and nonnegative latest required')
    before = json.loads((training / f'iter{max(0, latest - recent_most):03d}' / 'decks.json').read_text())
    after = json.loads((training / f'iter{latest:03d}' / 'decks.json').read_text())
    return {deck: dict(attempts=stats['fights'] - before[deck]['fights'],
                       wins=stats['wins'] - before[deck]['wins'])
            for deck, stats in after.items()}


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
            counts = iteration_counts(training, latest, recent_most)
    return pa.table({
        'deck_id': pa.array(sorted(counts), type=pa.string()),
        'attempts': pa.array([counts[d]['attempts'] for d in sorted(counts)], type=pa.int64()),
        'wins': pa.array([counts[d]['wins'] for d in sorted(counts)], type=pa.int64()),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recent_most', type=int)
    args = parser.parse_args()
    print(newest_runs(args.recent_most))
