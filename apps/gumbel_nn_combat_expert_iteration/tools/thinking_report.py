"""Bin held-out thinking predictions by round; bootstrap whole decks (including their fights)."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from agents.combat.nn.model import ModelSpec, THINKING_ROUNDS, build_model
from agents.combat.nn.training.replay import Replay, TARGETS, index


def checkpoint_targets(probe):
    target = np.full(THINKING_ROUNDS, np.nan, dtype=np.float32)
    for i, r in enumerate(probe['rounds']):
        if r['prior'] == r['selected']:
            target[i] = 0
        elif 'gain' in r:
            if r['gain'] is not None:
                target[i] = r['gain']
        elif r['prior_q'] is not None and r['selected_q'] is not None:
            target[i] = r['selected_q'] - r['prior_q']
    return target


def bins(rows, rng, groups=4, resamples=2000):
    if not rows:
        return []
    # Keep tied predictions together; constant predictions make one bin.
    predicted = np.array([p for p, *_ in rows])
    edges = np.quantile(predicted, np.arange(1, groups) / groups)
    edges = np.unique(edges[(edges >= predicted.min()) & (edges < predicted.max())])
    assignments = np.searchsorted(edges, predicted, side='left')
    out = []
    for i in np.unique(assignments):
        selected = [row for row, group in zip(rows, assignments) if group == i]
        p, target, decks, fights = zip(*selected)
        counts = Counter(decks)
        totals = Counter()
        for deck, y in zip(decks, target):
            totals[deck] += y
        interval = None
        if len(counts) >= 2:
            n = np.array(list(counts.values()))
            y = np.array([totals[d] for d in counts])
            chosen = rng.integers(len(counts), size=(resamples, len(counts)))
            interval = np.quantile(y[chosen].sum(1) / n[chosen].sum(1), [.025, .975]).tolist()
        out.append(dict(predicted_range=[min(p), max(p)], predicted_mean=float(np.mean(p)),
                        teacher_mean=float(np.mean(target)), rows=len(p), fights=len(set(fights)),
                        decks=len(counts), deck_cluster_bootstrap_95_interval=interval))
    return out


@torch.inference_mode()
def report(model, replay):
    model.eval()
    rounds = [[] for _ in range(THINKING_ROUNDS)]
    example = None
    for deck, fight, records, probes in replay.probe_fights():
        x, _, y = replay.collate(records)
        targets = y[:, len(TARGETS):].numpy()
        np.testing.assert_array_equal(targets, np.stack([checkpoint_targets(probe) for probe in probes]))
        value, policy, thinking = model.predict(x)
        for i, probe in enumerate(probes):
            mask = np.isfinite(targets[i])
            for r in np.flatnonzero(mask):
                rounds[r].append((float(thinking[i, r]), float(targets[i, r]), deck, fight))
            if example is None and mask.any():
                action_logits = policy[i, len(records[i][0]) - len(records[i][1]):len(records[i][0])]
                example = dict(fight=fight, **probe, target=[float(t) if keep else None for t, keep in zip(targets[i], mask)],
                               mask=mask.tolist(), predictions=thinking[i].tolist(), value=float(value[i]),
                               action_logits=action_logits.tolist())
    if example is None:
        raise ValueError('no labelled probe rounds in held-out replay')
    rng = np.random.default_rng(229)
    return dict(metric='Shared-particle out-of-sample greedy-prior plunge gain (teacher proxy, not realized win gain)',
                rounds=[bins(rows, rng) for rows in rounds], example=example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--train', type=Path, nargs='+', required=True, help='sample files used to fit the checkpoint')
    parser.add_argument('--heldout', type=Path, nargs='+', required=True, help='sample files from disjoint decks')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    spec = ModelSpec(**{k: v for k, v in checkpoint['spec'].items() if k != 'encoding_version'})
    net = build_model(spec)
    net.load_state_dict(checkpoint['net'])
    train_decks = {deck for path in args.train for deck in index(path)}
    with Replay(args.heldout, encoding=net.encoding) as replay:
        if train_decks.intersection(replay.groups):
            parser.error('training and held-out replays share decks')
        print(json.dumps(report(net, replay), indent=2, allow_nan=False))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
