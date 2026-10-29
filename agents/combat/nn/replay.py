"""Disk-indexed, deck-balanced replay. Only the current minibatch is read and padded."""
import json
import os
from pathlib import Path
import struct

import numpy as np
import torch

from infra.workers.reader import Reader
from .model import WIDTH

HEADER = struct.Struct('<iif')


def index(path):
    """Native writes each game's rows and result under the same output lock."""
    cache = path.with_suffix('.index.npz')
    if cache.exists():
        with np.load(cache) as stored:
            return {name: stored[name] for name in stored.files}
    raw = np.memmap(path, dtype=np.uint8, mode='r')
    groups, at = {}, 0
    for line in (path.parent / 'results.jsonl').read_text().splitlines():
        game = json.loads(line)
        deck = game['fight_id'].split(':')[-2]  # namespace:iteration:deck:index
        offsets = groups.setdefault(deck, [])
        for _ in range(game['decisions']):
            if at + HEADER.size > len(raw):
                raise ValueError(f'{path}: truncated sample header')
            count, actions, won = HEADER.unpack_from(raw, at)
            if not 0 < actions <= count or won != float(game['won']):
                raise ValueError(f'{path}: sample/result mismatch')
            offsets.append(at)
            at += HEADER.size + 4 * count * WIDTH + 4 * actions
            if at > len(raw):
                raise ValueError(f'{path}: truncated sample body')
    if at != len(raw):
        raise ValueError(f'{path}: samples do not match completed game results')
    groups = {k: np.asarray(v, dtype=np.uint64) for k, v in groups.items() if v}
    temporary = cache.with_name('.' + cache.name)
    np.savez(temporary, **groups)
    temporary.replace(cache)
    return groups


class Replay:
    """Rows are read by parallel exact-size preads (infra Reader); close() (or `with`) releases the files."""

    def __init__(self, paths, max_rows=0):
        paths, groups = [Path(p) for p in paths], {}
        for file, path in enumerate(paths):
            decks = index(path)
            if not decks:
                continue
            # Records are contiguous, so each one ends where the next begins (or at end of file).
            starts = np.sort(np.concatenate(list(decks.values())))
            ends = np.append(starts[1:], np.uint64(os.path.getsize(path)))
            for name, offsets in decks.items():
                sizes = ends[np.searchsorted(starts, offsets)] - offsets
                groups.setdefault(name, []).append(
                    np.column_stack((np.full(len(offsets), file, dtype=np.uint64), offsets, sizes)))
        self.groups = {name: np.concatenate(parts) for name, parts in sorted(groups.items())}
        if not self.groups:
            raise ValueError('no training decisions in replay')
        self.available = {name: len(rows) for name, rows in self.groups.items()}
        if max_rows:
            quota = max_rows // len(self.groups)
            if quota < 1:
                raise ValueError('max-train-rows must allow at least one row per deck')
            for name, rows in self.groups.items():
                if len(rows) > quota:
                    self.groups[name] = rows[torch.randperm(len(rows))[:quota].numpy()]
        self.selected = {name: len(rows) for name, rows in self.groups.items()}
        self.drawn = {name: 0 for name in self.groups}
        self.reader = Reader(paths)

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __len__(self):
        return sum(self.selected.values())

    def sample(self, size):
        """Equal deck quotas (difference <=1), then uniform rows within each deck."""
        n = len(self.groups)
        if size < n:
            raise ValueError('batch must be at least the number of decks')
        extra = set(torch.randperm(n)[:size % n].tolist())
        picked = []
        for i, (name, rows) in enumerate(self.groups.items()):
            take = size // n + (i in extra)
            self.drawn[name] += take
            picked.append(rows[torch.randint(len(rows), (take,)).numpy()])
        records = []
        for raw in self.reader.read(np.concatenate(picked).tolist()):
            count, actions, won = HEADER.unpack_from(raw)
            tokens = np.frombuffer(raw, dtype=np.float32, count=count * WIDTH, offset=HEADER.size).reshape(count, WIDTH)
            policy = np.frombuffer(raw, dtype=np.float32, count=actions, offset=HEADER.size + count * WIDTH * 4)
            records.append((tokens, policy, won))
        return records

    @staticmethod
    def collate(records):
        size = len(records)
        longest = max(len(t) for t, _, _ in records)
        tokens = np.zeros((size, longest, WIDTH), dtype=np.float32)
        policy = np.zeros((size, longest), dtype=np.float32)
        won = np.empty(size, dtype=np.float32)
        for i, (t, p, w) in enumerate(records):
            tokens[i, :len(t)] = t
            policy[i, len(t) - len(p):len(t)] = p
            won[i] = w
        return torch.from_numpy(tokens), torch.from_numpy(policy), torch.from_numpy(won)

    def batch(self, size):
        return self.collate(self.sample(size))

    def batches(self, size, attention_cells=2048 * 64 * 64):
        """Keep B*T*T bounded; rare long states never pad the entire effective batch.

        All sampled rows are retained. The trainer accumulates weighted gradients
        across these microbatches before one optimizer step.
        """
        records = sorted(self.sample(size), key=lambda r: len(r[0]))
        chunk = []
        for record in records:
            length = len(record[0])
            if length * length > attention_cells:
                raise ValueError(f'single state of {length} tokens exceeds attention budget')
            if chunk and (len(chunk) + 1) * length * length > attention_cells:
                yield self.collate(chunk)
                chunk = []
            chunk.append(record)
        if chunk:
            yield self.collate(chunk)
