"""Disk-indexed replay with uniform or deck-balanced sampling. Only the current minibatch is read and padded."""
import json
import os
import random
from pathlib import Path
import struct

import numpy as np
import torch

from infra.workers.reader import Reader

# nn_play sample record: token count, action count, token width, targets (targets.hpp order), tokens, policy.
SAMPLE_FORMAT = 3
TARGETS = ('won', 'hp', 'enemy_hp', 'hp_lost', 'enemy_hp_lost', 'turns_remaining')
HEADER = struct.Struct('<iii' + 'f' * len(TARGETS))


def index(path):
    """Native writes each game's rows and result under the same output lock."""
    cache = path.with_suffix('.index.npz')
    if cache.exists():
        with np.load(cache) as stored:
            return {name: stored[name] for name in stored.files}
    raw = np.memmap(path, dtype=np.uint8, mode='r') if path.stat().st_size else b''
    groups, at = {}, 0
    for line in (path.parent / 'results.jsonl').read_text().splitlines():
        game = json.loads(line)
        if game.get('sample_format') != SAMPLE_FORMAT:
            raise ValueError(f'{path}: sample format {game.get("sample_format")}, expected {SAMPLE_FORMAT}')
        deck = game['fight_id'].split(':')[-2]  # namespace:iteration:deck:index
        offsets = groups.setdefault(deck, [])
        for _ in range(game['decisions']):
            if at + HEADER.size > len(raw):
                raise ValueError(f'{path}: truncated sample header')
            count, actions, width, won, *_ = HEADER.unpack_from(raw, at)
            if not 0 < actions <= count or won != float(game['won']):
                raise ValueError(f'{path}: sample/result mismatch')
            offsets.append(at)
            at += HEADER.size + 4 * count * width + 4 * actions
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

    def __init__(self, paths, max_rows=0, balanced=True):
        paths, groups = [Path(p) for p in paths], {}
        self.paths = paths
        self.max_rows = max_rows
        self.combats = None
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
        self.balanced = balanced
        if max_rows and balanced:
            quota = max_rows // len(self.groups)
            if quota < 1:
                raise ValueError('max-train-rows must allow at least one row per deck')
            for name, rows in self.groups.items():
                if len(rows) > quota:
                    self.groups[name] = rows[torch.randperm(len(rows))[:quota].numpy()]
        if not balanced:
            self.uniform_rows = np.concatenate(list(self.groups.values()))
            self.uniform_decks = np.repeat(np.arange(len(self.groups)), [len(rows) for rows in self.groups.values()])
            if max_rows and len(self.uniform_rows) > max_rows:
                selected = torch.randperm(len(self.uniform_rows))[:max_rows].numpy()
                self.uniform_rows, self.uniform_decks = self.uniform_rows[selected], self.uniform_decks[selected]
            self.groups = {name: self.uniform_rows[self.uniform_decks == i] for i, name in enumerate(self.groups)}
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
        """Uniform rows, or equal deck quotas followed by uniform rows within each deck."""
        if self.balanced:
            n = len(self.groups)
            if size < n:
                raise ValueError('batch must be at least the number of decks')
            extra = set(torch.randperm(n)[:size % n].tolist())
            picked = []
            for i, (name, rows) in enumerate(self.groups.items()):
                take = size // n + (i in extra)
                self.drawn[name] += take
                picked.append(rows[torch.randint(len(rows), (take,)).numpy()])
            picked = np.concatenate(picked)
        else:
            selected = torch.randint(len(self.uniform_rows), (size,)).numpy()
            picked = self.uniform_rows[selected]
            counts = np.bincount(self.uniform_decks[selected], minlength=len(self.groups))
            for name, count in zip(self.groups, counts):
                self.drawn[name] += int(count)
        return self._read(picked)

    def _index_combats(self):
        if self.max_rows:
            raise ValueError('combat-balanced value sampling requires max_train_rows = 0')
        rows = np.concatenate(list(self.groups.values()))
        self.combat_rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
        spans = []
        for file, path in enumerate(self.paths):
            lo, hi = np.searchsorted(self.combat_rows[:, 0], [file, file + 1])
            offsets = np.append(self.combat_rows[lo:hi, 1], np.uint64(path.stat().st_size))
            for line in (path.parent / 'results.jsonl').read_text().splitlines():
                game = json.loads(line)
                span = game.get('sample_range')
                if span is None:
                    raise ValueError(f'{path}: value sampling requires recorded sample_range metadata')
                if span['file'] != path.name:
                    continue
                begin, end = span['begin'], span['end']
                if not 0 <= begin <= end <= offsets[-1]:
                    raise ValueError(f'{path}: invalid combat sample range')
                first, last = np.searchsorted(offsets, [begin, end])
                if offsets[first] != begin or offsets[last] != end or last - first != game['decisions']:
                    raise ValueError(f'{path}: combat range does not match sample records')
                if last > first:
                    spans.append((lo + first, lo + last))
        combats = np.asarray(sorted(spans), dtype=np.int64).reshape(-1, 2)
        if (not len(combats) or combats[0, 0] != 0 or combats[-1, 1] != len(rows)
                or np.any(combats[1:, 0] != combats[:-1, 1])):
            raise ValueError('combat ranges must partition replay records')
        self.combats = combats

    @property
    def combat_count(self):
        if self.combats is None:
            self._index_combats()
        return len(self.combats)

    def sample_combats(self, size):
        """Uniform distinct combats, one uniform state each; policy accounting is untouched."""
        if not 0 < size <= self.combat_count:
            raise ValueError('value batch must be positive and no larger than the combat count')
        # random.sample avoids a full replay-sized permutation; its seed is checkpointed through Torch.
        rng = random.Random(torch.randint(2**63 - 1, ()).item())
        selected = rng.sample(range(self.combat_count), size)
        positions = [rng.randrange(int(begin), int(end)) for begin, end in self.combats[selected]]
        return self._read(self.combat_rows[positions])

    def _read(self, picked):
        records = []
        for raw in self.reader.read(picked.tolist()):
            count, actions, width, *targets = HEADER.unpack_from(raw)
            tokens = np.frombuffer(raw, dtype=np.float32, count=count * width, offset=HEADER.size).reshape(count, width)
            policy = np.frombuffer(raw, dtype=np.float32, count=actions, offset=HEADER.size + count * width * 4)
            records.append((tokens, policy, targets))
        return records

    @staticmethod
    def collate(records):
        size, (width,) = len(records), {t.shape[1] for t, _, _ in records}  # one encoding per replay
        longest = max(len(t) for t, _, _ in records)
        tokens = np.zeros((size, longest, width), dtype=np.float32)
        policy = np.zeros((size, longest), dtype=np.float32)
        targets = np.empty((size, len(TARGETS)), dtype=np.float32)
        for i, (t, p, y) in enumerate(records):
            tokens[i, :len(t)] = t
            policy[i, len(t) - len(p):len(t)] = p
            targets[i] = y
        return torch.from_numpy(tokens), torch.from_numpy(policy), torch.from_numpy(targets)

    def batch(self, size):
        return self.collate(self.sample(size))

    def batches(self, size, attention_cells=2048 * 64 * 64, *, combat_balanced=False):
        """Keep B*T*T bounded; rare long states never pad the entire effective batch.

        All sampled rows are retained. The trainer accumulates weighted gradients
        across these microbatches before one optimizer step.
        """
        records = sorted(self.sample_combats(size) if combat_balanced else self.sample(size),
                         key=lambda r: len(r[0]))
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
