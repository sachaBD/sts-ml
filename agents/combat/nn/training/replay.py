"""Combat replay sampling and minibatch encoding."""
import json
import os
import random
from pathlib import Path
import struct

import numpy as np
import torch

from agents.combat.nn import encoding_bridge
from agents.combat.nn.layout import LAYOUT
from infra.workers.reader import Reader

SAMPLE_FORMAT = LAYOUT.sample.format
TOKEN_FORMAT = 3
TARGETS = tuple(LAYOUT.sample.targets)
HEADER = struct.Struct('<iii' + 'f' * len(TARGETS))
STATE_HEADER = struct.Struct('<III' + 'f' * len(TARGETS))


def games(path):
    for line in (path.parent / 'results.jsonl').read_text().splitlines():
        game = json.loads(line)
        if game.get('sample_range', {}).get('file', path.name) == path.name:
            yield game


def record_format(path):
    formats = {game.get('sample_format') for game in games(path)}
    if not formats:
        return SAMPLE_FORMAT
    if len(formats) != 1 or not formats <= {TOKEN_FORMAT, SAMPLE_FORMAT}:
        raise ValueError(f'{path}: unsupported or mixed sample formats {formats}')
    return formats.pop()


def index(path):
    """Native writes each game's rows and result under the same output lock."""
    format_ = record_format(path)
    cache = path.with_suffix('.index.npz')
    if cache.exists():
        with np.load(cache) as stored:
            return {name: stored[name] for name in stored.files}
    raw = np.memmap(path, dtype=np.uint8, mode='r') if path.stat().st_size else b''
    groups, at = {}, 0
    header = STATE_HEADER if format_ == SAMPLE_FORMAT else HEADER
    for game in games(path):
        deck = game['fight_id'].split(':')[-2]
        offsets = groups.setdefault(deck, [])
        for _ in range(game['decisions']):
            if at + header.size > len(raw):
                raise ValueError(f'{path}: truncated sample header')
            if format_ == SAMPLE_FORMAT:
                _version, state_bytes, actions, won, *_ = header.unpack_from(raw, at)
                valid = state_bytes > 0 and actions > 0
                body_size = state_bytes + 8 * actions
            else:
                count, actions, width, won, *_ = header.unpack_from(raw, at)
                valid = 0 < actions <= count and width > 0
                body_size = 4 * count * width + 4 * actions
            if not valid or won != float(game['won']):
                raise ValueError(f'{path}: sample/result mismatch')
            offsets.append(at)
            at += header.size + body_size
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
    """Selected state records stay in RAM; legacy token records remain disk-backed."""

    def __init__(self, paths, max_rows=0, balanced=True, *, encoding=None):
        paths, groups = [Path(p) for p in paths], {}
        self.paths = paths
        self.formats = [record_format(path) for path in paths]
        self.encoding = encoding
        if SAMPLE_FORMAT in self.formats and encoding is None:
            raise ValueError('state replay requires an encoding')
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
        self._resident = {}
        try:
            rows = [row.tolist() for group in self.groups.values() for row in group
                    if self.formats[int(row[0])] == SAMPLE_FORMAT]
            self._resident = {(file, offset): raw for (file, offset, _), raw
                              in zip(rows, self.reader.read(rows))}
        except BaseException:
            self.reader.close()
            raise

    def close(self):
        self.reader.close()
        self._resident.clear()

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
        rows = picked.tolist()
        requests = [row for row in rows if self.formats[row[0]] == TOKEN_FORMAT]
        legacy = iter(self.reader.read(requests)) if requests else iter(())
        raw = [self._resident[file, offset] if self.formats[file] == SAMPLE_FORMAT else next(legacy)
               for file, offset, _ in rows]
        states = [record for row, record in zip(rows, raw) if self.formats[row[0]] == SAMPLE_FORMAT]
        state_buffers, action_orders = [], []
        for record in states:
            version, state_bytes, actions, *_ = STATE_HEADER.unpack_from(record)
            if version != encoding_bridge.state_codec_version():
                raise ValueError('unsupported state codec version')
            at = STATE_HEADER.size
            state_buffers.append(record[at:at + state_bytes])
            action_orders.append(record[at + state_bytes:at + state_bytes + 4 * actions])
        encoded = iter(encoding_bridge.encode(state_buffers, action_orders, self.encoding)) if states else iter(())
        records = []
        for row, record in zip(rows, raw):
            if self.formats[row[0]] == SAMPLE_FORMAT:
                _version, state_bytes, actions, *targets = STATE_HEADER.unpack_from(record)
                tokens = np.frombuffer(next(encoded), dtype=np.float32).reshape(-1, LAYOUT.width)
                policy_at = STATE_HEADER.size + state_bytes + actions * 4
            else:
                count, actions, width, *targets = HEADER.unpack_from(record)
                tokens = np.frombuffer(record, dtype=np.float32, count=count * width,
                                       offset=HEADER.size).reshape(count, width)
                policy_at = HEADER.size + count * width * 4
                if self.encoding is not None and (width != LAYOUT.width or
                        tokens[0, LAYOUT.row.id2] != getattr(LAYOUT, self.encoding).version):
                    raise ValueError('legacy token replay does not match the requested encoding')
            policy = np.frombuffer(record, dtype=np.float32, count=actions, offset=policy_at)
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
