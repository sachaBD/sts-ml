"""Native-format replay indexing, balanced sampling, and bounded CPU minibatches."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .model import Net, WIDTH
from .replay import Replay, index
from .train import train


def samples(directory, decks):
    directory.mkdir()
    path = directory / 'samples-0.bin'
    results = []
    with path.open('wb') as stream:
        for deck, count, identity, won in decks:
            for i in range(count):
                n = 3 + i % 3
                tokens = np.zeros((n, WIDTH), dtype=np.float32)
                tokens[0, :2] = [1, identity]
                tokens[-2:, 0] = 6
                stream.write(struct.pack('<iif', n, 2, won))
                stream.write(tokens.tobytes())
                stream.write(np.array([.25, .75], dtype=np.float32).tobytes())
            results.append(dict(fight_id=f'test:1:{deck}:0', won=bool(won), decisions=count))
    (directory / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    return path


class ReplayTests(unittest.TestCase):
    def test_balanced_cap_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('long', 90, 11, 1), ('short', 10, 22, 0)])
            replay = Replay([path], max_rows=16)
            self.assertEqual(replay.available, dict(long=90, short=10))
            self.assertEqual(replay.selected, dict(long=8, short=8))
            tokens, policy, won = replay.batch(12)
            self.assertEqual(tokens.device.type, 'cpu')
            self.assertEqual(tokens.dtype, torch.float32)
            self.assertEqual(int((tokens[:, 0, 1] == 11).sum()), 6)
            self.assertEqual(int((tokens[:, 0, 1] == 22).sum()), 6)
            torch.testing.assert_close(policy.sum(1), torch.ones(12))
            torch.testing.assert_close(won, (tokens[:, 0, 1] == 11).float())
            self.assertTrue(torch.equal(policy > 0, tokens[:, :, 0] == 6))
            self.assertTrue(path.with_suffix('.index.npz').exists())
            self.assertEqual({k: len(v) for k, v in index(path).items()}, replay.available)

    def test_multiple_files_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = samples(root / 'iter001', [('a', 4, 1, 0), ('b', 5, 2, 1)])
            b = samples(root / 'iter002', [('a', 6, 1, 0), ('b', 8, 2, 1)])
            replay = Replay([a, b])
            self.assertEqual(replay.available, dict(a=10, b=13))
            state = torch.get_rng_state()
            first = replay.batch(5)
            torch.set_rng_state(state)
            second = replay.batch(5)
            for x, y in zip(first, second): torch.testing.assert_close(x, y)
            self.assertEqual(sum(replay.drawn.values()), 10)
            self.assertLessEqual(abs(replay.drawn['a'] - replay.drawn['b']), 2)

    def test_uniform_rows_do_not_rebalance_decks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 3, 1, 1), ('b', 1, 2, 0)])
            with Replay([path], balanced=False) as replay:
                with patch('torch.randint', return_value=torch.arange(4)):
                    rows = replay.sample(4)
                self.assertEqual(len(rows), 4)
                self.assertEqual(replay.drawn, {'a': 3, 'b': 1})

    def test_uniform_cap_can_be_smaller_than_deck_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 3, 1, 1), ('b', 1, 2, 0)])
            with Replay([path], max_rows=1, balanced=False) as replay:
                self.assertEqual(len(replay), 1)
                self.assertEqual(len(replay.sample(2)), 2)
                self.assertEqual(sum(replay.drawn.values()), 2)

    def test_reject_incomplete_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 3, 1, 1)])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'): index(path)
            self.assertFalse(path.with_suffix('.index.npz').exists())

    def test_attention_buckets_keep_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 4, 1, 1), ('b', 4, 2, 0)])
            replay = Replay([path])
            records = []
            for i, length in enumerate([3] * 5 + [30] * 3):
                t = np.zeros((length, WIDTH), dtype=np.float32)
                t[0, :2] = [1, i]; t[-2:, 0] = 6
                records.append((t, np.array([.5, .5], np.float32), float(i % 2)))
            with patch.object(replay, 'sample', return_value=records):
                chunks = list(replay.batches(8, attention_cells=900))
            self.assertEqual(sum(len(w) for _, _, w in chunks), 8)
            self.assertEqual(len(chunks), 4)
            identities = []
            for t, p, w in chunks:
                self.assertLessEqual(len(w) * t.shape[1] ** 2, 900)
                identities += t[:, 0, 1].int().tolist()
                torch.testing.assert_close(p.sum(1), torch.ones(len(w)))
            self.assertEqual(sorted(identities), list(range(8)))

    def test_replay_preserves_values_outside_fp16_range(self):
        t = np.zeros((3, WIDTH), dtype=np.float32)
        t[0, :4] = [1, 511, 0, 95144.78125]; t[-2:, 0] = 6
        tokens, _, _ = Replay.collate([(t, np.array([.5, .5], np.float32), 1)])
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(tokens[0, 0, 1].item(), 511)  # BF16 transport would corrupt odd IDs >256
        self.assertEqual(tokens[0, 0, 3].item(), 95144.78125)

    def test_thousand_decks_in_every_effective_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [(f'd{i}', 1, i % 100, i % 2) for i in range(1000)])
            replay = Replay([path])
            chunks = list(replay.batches(2048))
            self.assertEqual(sum(len(w) for _, _, w in chunks), 2048)
            self.assertEqual(len(replay.drawn), 1000)
            self.assertEqual(set(replay.drawn.values()), {2, 3})
            self.assertEqual(sum(replay.drawn.values()), 2048)

    def test_unequal_microbatches_match_one_effective_update(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(.2))
            def forward(self, t):
                return (self.scale * t[:, 0, 3]).sigmoid(), (self.scale * t[:, :, 3]).masked_fill(t[:, :, 0] != 6, -1e9)
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 4, 1, 1), ('b', 4, 2, 0)])
            replay = Replay([path])
            records = []
            for i in range(8):
                t = np.zeros((3, WIDTH), np.float32); t[0, 0] = 1; t[-2:, 0] = 6
                t[:, 3] = [i / 8, i / 4, -i / 4]
                records.append((t, np.array([.25, .75], np.float32), i % 2))
            full = Replay.collate(records); a, b = Tiny(), Tiny()
            with patch('torch.randint', return_value=torch.arange(8)):
                train(a, torch.optim.SGD(a.parameters(), lr=.1), *full, 1, 8, 'cpu', log=lambda _: None)
            chunks = [Replay.collate(records[:3]), Replay.collate(records[3:])]
            with patch.object(replay, 'batches', return_value=iter(chunks)):
                train(b, torch.optim.SGD(b.parameters(), lr=.1), replay, None, None, 1, 8, 'cpu', log=lambda _: None)
            torch.testing.assert_close(a.scale, b.scale, rtol=1e-6, atol=1e-6)

    def test_cpu_optimizer_uses_batches(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 4, 1, 1), ('b', 4, 2, 0)])
            replay = Replay([path])
            net = Net(dim=8, layers=1, heads=1)
            optimizer = torch.optim.AdamW(net.parameters())
            before = net.value.weight.detach().clone()
            train(net, optimizer, replay, None, None, 2, 4, 'cpu', log=lambda _: None)
            self.assertFalse(torch.equal(before, net.value.weight))
            self.assertEqual(replay.drawn, dict(a=4, b=4))


if __name__ == '__main__':
    unittest.main()
