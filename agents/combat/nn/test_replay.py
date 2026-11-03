"""Native-format replay indexing, balanced sampling, and bounded CPU minibatches."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .model import ModelSpec, build_model
from .tokens import WIDTH
from .replay import Replay, index
from .train import score, train


def samples(directory, decks):
    directory.mkdir()
    path = directory / 'samples-0.bin'
    results = []
    with path.open('wb') as stream:
        for deck, count, identity, won in decks:
            begin = stream.tell()
            for i in range(count):
                n = 3 + i % 3
                tokens = np.zeros((n, WIDTH), dtype=np.float32)
                tokens[0, :2] = [1, identity]
                tokens[-2:, 0] = 6
                stream.write(struct.pack('<iif', n, 2, won))
                stream.write(tokens.tobytes())
                stream.write(np.array([.25, .75], dtype=np.float32).tobytes())
            results.append(dict(fight_id=f'test:1:{deck}:0', won=bool(won), decisions=count,
                                sample_range=dict(file=path.name, begin=begin, end=stream.tell())))
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

    def test_value_sampling_balances_combats_not_decks_or_lengths(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = samples(root / 'a', [('same', 40, 1, 1), ('same', 2, 2, 0), ('empty', 0, 9, 0)])
            b = samples(root / 'b', [('other', 1, 3, 1)])
            c = samples(root / 'c', [('empty', 0, 4, 0)])
            with Replay([a, b, c], balanced=False) as replay:
                seen_lengths = set()
                for size in (1, 2, 3):
                    for _ in range(20):
                        records = replay.sample_combats(size)
                        identities = [int(t[0, 1]) for t, _, _ in records]
                        self.assertEqual(len(identities), size)
                        self.assertEqual(len(set(identities)), size)
                        self.assertTrue(set(identities) <= {1, 2, 3})
                        seen_lengths.update(len(t) for t, _, _ in records if t[0, 1] == 1)
                self.assertGreater(len(seen_lengths), 1)  # states are resampled, not fixed per combat
                draws = 600
                counts = np.zeros(3, dtype=int)
                for _ in range(draws):
                    tokens, _, _ = replay.sample_combats(1)[0]
                    counts[int(tokens[0, 1]) - 1] += 1
                # Equal combat inclusion, with a six-standard-deviation sampling tolerance.
                self.assertTrue(np.all(abs(counts - draws / 3) < 6 * np.sqrt(draws * (1 / 3) * (2 / 3))))
                self.assertEqual(sum(replay.drawn.values()), 0)  # value does not alter policy accounting
                with self.assertRaisesRegex(ValueError, 'combat count'):
                    replay.sample_combats(4)

    def test_combat_ranges_partition_replay_regardless_of_journal_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'a', [('same', 4, 1, 1), ('same', 4, 2, 0)])
            index(path)
            journal = path.parent / 'results.jsonl'
            games = list(reversed(journal.read_text().splitlines()))
            journal.write_text('\n'.join(games) + '\n')
            with Replay([path], balanced=False) as replay:
                self.assertEqual({int(t[0, 1]) for t, _, _ in replay.sample_combats(2)}, {1, 2})
            journal.write_text('\n'.join([games[0], games[0]]) + '\n')
            with Replay([path], balanced=False) as replay:
                with self.assertRaisesRegex(ValueError, 'partition'):
                    replay.sample_combats(2)

    def test_value_sampling_requires_uncapped_combat_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'a', [('same', 4, 1, 1)])
            for balanced in (False, True):
                with Replay([path], max_rows=2, balanced=balanced) as replay:
                    with self.assertRaisesRegex(ValueError, 'max_train_rows'):
                        replay.sample_combats(1)
            index(path)
            journal = path.parent / 'results.jsonl'
            game = json.loads(journal.read_text())
            game['decisions'] += 1
            journal.write_text(json.dumps(game) + '\n')
            with Replay([path], balanced=False) as replay:
                with self.assertRaisesRegex(ValueError, 'combat range'):
                    replay.sample_combats(1)
            game['decisions'] -= 1
            del game['sample_range']
            journal.write_text(json.dumps(game) + '\n')
            path.with_suffix('.index.npz').unlink()
            with Replay([path], balanced=False) as replay:
                self.assertEqual(len(replay.sample(1)), 1)  # legacy policy replay still works
                with self.assertRaisesRegex(ValueError, 'sample_range'):
                    replay.sample_combats(1)

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
            for combat_balanced in (False, True):
                sampler = 'sample_combats' if combat_balanced else 'sample'
                with patch.object(replay, sampler, return_value=records):
                    chunks = list(replay.batches(8, attention_cells=900, combat_balanced=combat_balanced))
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
            net = build_model(ModelSpec('transformer', 8, 1, 1))
            optimizer = torch.optim.AdamW(net.parameters())
            before = net.value.weight.detach().clone()
            train(net, optimizer, replay, None, None, 2, 4, 'cpu', log=lambda _: None)
            self.assertFalse(torch.equal(before, net.value.weight))
            self.assertEqual(replay.drawn, dict(a=4, b=4))

    def test_score_is_the_training_loss_without_the_update(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 4, 1, 1), ('b', 4, 2, 0)])
            replay = Replay([path])
            chunks = [Replay.collate(replay.sample(3)), Replay.collate(replay.sample(5))]
            net = build_model(ModelSpec('transformer', 8, 1, 1))
            optimizer = torch.optim.AdamW(net.parameters())
            before = {k: v.clone() for k, v in net.state_dict().items()}
            with patch.object(replay, 'batches', side_effect=lambda _: iter(chunks)):
                scored = score(net, replay, 8, 8, 'cpu')
                self.assertTrue(all(torch.equal(v, before[k]) for k, v in net.state_dict().items()))
                self.assertEqual(optimizer.state, {})
                trained = train(net, optimizer, replay, None, None, 1, 8, 'cpu', log=lambda _: None)
            torch.testing.assert_close(scored, trained)


if __name__ == '__main__':
    unittest.main()
