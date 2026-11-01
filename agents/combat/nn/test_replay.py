"""Replay preserves decisions and targets while balancing and partitioning batches."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from ._test_support import small_spec
from .model import build_model
from .replay import Replay, index
from .tokens import TokenKind, WIDTH
from .train import train


def decision(identity=1, length=3, won=True):
    tokens = np.zeros((length, WIDTH), dtype=np.float32)
    tokens[0, :2] = [TokenKind.CONTEXT, identity]
    tokens[-2:, 0] = TokenKind.ACTION
    return tokens, np.array([.25, .75], dtype=np.float32), float(won)


class TinyPolicyValue(torch.nn.Module):
    """A differentiable evaluator without attention cost, for the batch-weighting invariant."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, tokens):
        values = (self.scale * tokens[:, 0, 3]).sigmoid()
        logits = self.scale * tokens[:, :, 3]
        return values, logits.masked_fill(tokens[:, :, 0] != TokenKind.ACTION, -1e9)


def full_batch_update(model, batch):
    tokens, targets, outcomes = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    values, logits = model(tokens)
    loss = F.binary_cross_entropy(values, outcomes) + F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()


def unequal_length_training_decisions():
    records = []
    for identity, length in enumerate([3] * 5 + [1300] * 3):
        tokens, policy, outcome = decision(identity, length, won=bool(identity % 2))
        tokens[0, 3] = identity / 8
        tokens[-2:, 3] = [identity / 4, -identity / 4]
        records.append((tokens, policy, outcome))
    return records


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_decks(self, name, decks):
        directory = self.directory / name
        directory.mkdir()
        path = directory / 'samples-0.bin'
        results = []
        with path.open('wb') as stream:
            for deck, decisions in decks.items():
                for number, (tokens, policy, won) in enumerate(decisions):
                    stream.write(struct.pack('<iif', len(tokens), len(policy), won))
                    stream.write(tokens.tobytes())
                    stream.write(policy.tobytes())
                    results.append(dict(fight_id=f'test:1:{deck}:{number}', won=bool(won), decisions=1))
        (directory / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results))
        return path

    def unbalanced_decks(self):
        return self.write_decks('unbalanced', {
            'common': [decision(identity=11, won=True)] * 90,
            'rare': [decision(identity=22, won=False)] * 10,
        })

    def test_balanced_cap_keeps_every_deck_within_the_row_budget(self):
        path = self.unbalanced_decks()

        with Replay([path], max_rows=16) as replay:
            self.assertLessEqual(len(replay), 16)
            tokens, _, _ = replay.batch(2)

        self.assertEqual(set(tokens[:, 0, 1].tolist()), {11, 22})

    def test_balanced_batches_do_not_favor_the_larger_deck(self):
        path = self.unbalanced_decks()

        with Replay([path]) as replay:
            tokens, _, _ = replay.batch(12)

        self.assertEqual(tokens[:, 0, 1].tolist().count(11), 6)
        self.assertEqual(tokens[:, 0, 1].tolist().count(22), 6)

    def test_sampled_targets_stay_aligned_with_their_decisions(self):
        path = self.unbalanced_decks()

        with Replay([path]) as replay:
            tokens, policy, outcomes = replay.batch(12)

        torch.testing.assert_close(outcomes, (tokens[:, 0, 1] == 11).float())
        torch.testing.assert_close(policy.sum(dim=1), torch.ones(12))
        self.assertTrue(torch.equal(policy > 0, tokens[:, :, 0] == TokenKind.ACTION))

    def test_multiple_files_contribute_to_the_same_deck(self):
        first = self.write_decks('first', {'deck': [decision()] * 4})
        second = self.write_decks('second', {'deck': [decision()] * 6})

        with Replay([first, second]) as replay:
            self.assertEqual(replay.available, dict(deck=10))

    def test_restoring_random_state_repeats_the_sampled_batch(self):
        path = self.unbalanced_decks()
        with Replay([path]) as replay:
            state = torch.get_rng_state()
            expected = replay.batch(5)

            torch.set_rng_state(state)
            actual = replay.batch(5)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unbalanced_sampling_allows_a_cap_smaller_than_the_deck_count(self):
        path = self.unbalanced_decks()

        with Replay([path], max_rows=1, balanced=False) as replay:
            self.assertEqual(len(replay), 1)
            self.assertEqual(len(replay.sample(2)), 2)

    def test_incomplete_decisions_are_rejected(self):
        path = self.write_decks('truncated', {'deck': [decision()]})
        path.write_bytes(path.read_bytes()[:-1])

        with self.assertRaises(ValueError):
            index(path)

    def test_attention_partitioning_keeps_every_sample_within_budget(self):
        lengths = [3, 3, 3, 3, 3, 30, 30, 30]
        decks = {f'deck-{identity}': [decision(identity, length)] for identity, length in enumerate(lengths)}
        path = self.write_decks('different-lengths', decks)
        with Replay([path]) as replay:
            chunks = list(replay.batches(8, attention_cells=900))

        identities = []
        for tokens, _, outcomes in chunks:
            self.assertLessEqual(len(outcomes) * tokens.shape[1] ** 2, 900)
            identities.extend(tokens[:, 0, 1].int().tolist())
        self.assertEqual(sorted(identities), list(range(8)))

    def test_collation_preserves_values_that_half_precision_would_corrupt(self):
        tokens, policy, outcome = decision()
        tokens[0, 1] = 511  # odd IDs above 256 cannot be represented exactly in BF16
        tokens[0, 3] = 95144.78125  # outside the finite FP16 range

        batch, _, _ = Replay.collate([(tokens, policy, outcome)])

        self.assertEqual(batch[0, 0, 1].item(), 511)
        self.assertEqual(batch[0, 0, 3].item(), 95144.78125)

    def test_large_balanced_batches_represent_every_deck(self):
        decks = {f'deck-{identity}': [decision(identity)] for identity in range(1000)}
        path = self.write_decks('many-decks', decks)
        with Replay([path]) as replay:
            chunks = list(replay.batches(2048))

        identities = torch.cat([tokens[:, 0, 1] for tokens, _, _ in chunks]).int().tolist()
        self.assertEqual(len(identities), 2048)
        self.assertEqual(set(identities), set(range(1000)))

    def test_unequal_microbatches_match_a_full_batch_training_update(self):
        records = unequal_length_training_decisions()
        path = self.write_decks('training', {f'deck-{i}': [record] for i, record in enumerate(records)})
        full_batch = Replay.collate(records)
        reference, partitioned = TinyPolicyValue(), TinyPolicyValue()

        full_batch_update(reference, full_batch)
        with Replay([path]) as replay:
            train(partitioned, torch.optim.SGD(partitioned.parameters(), lr=.1), replay,
                  None, None, steps=1, batch=8, device='cpu', log=lambda _: None)

        torch.testing.assert_close(partitioned(full_batch[0]), reference(full_batch[0]), rtol=1e-6, atol=1e-6)

    def test_replay_training_can_change_model_predictions(self):
        path = self.write_decks('training', {'deck': [decision()]})
        model = build_model(small_spec('transformer')).eval()
        inputs, _, _ = Replay.collate([decision()])
        before_values, _ = model(inputs)

        with Replay([path]) as replay:
            train(model, torch.optim.AdamW(model.parameters()), replay,
                  None, None, steps=2, batch=4, device='cpu', log=lambda _: None)

        after_values, _ = model.eval()(inputs)
        self.assertFalse(torch.equal(before_values, after_values))


if __name__ == '__main__':
    unittest.main()
