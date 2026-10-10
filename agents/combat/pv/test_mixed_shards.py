"""Mixed-shard minibatches, per-fight sampling and duplicate-fight rejection (fake shards; no parquet, no play)."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import data
from .train import sum_groups


class Fake:
    """Stand-in for data.Shard: row r of shard `tag` has value tag*1000+r and a shard-specific token count."""
    shards = {}

    def __init__(self, path, valid, split=None, train_fights=None):
        tag, fights, per = self.shards[path]
        self.tag = tag
        self.fight_ids = np.repeat([f'{tag}:{i}' for i in range(fights)], per)
        self.rows = np.arange(fights * per)

    def batch(self, idx):
        n = 2 + self.tag  # different padded widths per shard
        x = torch.full((len(idx), n, 3), float(self.tag))
        value = torch.tensor([self.tag * 1000. + r for r in idx])
        return ({'context': torch.full((len(idx), 4), float(self.tag)), 'cards': x}, value,
                torch.ones(len(idx), n), torch.ones(len(idx), dtype=torch.bool), torch.full((len(idx),), float('nan')))


class MixedShards(unittest.TestCase):
    def setUp(self):
        Fake.shards = {'a': (0, 3, 10), 'b': (1, 2, 10), 'c': (2, 4, 5)}

    def test_batches_mix_shards_and_keep_per_fight_counts(self):
        with patch.object(data, 'Shard', Fake):
            ds = data.Dataset(['a', 'b', 'c'], False, states_per_fight=7, mix=True)
            batches = list(ds.batches(16, np.random.default_rng(0)))
        self.assertEqual(sum(len(b[1]) for b in batches), 9 * 7)
        self.assertTrue(all(c == 7 for c in ds.realized.values()) and len(ds.realized) == 9)
        mixed = [b for b in batches if len({int(v) // 1000 for v in b[1]}) >= 2]
        self.assertGreaterEqual(len(mixed), len(batches) - 1)
        b = mixed[0]
        self.assertEqual(b[0]['cards'].shape[1], 2 + max(int(v) // 1000 for v in b[1]))  # padded to common count
        self.assertEqual(b[2].shape[1], b[0]['cards'].shape[1])
        self.assertEqual(len({len(t) for t in (b[1], b[2], b[3], b[4], b[0]['cards'])}), 1)

    def test_default_path_unchanged_is_shard_homogeneous(self):
        with patch.object(data, 'Shard', Fake):
            ds = data.Dataset(['a', 'b'], False, states_per_fight=5)
            for b in ds.batches(8, np.random.default_rng(0)):
                self.assertEqual(len({int(v) // 1000 for v in b[1]}), 1)

    def test_duplicate_fight_across_shards_is_an_error(self):
        Fake.shards = {'a': (0, 2, 3), 'dup': (0, 2, 3)}
        with patch.object(data, 'Shard', Fake), self.assertRaisesRegex(ValueError, 'more than one shard'):
            data.Dataset(['a', 'dup'], False)

    def test_mix_rejects_stream_and_groups_sum(self):
        with self.assertRaises(ValueError):
            data.Dataset(['a'], False, stream=True, mix=True)
        self.assertEqual(sum_groups({'x:1': 3, 'x:2': 4, 'y:1': 5}, {'x:1': 'X', 'x:2': 'X', 'y:1': 'Y'}), {'X': 7, 'Y': 5})


if __name__ == '__main__':
    unittest.main()
