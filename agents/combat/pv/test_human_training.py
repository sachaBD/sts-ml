"""Opt-in deck manifests and flat policy supervision; no long-running gameplay."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .data import Shard, Dataset, load_split
from .model import CONTRACT, NAMES, WIDTHS
from .train import evaluate, policy_confidence


class HumanTraining(unittest.TestCase):
    def test_confidence_counts_legal_actions_including_unvisited(self):
        target = torch.tensor([[.25, .25, .25, .25], [.3, .3, .2, .2], [1., 0., 0., 0.], [.5, .5, 0., 0.]])
        legal = torch.tensor([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,0,0]], dtype=torch.bool)
        actual = policy_confidence(target, legal)
        self.assertTrue(torch.allclose(actual, torch.tensor([0., .2/3, 1., 0.]), atol=1e-6))

    def test_uniform_policy_zero_gradient_value_unchanged(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.v = torch.nn.Parameter(torch.tensor(50.))
                self.p = torch.nn.Parameter(torch.tensor([1., -1.]))
            def forward(self, *inputs):
                return self.v.expand(2), self.p.expand(2, 2)
        inputs = {name: torch.zeros(2, 2, width) for name, width in zip(NAMES, WIDTHS)}
        inputs['context'] = torch.zeros(2, WIDTHS[0])
        inputs['actions'][..., 260] = 1
        batch = (inputs, torch.tensor([0., 100.]), torch.tensor([[.5,.5], [.5,.5]]),
                 torch.ones(2, dtype=torch.bool), torch.full((2,), float('nan')))
        net = Net()
        v, p, *_ = evaluate(net, batch, 'cpu', flat_policy_weighting=True)
        v0, p0, *_ = evaluate(net, batch, 'cpu')
        self.assertEqual(v.item(), v0.item())
        self.assertGreater(v.item(), 0)
        self.assertEqual(p.item(), 0)
        self.assertGreater(p0.item(), 0)
        (v+p).backward()
        self.assertTrue(torch.equal(net.p.grad, torch.zeros(2)))

    def test_explicit_split_overrides_seed_hash_and_fails_closed(self):
        rows = []
        for i in range(3):
            r = dict(fight_id=f'fight:{i}', seed=i+1, won=bool(i%2), root_value=50.,
                     has_policy=True, policy_target=[.5,.5])
            for name, width in zip(NAMES, WIDTHS):
                count = 1 if name == 'context' else 2
                r[name] = [0.] * (width * count)
            for name, key in [('cards','n_cards'), ('monsters','n_monsters'), ('potions','n_potions'),
                              ('relics','n_relics'), ('actions','n_actions')]:
                r[key] = 2
            rows.append(r)
        table = pa.Table.from_pylist(rows).replace_schema_metadata({b'pv_contract': CONTRACT.encode()})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.parquet'
            pq.write_table(table, path)
            mapping = {'fight:0':'train', 'fight:1':'val', 'fight:2':'train'}
            manifest = Path(tmp) / 'split.json'
            manifest.write_text(json.dumps(mapping))
            self.assertEqual(load_split(manifest), mapping)
            self.assertTrue(np.array_equal(Shard(path, True, mapping).rows, [1]))
            self.assertTrue(np.array_equal(Shard(path, False, mapping).rows, [0, 2]))
            for streaming in [False, True]:
                batches = list(Dataset([path], True, streaming, mapping).batches(10))
                self.assertEqual(sum(len(batch[1]) for batch in batches), 1)
            balanced = Dataset([path], False, True, mapping, {'fight:0'}, states_per_fight=5)
            batches = list(balanced.batches(10, np.random.default_rng(0)))
            self.assertEqual(sum(len(batch[1]) for batch in batches), 5)
            self.assertTrue(all((batch[1] == 0).all() for batch in batches))
            unfiltered_val = Dataset([path], True, True, mapping, {'fight:0'}, states_per_fight=5)
            self.assertEqual(sum(len(batch[1]) for batch in unfiltered_val.batches(10)), 1)
            with self.assertRaisesRegex(ValueError, 'missing from split'):
                Shard(path, True, {'fight:0':'val'})
            manifest.write_text('{"fight:0":"test"}')
            with self.assertRaises(ValueError):
                load_split(manifest)


if __name__ == '__main__':
    unittest.main()
