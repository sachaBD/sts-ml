"""Recency replay contract tests, synthetic shards only; no games or optimizer steps."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from . import recency
from .test_mixed_shards import Fake


class RecencyTests(unittest.TestCase):
    def setUp(self):
        Fake.shards = {Path('old'): (0,2,100), Path('new'): (1,2,2), Path('duplicate'): (0,2,100)}
        self.specs = [dict(path='old',generation=0),dict(path='new',generation=1)]

    def pool(self, half_life=1.):
        with patch.object(recency,'Shard',Fake):
            return recency.ReplayPool(self.specs,{},1,half_life)

    def test_half_life_and_length_independence(self):
        p=self.pool()
        np.testing.assert_allclose(p.p,[1/6,1/6,1/3,1/3])
        self.assertAlmostEqual(p.expected_generation_share()['1'],2/3)

    def test_actual_sampling_and_reproducibility(self):
        a,b=self.pool(),self.pool()
        x=a.sample(np.random.default_rng(7),6000)
        y=b.sample(np.random.default_rng(7),6000)
        np.testing.assert_array_equal(x[1].numpy(),y[1].numpy())
        self.assertEqual(a.realized,b.realized)
        self.assertEqual(sum(a.realized.values()),6000)
        self.assertLess(abs(a.generation_draws['1']/6000-2/3),.025)
        # Padding/merge preserves alignment and full sample count.
        self.assertTrue(all(len(t)==6000 for t in x[1:]))

    def test_uniform_anchor_ignores_lengths_and_age(self):
        p=self.pool(None)
        self.assertIsNone(p.p)
        self.assertEqual(p.expected_generation_share(),{'0':.5,'1':.5})

    def test_old_ages_do_not_underflow_all_weights(self):
        np.testing.assert_allclose(recency.probabilities([0,1],100000,1.),[1/3,2/3],rtol=1e-10)

    def test_invalid_ages_and_half_lives(self):
        for generations,current,half in [([],1,1),([2],1,1),([.5],1,1),([0],1,0),([0],1,-1),([0],1,float('nan')),([0],1,float('inf'))]:
            with self.subTest((generations,current,half)),self.assertRaises(ValueError):
                recency.probabilities(generations,current,half)

    def test_duplicate_fights_rejected(self):
        with patch.object(recency,'Shard',Fake),self.assertRaisesRegex(ValueError,'duplicate'):
            recency.ReplayPool([dict(path='old',generation=0),dict(path='duplicate',generation=1)],{},1,1.)

    def test_explicit_split_required(self):
        with self.assertRaisesRegex(ValueError,'explicit'):
            recency.ReplayPool(self.specs,None,1)

    def test_no_sampling_validation_fights(self):
        class SplitFake(Fake):
            def __init__(self,path,valid,split):
                super().__init__(path,valid,split)
                self.rows=np.array([r for r in self.rows if split[self.fight_ids[r]]=='train'])
        split={'0:0':'train','0:1':'val','1:0':'train','1:1':'val'}
        with patch.object(recency,'Shard',SplitFake):
            pool=recency.ReplayPool(self.specs,split,1,1.)
        pool.sample(np.random.default_rng(0),100)
        self.assertEqual(set(pool.realized),{'0:0','1:0'})


if __name__=='__main__':
    unittest.main()
