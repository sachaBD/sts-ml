"""State replay preserves decision inputs and policies across collection, encoding, and training."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents.combat.nn.model import ModelSpec
from agents.combat.nn.training.learner import Learner
from agents.combat.nn.training.replay import Replay, native_encoder


class StateReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        build = Path(__file__).resolve().parents[4] / 'build' / os.environ.get('STSRL_BUILD_DIR', 'main')
        subprocess.run([str(build / 'agents/combat/nn/state_replay_fixture'), str(cls.root)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def path(self, encoding='state'):
        return self.root / encoding / 'samples-0.bin'

    def test_reencoding_matches_collection_tokens_in_both_encodings(self):
        for encoding in ('entity', 'inventory_v1'):
            with (self.subTest(encoding=encoding), Replay([self.path()], encoding=encoding) as replay,
                  Replay([self.path(encoding)], encoding=encoding) as reference):
                self.assertEqual(replay.available, reference.available)
                for deck in replay.groups:
                    actual = replay._read(replay.groups[deck])
                    expected = reference._read(reference.groups[deck])
                    for row, original in zip(actual, expected):
                        for value, before in zip(row, original):
                            np.testing.assert_array_equal(value, before)

    def test_only_selected_records_are_resident_and_batches_do_not_read_disk(self):
        for balanced in (True, False):
            self.check_resident_selection(balanced)

    def check_resident_selection(self, balanced):
        with Replay([self.path()], max_rows=6, balanced=balanced, encoding='entity') as replay:
            self.assertEqual(len(replay._resident), len(replay))
            self.assertEqual(replay.resident_bytes, sum(int(row[2]) for group in replay.groups.values() for row in group))
            self.assertLess(len(replay), sum(replay.available.values()))
            with patch.object(replay.reader, 'read', side_effect=AssertionError('unexpected disk read')):
                tokens, policy, targets = replay.batch(6)
            self.assertEqual(len(tokens), 6)
            self.assertTrue(torch.isfinite(targets).all())
            torch.testing.assert_close(policy.sum(1), torch.ones(6))
        self.assertFalse(replay._resident)

    def test_value_sampling_also_uses_resident_records(self):
        with Replay([self.path()], encoding='entity') as replay:
            with patch.object(replay.reader, 'read', side_effect=AssertionError('unexpected disk read')):
                records = replay.sample_combats(replay.combat_count)
            self.assertEqual(len(records), replay.combat_count)
            self.assertEqual(sum(replay.drawn.values()), 0)

    def test_old_and_new_records_can_share_a_replay_window(self):
        with Replay([self.path('entity'), self.path()], encoding='entity') as replay:
            rows = np.concatenate(list(replay.groups.values()))
            picked = np.stack([rows[rows[:, 0] == file][0] for file in (0, 1)])
            old, new = replay._read(picked)
            for before, after in zip(old, new):
                np.testing.assert_array_equal(before, after)

    def test_encoding_is_explicit_and_legacy_tokens_cannot_be_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, 'encoding'):
            Replay([self.path()])
        with Replay([self.path('entity')], encoding='inventory_v1') as replay:
            with self.assertRaisesRegex(ValueError, 'encoding'):
                replay.batch(3)

    def test_native_errors_do_not_poison_later_batches(self):
        with Replay([self.path()], encoding='entity') as replay:
            raw = next(iter(replay._resident.values()))
            for bad in (raw[:-1], bytes(4) + raw[4:]):
                with self.assertRaises(ValueError):
                    native_encoder().encode([bad], 'entity')
            self.assertEqual(len(native_encoder().encode([raw], 'entity')), 1)

    def test_state_replay_can_score_and_train_both_models(self):
        for encoding in ('entity', 'inventory_v1'):
            with self.subTest(encoding=encoding), Replay([self.path()], encoding=encoding) as replay:
                learner = Learner(ModelSpec(encoding, 8, 1, 1, 'win', False), 1e-3, 'cpu')
                self.assertTrue(np.isfinite(learner.score(replay, 6, 6)).all())
                before = {name: p.detach().clone() for name, p in learner.net.named_parameters()}
                measured = learner.fit(replay, 2, 6, log=lambda _: None)
                self.assertTrue(np.isfinite(measured).all())
                self.assertTrue(any(not torch.equal(before[name], p) for name, p in learner.net.named_parameters()))


if __name__ == '__main__':
    unittest.main()
