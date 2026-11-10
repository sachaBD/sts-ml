"""Native-format replay indexing, balanced sampling, and bounded CPU minibatches."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from agents.combat.nn.tokens import WIDTH
from agents.combat.nn.training.replay import HEADER, SAMPLE_FORMAT, TARGETS, Replay, index


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
                stream.write(HEADER.pack(n, 2, won, *[.5] * (len(TARGETS) - 1)))
                stream.write(tokens.tobytes())
                stream.write(np.array([.25, .75], dtype=np.float32).tobytes())
            results.append(dict(fight_id=f'test:1:{deck}:0', won=bool(won), decisions=count, sample_format=SAMPLE_FORMAT,
                                sample_range=dict(file=path.name, begin=begin, end=stream.tell())))
    (directory / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    return path


class ReplayTests(unittest.TestCase):
    def test_balanced_sampling_draws_every_deck_equally(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('long', 90, 11, 1), ('short', 10, 22, 0)])
            replay = Replay([path], max_rows=16)
            self.assertEqual(replay.available, dict(long=90, short=10))
            self.assertEqual(replay.selected, dict(long=8, short=8))
            tokens, policy, targets = replay.batch(12)
            self.assertEqual(tokens.device.type, 'cpu')
            self.assertEqual(tokens.dtype, torch.float32)
            self.assertEqual(int((tokens[:, 0, 1] == 11).sum()), 6)
            self.assertEqual(int((tokens[:, 0, 1] == 22).sum()), 6)
            torch.testing.assert_close(policy.sum(1), torch.ones(12))
            torch.testing.assert_close(targets[:, TARGETS.index('won')], (tokens[:, 0, 1] == 11).float())
            self.assertTrue(torch.equal(policy > 0, tokens[:, :, 0] == 6))
            self.assertTrue(path.with_suffix('.index.npz').exists())
            self.assertEqual({k: len(v) for k, v in index(path).items()}, replay.available)

    def test_value_samples_take_one_state_per_fight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = samples(root / 'a', [('same', 40, 1, 1), ('same', 2, 2, 0), ('empty', 0, 9, 0)])
            b = samples(root / 'b', [('other', 1, 3, 1)])
            for balanced in (False, True):
                with self.subTest(balanced=balanced), Replay([a, b], balanced=balanced) as replay:
                    for size in (1, 2, 3):
                        records = replay.sample_combats(size)
                        identities = [int(t[0, 1]) for t, _, _ in records]
                        self.assertEqual(len(identities), size)
                        self.assertEqual(len(set(identities)), size)
                        self.assertTrue(set(identities) <= {1, 2, 3})
                    self.assertEqual(sum(replay.drawn.values()), 0)

    def test_truncated_sample_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 3, 1, 1)])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'): index(path)
            self.assertFalse(path.with_suffix('.index.npz').exists())

    def test_other_sample_formats_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 3, 1, 1)])
            results = path.parent / 'results.jsonl'
            results.write_text(results.read_text().replace(f'"sample_format": {SAMPLE_FORMAT}', '"sample_format": 1'))
            with self.assertRaisesRegex(ValueError, 'sample format'): index(path)

    def test_large_values_and_ids_survive_loading(self):
        t = np.zeros((3, WIDTH), dtype=np.float32)
        t[0, :4] = [1, 511, 0, 95144.78125]; t[-2:, 0] = 6
        tokens, _, _ = Replay.collate([(t, np.array([.5, .5], np.float32), 1)])
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(tokens[0, 0, 1].item(), 511)  # BF16 transport would corrupt odd IDs >256
        self.assertEqual(tokens[0, 0, 3].item(), 95144.78125)

if __name__ == '__main__':
    unittest.main()
