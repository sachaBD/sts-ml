"""Native-format replay indexing, balanced sampling, and bounded CPU minibatches."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

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
            self.assertEqual(tokens.dtype, torch.float16)
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

    def test_reject_incomplete_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'iter001', [('a', 3, 1, 1)])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'): index(path)
            self.assertFalse(path.with_suffix('.index.npz').exists())

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
