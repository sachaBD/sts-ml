"""Replay returns exactly read()'s tensors over a sliding window and reads each file once."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agents.combat.nn import data
from agents.combat.nn.model import WIDTH


def write(path, seed, records, longest):
    rng = np.random.default_rng(seed)
    with open(path, 'wb') as f:
        for _ in range(records):
            count = int(rng.integers(1, longest + 1))
            actions = int(rng.integers(0, count + 1))
            f.write(np.array([count, actions], np.int32).tobytes() + np.float32(rng.random() < .5).tobytes())
            f.write(rng.normal(0, 100, (count, WIDTH)).astype(np.float32).tobytes())
            f.write(rng.random(actions).astype(np.float32).tobytes())
    return path


class ReplayTests(unittest.TestCase):
    def test_sliding_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Longest state grows then shrinks; an empty file and several files per iteration.
            files = [write(Path(tmp) / f'{k}.bin', k, 10, n) for k, n in enumerate([4, 9, 3, 12, 5, 2])]
            files.insert(2, Path(tmp) / 'empty.bin')
            files[2].write_bytes(b'')
            replay = data.Replay()
            with patch.object(data, 'read', wraps=data.read) as read:
                for start in range(len(files) - 2):
                    window = files[start:start + 3]
                    for expected, actual in zip(data.read(window), replay.read(window)):
                        self.assertEqual((expected.dtype, expected.shape), (actual.dtype, actual.shape))
                        self.assertEqual(expected.numpy().tobytes(), actual.numpy().tobytes())
            self.assertEqual(read.call_count, (len(files) - 2) + (len(files) - 1))  # reference per window + each nonempty file once
            os.utime(files[-1], ns=(0, 0))  # a changed file is read again
            with patch.object(data, 'read', wraps=data.read) as read:
                replay.read(files[-3:])
            self.assertEqual(read.call_count, 1)


if __name__ == '__main__':
    unittest.main()
