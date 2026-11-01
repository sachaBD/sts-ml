"""Dense replay returns the current window's data, regardless of caching strategy."""
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from .data import Replay
from .tokens import TokenKind, WIDTH


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.replay = Replay()

    def write_decision(self, name, identity, length=3):
        tokens = np.zeros((length, WIDTH), dtype=np.float32)
        tokens[0, :2] = [TokenKind.CONTEXT, identity]
        tokens[-1, 0] = TokenKind.ACTION
        path = self.directory / name
        path.write_bytes(struct.pack('<iif', length, 1, 1.) + tokens.tobytes() + struct.pack('<f', 1.))
        return path

    def test_window_excludes_old_files(self):
        old = self.write_decision('old.bin', identity=1)
        retained = self.write_decision('retained.bin', identity=2)
        new = self.write_decision('new.bin', identity=3)
        self.replay.read([old, retained])

        tokens, _, _ = self.replay.read([retained, new])

        self.assertEqual(tokens[:, 0, 1].tolist(), [2, 3])

    def test_padding_does_not_add_policy_targets(self):
        short = self.write_decision('short.bin', identity=1, length=3)
        long = self.write_decision('long.bin', identity=2, length=9)

        tokens, policy, _ = self.replay.read([short, long])

        non_actions = tokens[..., 0] != TokenKind.ACTION
        self.assertTrue((policy[non_actions] == 0).all())
        self.assertEqual(policy.sum(dim=1).tolist(), [1., 1.])

    def test_empty_files_contribute_no_rows(self):
        empty = self.directory / 'empty.bin'
        empty.touch()
        decision = self.write_decision('decision.bin', identity=7)

        tokens, _, _ = self.replay.read([empty, decision])

        self.assertEqual(tokens[:, 0, 1].tolist(), [7])

    def test_changed_files_do_not_return_stale_data(self):
        path = self.write_decision('decision.bin', identity=1)
        self.replay.read([path])
        modified = path.stat().st_mtime_ns + 1
        self.write_decision('decision.bin', identity=2)
        os.utime(path, ns=(modified, modified))

        tokens, _, _ = self.replay.read([path])

        self.assertEqual(tokens[:, 0, 1].tolist(), [2])


if __name__ == '__main__':
    unittest.main()
