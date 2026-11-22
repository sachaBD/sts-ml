"""The Python bridge preserves native token bytes and the caller's action order."""
import os
from pathlib import Path
import struct
import subprocess
import unittest

from agents.combat.nn.encoding_bridge import encode


class EncodingBridgeTests(unittest.TestCase):
    def test_native_tokens_survive_the_python_boundary_in_both_encodings(self):
        build = Path(__file__).resolve().parents[3] / 'build' / os.environ.get('STSRL_BUILD_DIR', 'main')
        raw = subprocess.check_output([str(build / 'agents/combat/nn/encoding_bridge_fixture')])
        buffers, at = [], 0
        while at < len(raw):
            size, = struct.unpack_from('<I', raw, at)
            at += 4
            buffers.append(raw[at:at + size])
            at += size
        state, order, *references = buffers
        for encoding, expected in zip(('entity', 'inventory_v1'), references):
            with self.subTest(encoding=encoding):
                self.assertEqual(encode([state, state], [order, order], encoding), [expected, expected])

    def test_invalid_input_raises_without_poisoning_subsequent_calls(self):
        for states, orders, encoding in (([], [], 'unknown'), ([b''], [], 'entity'),
                                        ([b''], [b'\0'], 'entity'), ([b''], [bytes(4)], 'entity')):
            with self.assertRaises(ValueError):
                encode(states, orders, encoding)
            self.assertEqual(encode([], [], 'entity'), [])


if __name__ == '__main__':
    unittest.main()
