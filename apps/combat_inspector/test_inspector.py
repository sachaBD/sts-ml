"""The checkpoint manifest selects the encoder and CPU artifact; inspection replays the player."""
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import tomli_w

from agents.combat.nn.layout import LAYOUT
from environments.combat.starts import rng_state
from sts_lightspeed.enums import CARDS, POTIONS

from . import decision

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / 'build' / os.environ.get('STSRL_BUILD_DIR', 'main') / 'apps/combat_inspector/combat_explain'


def jaw_worm(seed):
    # Fight-start IDs: Jaw Worm (2), monster room (4), no previous room (8).
    return dict(seed=seed, ascension=1, act=1, floor=1, encounter=2, cur_room=4, last_room=8,
                burning_elite_buff=-1, hp=80, max_hp=80, gold=99, misc_rng=rng_state(seed),
                potion_rng=rng_state(seed), potion_capacity=3,
                potions=[POTIONS.index('EMPTY_POTION_SLOT')] * 3, relics=[], bottled=[-1] * 3,
                deck=[dict(id=CARDS.index(card), upgraded=False, misc=0)
                      for card in ['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5])


def checkpoint(directory, encoding):
    # A token-sensitive network: value = sigmoid(context version); logits = mean(token).
    nodes = [helper.make_node('Gather', ['tokens', 'context_index'], ['context'], axis=1),
             helper.make_node('Gather', ['context', 'version_index'], ['version'], axis=1),
             helper.make_node('Sigmoid', ['version'], ['value']),
             helper.make_node('ReduceMean', ['tokens'], ['logits'], axes=[2], keepdims=0)]
    indices = [numpy_helper.from_array(np.array(value, dtype=np.int64), name)
               for name, value in [('context_index', 0), ('version_index', LAYOUT.row.id2)]]
    graph = helper.make_graph(nodes, 'encoding_probe',
                              [helper.make_tensor_value_info('tokens', TensorProto.FLOAT, ['batch', 'tokens', LAYOUT.width])],
                              [helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch']),
                               helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 'tokens'])], indices)
    network = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(network)
    onnx.save(network, directory / 'probe.onnx')
    manifest = dict(encoding=encoding, deployments=[dict(files=['unused-gpu.onnx'], lanes=1)],
                    cpu=dict(file='probe.onnx', lanes=1, batch=1))
    (directory / 'network.toml').write_text(tomli_w.dumps(manifest))


@unittest.skipUnless(BINARY.exists(), 'combat_explain not built')
class InspectorTests(unittest.TestCase):
    def test_checkpoint_determines_the_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for encoding in ('entity', 'inventory_v1'):
                with self.subTest(encoding=encoding):
                    checkpoint(directory, encoding)
                    result = decision(directory, jaw_worm(7), sims=0, executable=BINARY)
                    expected = 1 / (1 + math.exp(-getattr(LAYOUT, encoding).version))
                    self.assertAlmostEqual(result['decision']['value'], expected, places=6)

    def test_replay_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint(directory, 'inventory_v1')
            request = dict(checkpoint=directory, start=jaw_worm((1 << 63) + 1), sims=8, particles=2,
                           gumbel_scale=1, rollout_mix=0.5, executable=BINARY)
            first = decision(**request)
            self.assertEqual(first, decision(**request))

    def test_chosen_move_advances_the_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint(directory, 'inventory_v1')
            request = dict(checkpoint=directory, start=jaw_worm(7), sims=8, executable=BINARY)
            first = decision(**request)
            after = decision(**request, moves=[first['decision']['action']])
            self.assertGreater(after['steps'], first['steps'])

    def test_invalid_requests_fail_with_a_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint(directory, 'entity')
            for invalid in (dict(moves=[-1]), dict(moves=[0.5]), dict(moves=[(1 << 32) - 1]),
                            dict(particles=0), dict(rollout_mix=2), dict(sims=0.5), dict(particles=1 << 40)):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(subprocess.CalledProcessError) as raised:
                        decision(directory, jaw_worm(7), executable=BINARY, **invalid)
                    self.assertTrue(raised.exception.stderr.strip())

    def test_finished_fights_have_no_search(self):
        start = jaw_worm(7)
        start['hp'] = 1
        for card in start['deck']:
            card['id'] = CARDS.index('INJURY')
        result = decision('/unused-checkpoint', start, executable=BINARY)
        self.assertTrue(result['over'])
        self.assertNotIn('decision', result)
        self.assertNotIn('graph', result)


if __name__ == '__main__':
    unittest.main()
