"""The Python API exercises the native protocol with a tiny, dynamic-shape ONNX network."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from agents.combat.nn.layout import LAYOUT
from environments.combat.starts import rng_state
from sts_lightspeed.enums import CARDS, POTIONS

from . import Inspector

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / 'build' / os.environ.get('STSRL_BUILD_DIR', 'main') / 'apps/combat_inspector/combat_explain'


def start(seed):
    return dict(seed=seed, ascension=1, act=1, floor=1, encounter=2,
                cur_room=4, last_room=8, burning_elite_buff=-1, hp=80, max_hp=80, gold=99,
                misc_rng=rng_state(seed), potion_rng=rng_state(seed), potion_capacity=3,
                potions=[POTIONS.index('EMPTY_POTION_SLOT')] * 3, relics=[], bottled=[-1] * 3,
                deck=[dict(id=CARDS.index(card), upgraded=False, misc=0)
                      for card in ['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5])


def model(path):
    nodes = [helper.make_node('ReduceSum', ['tokens', 'width_axis'], ['sum'], keepdims=0),
             helper.make_node('Mul', ['sum', 'zero'], ['logits']),
             helper.make_node('ReduceMean', ['logits'], ['mean'], axes=[1], keepdims=0),
             helper.make_node('Add', ['mean', 'half'], ['value'])]
    constants = [numpy_helper.from_array(np.array(value, dtype=dtype), name)
                 for name, value, dtype in [('width_axis', [2], np.int64), ('zero', 0, np.float32),
                                            ('half', 0.5, np.float32)]]
    graph = helper.make_graph(nodes, 'flat', [helper.make_tensor_value_info('tokens', TensorProto.FLOAT,
                               ['batch', 'tokens', LAYOUT.width])],
                              [helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch']),
                               helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 'tokens'])], constants)
    network = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(network)
    onnx.save(network, path)


@unittest.skipUnless(BINARY.exists(), 'combat_explain not built')
class InspectorTests(unittest.TestCase):
    def test_replay_graph_and_errors_over_one_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.onnx'
            model(path)
            with Inspector(BINARY) as inspector:
                for encoding in ('entity', 'inventory_v1'):
                    request = dict(model=path, encoding=encoding, start=start((1 << 63) + 1), sims=8, particles=2)
                    first = inspector.decision(**request)
                    self.assertEqual(first, inspector.decision(**request))
                    graph = first['tree']
                    nodes = {n['observation']: n for n in graph['nodes']}
                    self.assertEqual(graph['root'], first['observation'])
                    self.assertTrue(any(e['bits'] == first['decision']['action'] for e in nodes[graph['root']]['edges']))
                    for node in nodes.values():
                        if not node['terminal']:
                            self.assertEqual(node['network'], 0.5)
                        for edge in node['edges']:
                            self.assertEqual(edge['visits'], sum(c['visits'] for c in edge['children']))
                            self.assertTrue(all(c['observation'] in nodes for c in edge['children']))
                    after = inspector.decision(**request, moves=[first['decision']['action']])
                    self.assertGreater(after['steps'], first['steps'])
                    with self.assertRaises(ValueError):
                        inspector.decision(**request, moves=[-1])
                    self.assertEqual(first, inspector.decision(**request))
            self.assertIsNone(inspector._process)


if __name__ == '__main__':
    unittest.main()
