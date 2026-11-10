"""Deployment must preserve predictions and follow the current weights."""
from pathlib import Path
import tempfile
import unittest

import onnxruntime as ort
import torch

from agents.combat.nn.entity.test_model import combat_tokens, small_spec
from agents.combat.nn.tokens import WIDTH
from agents.combat.nn.training.export import ExportSpec, ModelExporter, deployment_view
from agents.combat.nn.training.learner import Learner
from agents.combat.nn.training.test_learner import train_once


def onnx_predictions(path, tokens):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    return tuple(torch.from_numpy(output) for output in session.run(None, {'tokens': tokens.numpy()}))


def model_predictions(model, tokens):
    with torch.no_grad():
        return tuple(output.cpu() for output in model(tokens))


def target_tokens():
    """Two same-species monsters and actions that differ only in which one they target, plus that scene permuted."""
    tokens = torch.zeros(1, 8, WIDTH)
    tokens[0, :, 0] = torch.tensor([1, 3, 3, 6, 6, 6, 6, 6])
    tokens[0, 1:3, 1] = 17
    tokens[0, 1:3, 3] = torch.tensor([.2, .8])
    tokens[0, 3:5, 1:3] = torch.tensor([23, 1])  # non-damaging card
    tokens[0, 5:7, 1:3] = torch.tensor([11, 2])  # targeted potion
    tokens[0, 3:7, -1] = torch.tensor([1, 2, 1, 2])
    tokens[0, 7, 2] = 5  # no-target End Turn
    permuted = tokens[:, [0, 2, 1, 3, 4, 5, 6, 7]].clone()
    permuted[0, 3:7, -1] = torch.tensor([2, 1, 2, 1])
    return tokens, permuted


class EntityExportTests(unittest.TestCase):
    """One export (CPU dynamic + fixed GPU graph) shared by every check: tracing dominates test time."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.learner = Learner(small_spec(), lr=1e-3, device='cpu')
        cls.model = cls.learner.net.eval()
        cls.tokens = combat_tokens()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        train_once(cls.learner, cls.tokens)  # distinct trained weights let later exports reuse the graph
        cls.exporter = ModelExporter(cls.model.spec, ExportSpec(tokens=10, attended=6), reuse=True)
        cls.exporter.export(cls.model, cls.directory, batch=2, bank=())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_exports_match(self):
        self.model.eval()
        for filename, inputs in [('model.onnx', self.tokens[:1]), ('model-gpu.onnx', self.tokens.half())]:
            with self.subTest(artifact=filename):
                torch.testing.assert_close(onnx_predictions(self.directory / filename, inputs),
                                           model_predictions(self.model, inputs.float()), rtol=2e-4, atol=2e-5)

    def test_targets_survive_export_without_learning_monster_order(self):
        tokens, permuted = target_tokens()
        path = self.directory / 'model.onnx'
        actual = onnx_predictions(path, tokens)
        torch.testing.assert_close(actual, model_predictions(self.model, tokens), rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(onnx_predictions(path, permuted), actual, rtol=2e-4, atol=2e-5)
        _, logits = actual
        self.assertGreater(abs((logits[0, 3] - logits[0, 4]).item()), 1e-5)
        self.assertGreater(abs((logits[0, 5] - logits[0, 6]).item()), 1e-5)

    def test_exports_follow_the_current_weights(self):
        self.assert_exports_match()
        train_once(self.learner, self.tokens)
        self.exporter.export(self.model, self.directory, batch=2, bank=())  # the graph-reuse path
        self.assert_exports_match()

    def test_attention_overflow_is_nan_so_the_worker_falls_back_to_the_full_model(self):
        values, _ = deployment_view(self.model, ExportSpec(tokens=10, attended=1))(self.tokens)
        self.assertTrue(torch.isnan(values).all())
        self.assertTrue(torch.isfinite(self.model(self.tokens)[0]).all())


if __name__ == '__main__':
    unittest.main()
