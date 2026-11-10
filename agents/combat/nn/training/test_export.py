"""Deployment must preserve predictions, current weights and the training model."""
import os
from pathlib import Path
import tempfile
import unittest

import onnxruntime as ort
import torch

from agents.combat.nn._test_support import combat_tokens, small_spec, train_once
from agents.combat.nn.training.export import ExportSpec, ModelExporter
from agents.combat.nn.training.learner import Learner
from agents.combat.nn.model import ModelSpec, build_model
from agents.combat.nn.tokens import WIDTH


def onnx_predictions(path, tokens):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    return tuple(torch.from_numpy(output) for output in session.run(None, {'tokens': tokens.numpy()}))


def model_predictions(model, tokens):
    with torch.no_grad():
        return tuple(output.cpu() for output in model(tokens))


class ExportContract:
    def setUp(self):
        torch.set_num_threads(1)
        self.learner = Learner(small_spec(self.kind), lr=1e-3, device='cpu')
        self.model = self.learner.net.eval()
        self.tokens = combat_tokens()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        capacity = 6 if self.kind == 'entity' else None
        self.deployment = ExportSpec(tokens=10, attended=capacity)
        self.exporter = ModelExporter(self.model.spec, self.deployment, reuse=True)

    def test_exported_predictions_follow_current_weights(self):
        for _ in range(3):
            train_once(self.learner, self.tokens)
            self.exporter.export(self.model, self.directory, batch=2, bank=(1,))
        self.model.eval()

        artifacts = [('model.onnx', self.tokens[:1]),
                     ('model-gpu.onnx', self.tokens.half()),
                     ('model-gpu-b1.onnx', self.tokens[:1].half())]
        for filename, inputs in artifacts:
            actual = onnx_predictions(self.directory / filename, inputs)
            expected = model_predictions(self.model, inputs.float())
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_export_does_not_change_the_training_model(self):
        expected = model_predictions(self.model, self.tokens)
        self.model.train()

        self.exporter.export(self.model, self.directory, batch=2, bank=())

        self.assertTrue(self.model.training)
        actual = model_predictions(self.model.eval(), self.tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('NN_TEST_CUDA_EXPORT'), 'explicit CUDA export check required')
    def test_cuda_export_preserves_predictions(self):
        self.model.cuda()
        deployment = ExportSpec(tokens=10, attended=self.deployment.attended, mixed_precision=True)
        exporter = ModelExporter(self.model.spec, deployment)

        exporter.export(self.model, self.directory, batch=2, bank=())
        actual = onnx_predictions(self.directory / 'model-gpu.onnx', self.tokens.half())
        expected = model_predictions(self.model, self.tokens.half().float().cuda())

        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-3)


class TransformerExportTests(ExportContract, unittest.TestCase):
    kind = 'entity'


class PooledTransformerExportTests(ExportContract, unittest.TestCase):
    kind = 'entity'

    def test_attention_overflow_requires_the_full_cpu_model(self):
        exporter = ModelExporter(self.model.spec, ExportSpec(tokens=10, attended=1))

        exporter.export(self.model, self.directory, batch=2, bank=())
        gpu_values, _ = onnx_predictions(self.directory / 'model-gpu.onnx', self.tokens.half())
        cpu_predictions = onnx_predictions(self.directory / 'model.onnx', self.tokens)

        self.assertTrue(torch.isnan(gpu_values).all())
        torch.testing.assert_close(cpu_predictions, model_predictions(self.model, self.tokens), rtol=2e-4, atol=2e-5)


class TargetAssociationExportTests(unittest.TestCase):
    def test_targets_survive_export_without_learning_monster_order(self):
        torch.set_num_threads(1)
        # Same species, different instances; identical source card/potion action features.
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

        for kind in ('entity',):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(7)
                model = build_model(ModelSpec(kind, 16, 1, 2)).eval()
                directory = Path(temporary)
                ModelExporter(model.spec, ExportSpec(tokens=8, attended=8 if kind == 'entity' else None)).export(
                    model, directory, batch=1, bank=())
                for filename, dtype in [('model.onnx', torch.float32), ('model-gpu.onnx', torch.float16)]:
                    path = directory / filename
                    inputs = tokens.to(dtype)
                    expected = model_predictions(model, tokens)
                    actual = onnx_predictions(path, inputs)
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                    torch.testing.assert_close(onnx_predictions(path, permuted.to(dtype)), actual,
                                               rtol=2e-4, atol=2e-5)
                    _, logits = actual
                    self.assertGreater(abs((logits[0, 3] - logits[0, 4]).item()), 1e-5)
                    self.assertGreater(abs((logits[0, 5] - logits[0, 6]).item()), 1e-5)


if __name__ == '__main__':
    unittest.main()
