"""Component tests of deployed predictions, not ONNX internals or tracing calls."""
import os
from pathlib import Path
import tempfile
import unittest

import onnxruntime as ort
import torch

from ._test_support import combat_tokens, small_spec, train_once
from .export import ExportSpec, ModelExporter
from .learner import Learner
from .tokens import TokenKind, WIDTH


def onnx_predictions(path, tokens):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    values, logits = session.run(None, {'tokens': tokens.numpy()})
    return torch.from_numpy(values), torch.from_numpy(logits)


def model_predictions(model, tokens):
    with torch.no_grad():
        values, logits = model(tokens)
    return values.cpu(), logits.cpu()


class ExportContract:
    def setUp(self):
        torch.set_num_threads(1)
        self.learner = Learner(small_spec(self.kind), lr=1e-3, device='cpu')
        self.model = self.learner.net.eval()
        self.tokens = combat_tokens()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.exporter = ModelExporter(self.model.spec, ExportSpec(tokens=10), reuse=True)

    def export_models(self, directory=None):
        self.exporter.export(self.model, directory or self.directory, batch=2, bank=(1,))

    def test_cpu_export_matches_the_model(self):
        self.export_models()

        actual = onnx_predictions(self.directory / 'model.onnx', self.tokens)
        expected = model_predictions(self.model, self.tokens)

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_cpu_export_accepts_a_different_batch_and_sequence_length(self):
        self.export_models()
        padding = torch.zeros(1, 4, WIDTH)
        different_shape = torch.cat([self.tokens[:1], padding], dim=1)

        actual = onnx_predictions(self.directory / 'model.onnx', different_shape)
        expected = model_predictions(self.model, different_shape)

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_fixed_export_matches_the_model_on_half_precision_inputs(self):
        self.export_models()
        inputs = self.tokens.half()

        actual = onnx_predictions(self.directory / 'model-gpu.onnx', inputs)
        expected = model_predictions(self.model, inputs.float())

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_smaller_bank_model_matches_the_same_weights(self):
        self.export_models()
        inputs = self.tokens[:1].half()

        actual = onnx_predictions(self.directory / 'model-gpu-b1.onnx', inputs)
        expected = model_predictions(self.model, inputs.float())

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def export_after_training_updates(self):
        for update in range(3):
            train_once(self.learner, self.tokens)
            directory = self.directory / f'update-{update}'
            directory.mkdir()
            self.export_models(directory)
        self.model.eval()
        return directory

    def test_repeated_cpu_exports_follow_training_updates(self):
        directory = self.export_after_training_updates()

        actual = onnx_predictions(directory / 'model.onnx', self.tokens)
        expected = model_predictions(self.model, self.tokens)

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_repeated_fixed_exports_follow_training_updates(self):
        directory = self.export_after_training_updates()
        inputs = self.tokens.half()

        actual = onnx_predictions(directory / 'model-gpu.onnx', inputs)
        expected = model_predictions(self.model, inputs.float())

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_repeated_bank_exports_follow_training_updates(self):
        directory = self.export_after_training_updates()
        inputs = self.tokens[:1].half()

        actual = onnx_predictions(directory / 'model-gpu-b1.onnx', inputs)
        expected = model_predictions(self.model, inputs.float())

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_export_leaves_the_training_model_usable(self):
        expected = model_predictions(self.model, self.tokens)
        self.model.train()

        self.export_models()

        self.assertTrue(self.model.training)
        actual = model_predictions(self.model.eval(), self.tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_failed_export_leaves_the_training_model_usable(self):
        expected = model_predictions(self.model, self.tokens)
        self.model.train()
        blocked = self.directory / 'not-a-directory'
        blocked.write_text('a file cannot contain exported models')

        with self.assertRaises(OSError):
            self.export_models(blocked)

        self.assertTrue(self.model.training)
        actual = model_predictions(self.model.eval(), self.tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('NN_TEST_CUDA_EXPORT'), 'explicit CUDA export check required')
    def test_cuda_mixed_precision_export_matches_full_precision_predictions(self):
        self.model.cuda()
        exporter = ModelExporter(self.model.spec, ExportSpec(tokens=10, mixed_precision=True))

        exporter.export(self.model, self.directory, batch=2, bank=())
        actual = onnx_predictions(self.directory / 'model-gpu.onnx', self.tokens.half())
        expected = model_predictions(self.model, self.tokens.half().float().cuda())

        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-3)

    @unittest.skipUnless(os.environ.get('NN_TEST_CUDA_EXPORT'), 'explicit CUDA export check required')
    def test_cuda_export_leaves_the_training_model_usable(self):
        self.model.cuda()
        inputs = self.tokens.cuda()
        expected = model_predictions(self.model, inputs)
        self.model.train()
        exporter = ModelExporter(self.model.spec, ExportSpec(tokens=10, mixed_precision=True))

        exporter.export(self.model, self.directory, batch=2, bank=())

        self.assertTrue(self.model.training)
        actual = model_predictions(self.model.eval(), inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TransformerExportTests(ExportContract, unittest.TestCase):
    kind = 'transformer'


class PooledTransformerExportTests(ExportContract, unittest.TestCase):
    kind = 'entity'


class PooledCapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = Learner(small_spec('entity'), lr=1e-3, device='cpu').net.eval()
        self.tokens = torch.zeros(1, 8, WIDTH)
        self.tokens[:, 0, 0] = TokenKind.CONTEXT
        self.tokens[:, 1:4, 0] = TokenKind.ACTION  # four retained tokens: context + three actions
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_bounded_export_matches_full_model_when_all_actions_fit(self):
        exporter = ModelExporter(self.model.spec, ExportSpec(tokens=8, attended=4))

        exporter.export(self.model, self.directory, batch=1, bank=())
        actual = onnx_predictions(self.directory / 'model-gpu.onnx', self.tokens.half())
        expected = model_predictions(self.model, self.tokens)

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_overflow_is_rejected_instead_of_silently_dropping_actions(self):
        exporter = ModelExporter(self.model.spec, ExportSpec(tokens=8, attended=3))

        exporter.export(self.model, self.directory, batch=1, bank=())
        values, _ = onnx_predictions(self.directory / 'model-gpu.onnx', self.tokens.half())

        self.assertTrue(torch.isnan(values).all())

    def test_gpu_capacity_does_not_limit_the_cpu_fallback(self):
        exporter = ModelExporter(self.model.spec, ExportSpec(tokens=8, attended=3))

        exporter.export(self.model, self.directory, batch=1, bank=())
        actual = onnx_predictions(self.directory / 'model.onnx', self.tokens)
        expected = model_predictions(self.model, self.tokens)

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
