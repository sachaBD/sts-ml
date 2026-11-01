"""Export parity, safe capacity limits and preservation of the training model."""
from pathlib import Path
import tempfile
import os
import unittest
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from .export import ExportSpec, ModelExporter, deployment_view
from .model import ModelSpec, build_model
from .test_model import tokens


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


class ExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_exports_follow_weights_and_match_full_model_for_both_topologies(self):
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                net = build_model(ModelSpec(kind, 16, 1, 2)).train()
                deployment = ExportSpec(16, attended=6 if kind == 'entity' else None)
                exporter = ModelExporter(net.spec, deployment, reuse=True)
                for snapshot in range(3):
                    out = Path(tmp) / str(snapshot)
                    out.mkdir()
                    before = {k: v.clone() for k, v in net.state_dict().items()}
                    exporter.export(net, out, 2, (1,))
                    self.assertTrue(net.training)
                    self.assertEqual(next(net.parameters()).device.type, 'cpu')
                    for name, weight in net.state_dict().items():
                        torch.testing.assert_close(weight, before[name], rtol=0, atol=0)
                    net.eval()
                    x = tokens()
                    for filename, count, dtype in [('model.onnx', 2, np.float32),
                                                   ('model-gpu.onnx', 2, np.float16),
                                                   ('model-gpu-b1.onnx', 1, np.float16)]:
                        actual_input = x[:count].numpy().astype(dtype)
                        expected = net(torch.from_numpy(actual_input).float())
                        actual = session(out / filename).run(None, {'tokens': actual_input})
                        for a, e in zip(actual, expected):
                            np.testing.assert_allclose(a, e.detach().numpy(), rtol=2e-4, atol=2e-5)
                    # The CPU model must remain dynamic in both axes.
                    short = x[:1, :10]
                    actual = session(out / 'model.onnx').run(None, {'tokens': short.numpy()})
                    for a, e in zip(actual, net(short)):
                        np.testing.assert_allclose(a, e.detach().numpy(), rtol=2e-4, atol=2e-5)
                    with torch.no_grad():
                        for parameter in net.parameters():
                            parameter.add_(0.02 * torch.randn_like(parameter))
                    net.train()

    def test_capacity_overflow_is_explicit_and_full_model_remains_valid(self):
        net = build_model(ModelSpec('entity', 16, 1, 2)).eval()
        x = tokens()
        full = net(x)
        bounded = deployment_view(net, ExportSpec(16, attended=6))
        torch.testing.assert_close(full, bounded(x))
        overflow = deployment_view(net, ExportSpec(16, attended=5))
        self.assertTrue(torch.isnan(overflow(x)[0]).all())
        self.assertTrue(torch.isfinite(net(x)[0]).all())
        with tempfile.TemporaryDirectory() as tmp:
            ModelExporter(net.spec, ExportSpec(16, attended=5)).export(net, Path(tmp), 2, ())
            gpu = session(Path(tmp) / 'model-gpu.onnx').run(None, {'tokens': x.numpy().astype(np.float16)})
            cpu = session(Path(tmp) / 'model.onnx').run(None, {'tokens': x.numpy()})
            self.assertTrue(np.isnan(gpu[0]).all())
            self.assertTrue(np.isfinite(cpu[0]).all())

    def test_failed_export_cannot_mutate_training_model(self):
        net = build_model(ModelSpec('entity', 16, 1, 2)).train()
        x = tokens()
        before = net(x)
        with tempfile.TemporaryDirectory() as tmp, patch('torch.onnx.export', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                ModelExporter(net.spec, ExportSpec(16)).export(net, Path(tmp), 2, ())
        self.assertTrue(net.training)
        torch.testing.assert_close(before, net(x), rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('NN_TEST_CUDA_EXPORT'), 'explicit CUDA export check required')
    def test_mixed_precision_export_preserves_live_cuda_model_and_predictions(self):
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                net = build_model(ModelSpec(kind, 16, 1, 2)).cuda().train()
                x = tokens().cuda()
                before = {k: v.clone() for k, v in net.state_dict().items()}
                spec = ExportSpec(16, attended=6 if kind == 'entity' else None, mixed_precision=True)
                ModelExporter(net.spec, spec, reuse=True).export(net, Path(tmp), 2, ())
                self.assertTrue(net.training)
                self.assertTrue(next(net.parameters()).is_cuda)
                for name, weight in net.state_dict().items():
                    torch.testing.assert_close(weight, before[name], rtol=0, atol=0)
                with torch.autocast('cuda', torch.float16):
                    expected = net(x.half().float())
                actual = session(Path(tmp) / 'model-gpu.onnx').run(None, {'tokens': x.cpu().numpy().astype(np.float16)})
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a, e.detach().float().cpu().numpy(), rtol=3e-3, atol=2e-3)

    def test_invalid_deployment_is_rejected_before_export(self):
        for kind, spec in [('transformer', ExportSpec(16, attended=5)),
                           ('entity', ExportSpec(16, attended=0)),
                           ('entity', ExportSpec(16, attended=17)), ('entity', ExportSpec(0))]:
            with self.subTest(kind=kind, spec=spec), self.assertRaises(ValueError):
                ModelExporter(ModelSpec(kind, 16, 1, 2), spec)


if __name__ == '__main__':
    unittest.main()
