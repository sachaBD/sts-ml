"""An exported model directory describes itself, and every model it names evaluates as the network does."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnxruntime
import torch

from agents.combat.nn.model import ModelSpec, build_model
from agents.combat.nn.pytorch_to_onnx import MANIFEST, ExportSpec, ModelExporter


def assert_evaluates_as(net, path, tokens):
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        expected = net(tokens.float())
    for actual, value in zip(session.run(None, {'tokens': tokens.numpy()}), expected):
        np.testing.assert_allclose(actual, value.numpy(), rtol=1e-4, atol=1e-4)


class ExportedDirectory(unittest.TestCase):
    def test_the_manifest_names_models_that_match_the_network(self):
        # The entity model caps attention in its GPU deployments: the most a deployment changes the network.
        torch.manual_seed(0)
        net = build_model(ModelSpec('entity', dim=8, layers=1, heads=1)).eval()
        narrow, wide = ExportSpec(6, (4, 2), attended=6), ExportSpec(12, (2,), attended=12)
        exporter = ModelExporter(net.spec, narrow, wide)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            exporter.export(net, directory)
            with torch.no_grad():  # a second export reuses the traced graphs with new weights
                for parameter in net.parameters():
                    parameter.add_(0.1)
            exporter.export(net, directory)

            manifest = json.loads((directory / MANIFEST).read_text())
            self.assertEqual(manifest['encoding'], net.encoding)
            assert_evaluates_as(net, directory / manifest['cpu'], net.example_tokens(3, 5))
            for name, deployment in (('gpu', narrow), ('wide', wide)):
                self.assertEqual(manifest[name]['attended'], deployment.attended)
                self.assertEqual(len(manifest[name]['files']), len(deployment.batches))
                for file, batch in zip(manifest[name]['files'], deployment.batches):
                    assert_evaluates_as(net, directory / file, net.example_tokens(batch, deployment.tokens).half())


if __name__ == '__main__':
    unittest.main()
