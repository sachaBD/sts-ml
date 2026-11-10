"""An exported model directory describes itself, and every model it names evaluates as the network does."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnxruntime
import torch

from agents.combat.nn.model import ModelSpec, build_model
from agents.combat.nn.pytorch_to_onnx import MANIFEST, Cpu, Deployment, ModelExporter


class ExportedDirectory(unittest.TestCase):
    def test_the_manifest_names_models_that_match_the_network(self):
        torch.manual_seed(0)
        net = build_model(ModelSpec('entity', dim=8, layers=1, heads=1, value='outcome', auxiliary=True)).eval()
        deployments = (Deployment(6, (4, 2), lanes=2), Deployment(12, (2,), lanes=1))
        exporter = ModelExporter(net.spec, deployments, Cpu(lanes=1, batch=4), mixed_precision=False)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            exporter.export(net, directory)
            with torch.no_grad():  # a second export reuses the traced graphs with new weights
                for parameter in net.parameters():
                    parameter.add_(0.1)
            exporter.export(net, directory)

            manifest = json.loads((directory / MANIFEST).read_text())
            self.assertEqual((directory / manifest['layout']).read_text(), net.layout.path.read_text())
            self.assertEqual(len(manifest['deployments']), len(deployments))
            evaluated = [(manifest['cpu']['file'], net.example_tokens(3, 20))]  # longer than every deployment
            for described, deployment in zip(manifest['deployments'], deployments):
                self.assertEqual(len(described['files']), len(deployment.batches))
                evaluated += [(file, net.example_tokens(batch, deployment.tokens).half())
                              for file, batch in zip(described['files'], deployment.batches)]
            for file, tokens in evaluated:
                session = onnxruntime.InferenceSession(str(directory / file), providers=['CPUExecutionProvider'])
                with torch.no_grad():
                    expected = net(tokens.float())
                for actual, value in zip(session.run(None, {'tokens': tokens.numpy()}), expected):
                    np.testing.assert_allclose(actual, value.numpy(), rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
