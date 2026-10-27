"""Exporter writes the same files as the ONNX exporter for every weight snapshot."""
from pathlib import Path
import tempfile
import unittest

import torch

from agents.combat.nn.graphs import Exporter
from agents.combat.nn.model import Net, export, export_static


class ExporterTests(unittest.TestCase):
    def test_matches_full_export(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        net, exporter = Net(dim=16, layers=2, heads=2), Exporter()
        with tempfile.TemporaryDirectory() as tmp:
            for snapshot in range(4):  # 0: untrained (merged equal weights, not reusable), 1: reused from here on
                for batch in (None, 4):
                    out, fresh = Path(tmp) / f'{snapshot}-{batch}', Path(tmp) / f'{snapshot}-{batch}-fresh'
                    out.mkdir(), fresh.mkdir()
                    if batch is None:
                        exporter.export(net, out / 'model.onnx'), export(net, fresh / 'model.onnx')
                    else:
                        exporter.export_static(net, out / 'model.onnx', 4, 8), export_static(net, fresh / 'model.onnx', 4, 8)
                    for name in ('model.onnx', 'model.onnx.data'):
                        self.assertEqual((out / name).read_bytes(), (fresh / name).read_bytes(), (snapshot, batch, name))
                self.assertEqual(len(exporter.templates), 0 if snapshot == 0 else 2)
                with torch.no_grad():  # a different change to every parameter
                    for p in net.parameters():
                        p.add_(torch.randn(p.shape) * 0.1)


if __name__ == '__main__':
    unittest.main()
