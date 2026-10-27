"""export/export_static that run the ONNX exporter once per shape, then only rewrite the weights.

After the first (normal) export, each parameter is located as the one initializer equal to it or its transpose. If any
parameter is not found exactly once (e.g. the exporter merged equal untrained weights), that export is not reused and
the next call tries again. Later calls write the current weights at the same places in the graph and data file, which
reproduces the exporter's files byte-for-byte (test_graphs.py). Files are written in place, like the exporter's.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from .model import export, export_static


def template(path, weights):
    """(graph, data file bytes, [(initializer, parameter, transposed)]) or None."""
    model = onnx.load(str(path), load_external_data=False)
    data = Path(f'{path}.data')
    mapping = []
    for init in model.graph.initializer:
        value = numpy_helper.to_array(init, base_dir=str(path.parent))
        found = [(k, t) for k, w in weights.items() for t in (False, True)
                 if value.dtype == w.dtype and (w.ndim == 2 or not t) and np.array_equal(value, w.T if t else w)]
        if len(found) > 1:
            return None
        mapping += [(init.name, *found[0])] if found else []
    if sorted(k for _, k, _ in mapping) != sorted(weights):
        return None
    return model, data.read_bytes() if data.exists() else b'', mapping


class Exporter:
    def __init__(self):
        self.templates = {}  # (batch, tokens) -> template

    def export(self, net, path):
        self.export_static(net, path, None, None)

    def export_static(self, net, path, batch, tokens):
        path, key = Path(path), (batch, tokens)
        weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        if key not in self.templates:
            export(net, path) if batch is None else export_static(net, path, batch, tokens)
            if found := template(path, weights):
                self.templates[key] = found
            return
        graph, payload, mapping = self.templates[key]
        model, payload, inits = onnx.ModelProto(), bytearray(payload), {}
        model.CopyFrom(graph)
        for init in model.graph.initializer:
            inits[init.name] = init
            for entry in init.external_data:
                entry.value = f'{path.name}.data' if entry.key == 'location' else entry.value
        for name, param, transposed in mapping:
            raw = np.ascontiguousarray(weights[param].T if transposed else weights[param]).tobytes()
            offset = next((int(e.value) for e in inits[name].external_data if e.key == 'offset'), 0)
            if inits[name].external_data:
                payload[offset:offset + len(raw)] = raw
            else:
                inits[name].raw_data = raw
        if payload:
            Path(f'{path}.data').write_bytes(payload)
        path.write_bytes(model.SerializeToString())
