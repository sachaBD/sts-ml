"""PyTorch to ONNX: writes the trained network as the model directory the player (nn_play) loads. Training models
are never moved or mutated.

Tracing a graph takes ~20s a model, so after the first export each graph is reused: the current weights are written
into the same places in the graph and its data file. That is only done when every parameter maps to exactly one
initializer (equal to it or its transpose); otherwise that file is traced again, so reuse never guesses.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

from agents.combat.nn.model import CombatModel, ModelSpec, model_class

# The model directory's description of itself: everything nn_play needs to know about the network.
MANIFEST = 'network.json'


@dataclass(frozen=True)
class ExportSpec:
    """One fixed-shape GPU deployment: its token capacity, optional attention cap and batch sizes."""
    tokens: int
    batches: tuple[int, ...]
    attended: int | None = None
    mixed_precision: bool = False

    def validate(self, model: ModelSpec):
        if self.tokens <= 0:
            raise ValueError('gpu_tokens must be positive')
        if not self.batches or len(set(self.batches)) != len(self.batches) or min(self.batches) <= 0:
            raise ValueError('distinct positive batch sizes required')
        if self.attended is not None:
            if not model_class(model.kind).bounded_attention:
                raise ValueError(f'gpu_attended is not supported by {model.kind} models')
            if not 0 < self.attended <= self.tokens:
                raise ValueError('0 < gpu_attended <= gpu_tokens required')

    def files(self, directory=''):
        """Its ONNX files, one per batch size; the largest batch is model-gpu.onnx."""
        return [Path(directory, 'model-gpu.onnx' if b == max(self.batches) else f'model-gpu-b{b}.onnx')
                for b in self.batches]


class HalfInput(nn.Module):
    """fp16 transport, fp32 outputs; optional CUDA autocast fixes compute precision."""
    def __init__(self, net, mixed_precision):
        super().__init__()
        self.net, self.mixed_precision = net, mixed_precision

    def forward(self, tokens):
        with torch.autocast('cuda', torch.float16, enabled=self.mixed_precision):
            value, logits = self.net(tokens.float())
        return value.float(), logits.float()


class ModelExporter:
    """Exports one architecture as a model directory nn_play plays from: the dynamic CPU model, the fixed GPU
    models of the narrow deployment and, optionally, a wide one (in wide/) for inputs too long for narrow,
    and MANIFEST naming them. Precision and capacity are deployment choices, not checkpoint architecture.
    """
    def __init__(self, model: ModelSpec, narrow: ExportSpec, wide: ExportSpec | None = None):
        for deployment in (narrow, wide or narrow):
            deployment.validate(model)
        self.model, self.narrow, self.wide = model, narrow, wide
        self.cache = GraphCache()

    def export(self, net: CombatModel, directory):
        self._export(net, Path(directory), dynamic=True, narrow=True)

    def export_wide(self, net: CombatModel, directory):
        """Re-exports only the wide deployment, which may change when a run resumes."""
        self._export(net, Path(directory), dynamic=False, narrow=False)

    def _export(self, net, directory, dynamic, narrow):
        if net.spec != self.model:
            raise ValueError('export architecture differs from model')
        deployments = [(directory, self.narrow)] if narrow else []
        if self.wide:
            (directory / 'wide').mkdir(exist_ok=True)
            deployments.append((directory / 'wide', self.wide))
        snapshot = deepcopy(net).cpu().eval()
        try:
            if dynamic:
                self.cache.write(snapshot, directory / 'model.onnx', 'model.onnx',
                                 lambda: self._trace(snapshot, directory / 'model.onnx', None, None))
            for folder, deployment in deployments:
                snapshot.to('cuda' if deployment.mixed_precision else 'cpu')
                for batch, path in zip(deployment.batches, deployment.files(folder)):
                    self.cache.write(snapshot, path, str(path.relative_to(directory)),
                                     lambda: self._trace(snapshot, path, deployment, batch))
        finally:
            del snapshot
            if next(net.parameters()).is_cuda or self.narrow.mixed_precision:
                torch.cuda.empty_cache()  # release export/training scratch before the native worker
        (directory / MANIFEST).write_text(json.dumps(self.manifest(net), indent=2) + '\n')

    def manifest(self, net):
        def gpu(deployment, folder):
            return dict(files=[str(f) for f in deployment.files(folder)], attended=deployment.attended or 0)
        network = dict(encoding=net.encoding, cpu='model.onnx', gpu=gpu(self.narrow, ''))
        if self.wide:
            network['wide'] = gpu(self.wide, 'wide')
        return network

    def _trace(self, net, path, deployment, batch):
        options = dict(dynamo=True, input_names=['tokens'], output_names=['value', 'logits'])
        if deployment is None:
            example = net.example_tokens()
            options['dynamic_shapes'] = {'tokens': {0: torch.export.Dim('batch'), 1: torch.export.Dim('tokens')}}
            model = net
        else:
            device = next(net.parameters()).device
            example = net.example_tokens(batch, deployment.tokens).to(device=device, dtype=torch.float16)
            model = HalfInput(net.deployed(deployment.attended), deployment.mixed_precision).eval()
        torch.onnx.export(model, (example,), str(path), **options)


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


class GraphCache:
    def __init__(self):
        self.templates = {}  # caller-owned graph identity -> template

    def write(self, net, path, key, trace):
        """Write current weights, tracing when no exact reusable template exists."""
        path = Path(path)
        weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        if key not in self.templates:
            trace()
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
