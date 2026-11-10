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

from agents.combat.nn.model import CombatModel, ModelSpec

# The model directory's description of itself: everything nn_play needs to know about the network.
MANIFEST = 'network.json'
CPU_MODEL = 'model.onnx'  # dynamic shape, fp32


@dataclass(frozen=True)
class Deployment:
    """Fixed-shape fp16 GPU models of one token capacity, one per batch size, served by `lanes` concurrent streams."""
    tokens: int
    batches: tuple[int, ...]
    lanes: int

    def files(self):
        return [Path(f'model-t{self.tokens}-b{batch}.onnx') for batch in self.batches]


@dataclass(frozen=True)
class Cpu:
    """CPU_MODEL, served by `lanes` threads in batches of up to `batch` inputs."""
    lanes: int
    batch: int


def validate(deployments, cpu):
    if not deployments:
        raise ValueError('at least one deployment required')
    if any(a.tokens >= b.tokens for a, b in zip(deployments, deployments[1:])) or deployments[0].tokens <= 0:
        raise ValueError('deployments: positive token capacities, strictly increasing')
    for d in deployments:
        if not d.batches or len(set(d.batches)) != len(d.batches) or min(d.batches) <= 0 or d.lanes <= 0:
            raise ValueError(f'deployment of {d.tokens} tokens: distinct positive batch sizes and positive lanes required')
    if cpu.lanes <= 0 or cpu.batch <= 0:
        raise ValueError('cpu: positive lanes and batch required')


class HalfInput(nn.Module):
    """fp16 transport, fp32 outputs; CUDA autocast when mixed_precision fixes compute precision."""
    def __init__(self, net, mixed_precision):
        super().__init__()
        self.net, self.mixed_precision = net, mixed_precision

    def forward(self, tokens):
        with torch.autocast('cuda', torch.float16, enabled=self.mixed_precision):
            value, logits = self.net(tokens.float())
        return value.float(), logits.float()


class ModelExporter:
    """Writes one architecture as a model directory nn_play plays from: every deployment's ONNX files, CPU_MODEL and
    MANIFEST describing them. Precision and capacity are deployment choices, not checkpoint architecture."""
    def __init__(self, model: ModelSpec, deployments: tuple[Deployment, ...], cpu: Cpu, mixed_precision: bool):
        validate(deployments, cpu)
        self.model, self.deployments, self.cpu, self.mixed_precision = model, deployments, cpu, mixed_precision
        self.cache = GraphCache()

    def export(self, net: CombatModel, directory: Path):
        if net.spec != self.model:
            raise ValueError('export architecture differs from model')
        snapshot = deepcopy(net).cpu().eval()
        try:
            self.cache.write(snapshot, directory / CPU_MODEL, CPU_MODEL, lambda: torch.onnx.export(
                snapshot, (snapshot.example_tokens(),), str(directory / CPU_MODEL), dynamo=True,
                input_names=['tokens'], output_names=['value', 'logits'],
                dynamic_shapes={'tokens': {0: torch.export.Dim('batch'), 1: torch.export.Dim('tokens')}}))
            snapshot.to('cuda' if self.mixed_precision else 'cpu')
            for deployment in self.deployments:
                for batch, file in zip(deployment.batches, deployment.files()):
                    self.cache.write(snapshot, directory / file, file,
                                     lambda: self._trace(snapshot, directory / file, deployment.tokens, batch))
        finally:
            del snapshot
            if next(net.parameters()).is_cuda or self.mixed_precision:
                torch.cuda.empty_cache()  # release export/training scratch before the native worker
        manifest = dict(encoding=net.encoding,
                        deployments=[dict(files=[str(f) for f in d.files()], lanes=d.lanes) for d in self.deployments],
                        cpu=dict(file=CPU_MODEL, lanes=self.cpu.lanes, batch=self.cpu.batch))
        (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')

    def _trace(self, net, path, tokens, batch):
        device = next(net.parameters()).device
        example = net.example_tokens(batch, tokens).to(device=device, dtype=torch.float16)
        torch.onnx.export(HalfInput(net, self.mixed_precision).eval(), (example,), str(path), dynamo=True,
                          input_names=['tokens'], output_names=['value', 'logits'])


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
