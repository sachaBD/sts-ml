"""Deployment adapters and ONNX export. Training models are never moved or mutated."""
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from agents.combat.nn.entity.model import EntityModel
from agents.combat.nn.training.graphs import GraphCache
from agents.combat.nn.model import CombatModel, ModelSpec


def model_files(directory, gpu_bank):
    """What nn_play loads from a model directory: the dynamic model, the fixed-batch GPU model and its smaller bank."""
    return dict(dynamic=directory / 'model.onnx', gpu=directory / 'model-gpu.onnx',
                bank=[directory / f'model-gpu-b{b}.onnx' for b in gpu_bank])


@dataclass(frozen=True)
class ExportSpec:
    tokens: int
    attended: int | None = None
    mixed_precision: bool = False

    def validate(self, model: ModelSpec):
        if self.tokens <= 0:
            raise ValueError('gpu_tokens must be positive')
        if self.attended is not None:
            if model.kind != 'entity':
                raise ValueError('gpu_attended is only supported for entity models')
            if not 0 < self.attended <= self.tokens:
                raise ValueError('0 < gpu_attended <= gpu_tokens required')


class PooledAttentionView(nn.Module):
    """Fixed attention capacity; the native backend retries overflow on the full CPU model."""
    def __init__(self, net: EntityModel, slots: int | None):
        super().__init__()
        self.net, self.slots = net, slots

    def forward(self, tokens):
        return self.net.predict(tokens, self.slots)


def deployment_view(net: CombatModel, spec: ExportSpec) -> nn.Module:
    """The single topology-specific deployment boundary."""
    spec.validate(net.spec)
    if isinstance(net, EntityModel):
        return PooledAttentionView(net, spec.attended)
    return net


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
    """Exports one architecture to the dynamic CPU model and fixed GPU model bank.

    Each instance owns its graph cache. Precision and capacity are deployment choices,
    not checkpoint architecture. Reuse falls back to tracing unless every weight maps
    exactly; mixed-precision graphs may therefore require full re-export.
    """
    def __init__(self, model: ModelSpec, deployment: ExportSpec, reuse=False):
        deployment.validate(model)
        self.model, self.deployment = model, deployment
        self.cache = GraphCache() if reuse else None

    def export(self, net: CombatModel, directory, batch, bank):
        if net.spec != self.model:
            raise ValueError('export architecture differs from model')
        if batch <= 0 or len(set(bank)) != len(bank) or any(not 0 < b < batch for b in bank):
            raise ValueError('positive main batch and distinct smaller bank batches required')
        files = model_files(directory, bank)
        snapshot = deepcopy(net).cpu().eval()
        try:
            self._write(snapshot, files['dynamic'], None)
            device = 'cuda' if self.deployment.mixed_precision else 'cpu'
            snapshot.to(device)
            for size, path in zip((batch, *bank), (files['gpu'], *files['bank'])):
                self._write(snapshot, path, size)
        finally:
            del snapshot
            if next(net.parameters()).is_cuda or self.deployment.mixed_precision:
                torch.cuda.empty_cache()  # release export/training scratch before the native worker

    def _write(self, net, path, batch):
        trace = lambda: self._trace(net, path, batch)
        if self.cache is None:
            trace()
        else:
            self.cache.write(net, path, batch, trace)

    def _trace(self, net, path, batch):
        options = dict(dynamo=True, input_names=['tokens'], output_names=['value', 'logits'])
        if batch is None:
            example = net.example_tokens()
            options['dynamic_shapes'] = {'tokens': {0: torch.export.Dim('batch'), 1: torch.export.Dim('tokens')}}
            model = net
        else:
            device = next(net.parameters()).device
            example = net.example_tokens(batch, self.deployment.tokens).to(device=device, dtype=torch.float16)
            model = HalfInput(deployment_view(net, self.deployment), self.deployment.mixed_precision).eval()
        torch.onnx.export(model, (example,), str(path), **options)
