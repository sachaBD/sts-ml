"""Transformer over a decision's tokens -> (win probability, one logit per token; only action tokens are legal).

Token = [type, id, id2, numeric x 65] (agents/combat/nn/tokens.hpp); type 0 = padding, 1 = context (first), 6 = action.
"""
import torch
from torch import nn

WIDTH = 68
ACTION = 6


class Net(nn.Module):
    def __init__(self, dim=64, layers=2, heads=2):
        super().__init__()
        if dim <= 0 or layers <= 0 or heads <= 0 or dim % heads:
            raise ValueError('positive dim/layers/heads required; dim must be divisible by heads')
        self.spec = dict(dim=dim, layers=layers, heads=heads)
        self.kind = nn.Embedding(8, dim)
        self.id = nn.Embedding(8 * 1024, dim)   # per type: type * 1024 + id
        self.id2 = nn.Embedding(1024, dim)
        self.numeric = nn.Linear(WIDTH - 3, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, 1)
        self.policy = nn.Linear(dim, 1)

    def forward(self, tokens):
        kind = tokens[..., 0].long()
        x = (self.kind(kind) + self.id(kind * 1024 + tokens[..., 1].long()) + self.id2(tokens[..., 2].long())
             + self.numeric(tokens[..., 3:]))
        # Masks use < / > on the float type column: ONNX Runtime's CUDA provider runs no Equal here (CPU fallback).
        h = self.norm(self.encoder(x, src_key_padding_mask=tokens[..., 0] < 0.5))
        value = torch.sigmoid(self.value(h[:, 0]).squeeze(-1))
        logits = self.policy(h).squeeze(-1).masked_fill((tokens[..., 0] - ACTION).abs() > 0.5, -1e9)
        return value, logits


def export(net, path):
    """ONNX: tokens [B, T, WIDTH] -> value [B], logits [B, T]."""
    net = net.cpu().eval()
    example = torch.zeros(2, 3, WIDTH)
    example[:, 0, 0], example[:, 1:, 0] = 1, ACTION
    torch.onnx.export(net, (example,), str(path), dynamo=True, input_names=['tokens'], output_names=['value', 'logits'],
                      dynamic_shapes={'tokens': {0: torch.export.Dim('batch'), 1: torch.export.Dim('tokens')}})


class HalfInput(nn.Module):
    """Half-precision transport; TensorRT chooses internal precision, outputs remain float32."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, tokens):
        return self.net(tokens.float())


def export_static(net, path, batch, tokens, attended=None):
    """Fixed-shape GPU model: fp16 tokens [batch, tokens, WIDTH], fp32 value/logits."""
    if getattr(net, 'spec', {}).get('kind') == 'entity':
        return export_entity_static(net, path, batch, tokens, attended)
    net = HalfInput(net.cpu().eval()).eval()
    example = torch.zeros(batch, tokens, WIDTH, dtype=torch.float16)
    torch.onnx.export(net, (example,), str(path), dynamo=True,
                      input_names=['tokens'], output_names=['value', 'logits'])


class MixedInput(nn.Module):
    """fp16 transport and fp16 compute written into the graph (autocast: fp16 matmuls, fp32 norms/softmax).
    Left to TensorRT, the precision choice is per shape and sometimes keeps whole GEMMs in fp32 (about 2x slower).
    Outputs remain float32."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, tokens):
        with torch.autocast('cuda', torch.float16):
            value, logits = self.net(tokens.float())
        return value.float(), logits.float()


def export_entity_static(net, path, batch, tokens, attended):
    """Fixed-shape GPU model: fp16 tokens [batch, tokens, WIDTH], fp32 value/logits.
    attended: compacted attention slots for an EntityNet (entity.py); ignored by Net."""
    device = next(net.parameters()).device
    net = net.cuda().eval()
    previous = getattr(net, 'attended', None)
    if hasattr(net, 'attended'):
        net.attended = attended
    try:
        example = torch.zeros(batch, tokens, WIDTH, dtype=torch.float16, device='cuda')
        torch.onnx.export(MixedInput(net).eval(), (example,), str(path), dynamo=True,
                          input_names=['tokens'], output_names=['value', 'logits'])
    finally:
        if hasattr(net, 'attended'):
            net.attended = previous
        net.to(device)
