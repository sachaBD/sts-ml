"""MCTS bootstrap learner. Reads shards produced by pv.data; saves a PyTorch checkpoint and ONNX model.

Validation membership is a stable hash of run seed, shared across datasets/iterations. Policy: search visits;
value: 100 × won (scaled MSE). These losses are diagnostics, not a replacement for held-out play evaluation.
Per epoch it logs train/val value loss and policy loss separately, and val top-1 agreement with the search's most-visited move.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from .data import Dataset
from .model import CONTRACT, NAMES, VALUE_SCALE, PolicyValue


def evaluate(net, batch, device, policy_temp=1.0, value_mix=1.0, teacher_root_mix=1.0):
    """Return (value loss sum, policy loss sum, top-1 hits, states, policy states) of one batch; grads flow."""
    inputs, target_value, target, has_policy = (
        {k: v.to(device) for k, v in batch[0].items()}, batch[1].to(device), batch[2].to(device), batch[3].to(device))
    if teacher_root_mix != 1.0:
        raise ValueError('teacher root mixing uses old HP units and is incompatible with the win-only contract')
    if not 0 <= value_mix <= 1:
        raise ValueError('value-mix must be in [0, 1]')
    if value_mix != 1.0:  # PV search roots and outcomes both use 100 × win-probability units.
        root = batch[4].to(device)
        searched = ~torch.isnan(root)  # forced moves have no search: outcome only
        if not torch.isfinite(root[searched]).all() or ((root[searched] < 0) | (root[searched] > 100)).any():
            raise ValueError('value mixing needs PV search root values in [0, 100]')
        target_value = torch.where(searched, value_mix * target_value + (1 - value_mix) * root, target_value)
    value, logits = net(*(inputs[n] for n in NAMES))
    if not torch.isfinite(value).all() or not torch.isfinite(logits).all():
        raise ValueError('network produced non-finite predictions')
    if not torch.isfinite(target_value).all() or (target_value < 0).any():
        raise ValueError('value targets must be finite, nonnegative win-score units')
    if (target < 0).any() or not torch.allclose(target.sum(-1)[has_policy], torch.ones((), device=device), atol=1e-5):
        raise ValueError('policy targets must be nonnegative and sum to one')
    if policy_temp != 1.0:  # sharpen flat (e.g. UCB1 teacher) visit distributions: p ∝ visits^(1/T)
        target = target.pow(1 / policy_temp)
        target = target / target.sum(-1, keepdim=True).clamp(min=1e-12)
    # Balance the two losses without changing the units of predictions or targets.
    value_loss = ((value - target_value) / VALUE_SCALE).square().sum()
    policy_loss = (-(target * logits.log_softmax(-1)).sum(-1) * has_policy).sum()
    hits = ((logits.argmax(-1) == target.argmax(-1)) & has_policy).sum().item()
    return value_loss, policy_loss, hits, len(value), has_policy.sum().item()


def run_epoch(net, data, a, optimizer=None, rng=None):
    totals = np.zeros(4)  # value, policy, hits, states; policy states separately
    policy_states = 0
    for batch in data.batches(a.batch, rng):
        with torch.set_grad_enabled(optimizer is not None):
            value, policy, hits, n, n_policy = evaluate(net, batch, a.device, a.policy_temp, a.value_mix, a.teacher_root_mix)
            loss = value / n + policy / max(n_policy, 1)
            if not torch.isfinite(loss): raise ValueError('non-finite training loss')
            if optimizer:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        totals += (value.item(), policy.item(), hits, n); policy_states += n_policy
    if not totals[3]: raise ValueError('no states in this split (need more independent run seeds)')
    return dict(value_loss=totals[0] / totals[3], policy_loss=totals[1] / max(policy_states, 1),
                top1=totals[2] / max(policy_states, 1), states=int(totals[3]), policy_states=policy_states)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, nargs='+', required=True, help='shard files (rows.parquet)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--init', type=Path)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--policy-temp', type=float, default=1.0, help='policy target temperature (<1 sharpens visits)')
    p.add_argument('--value-mix', type=float, default=1.0,
                   help='mix·100·won + (1-mix)·PV search root in [0,100]; no-root rows keep outcome')
    p.add_argument('--teacher-root-mix', type=float, default=1.0,
                   help='must remain 1: old HP-unit teacher root mixing is unsupported under the win-only contract')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--stream', action='store_true', help='re-read one shard at a time per epoch instead of holding all in RAM')
    a = p.parse_args()
    if not 0 <= a.value_mix <= 1:
        p.error('value-mix must be in [0, 1]')
    if a.teacher_root_mix != 1:
        p.error('teacher root mixing uses old HP units and is incompatible with the win-only contract')
    if a.batch < 1 or a.epochs < 1: p.error('batch and epochs must be positive')
    torch.manual_seed(0); torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    if a.init:
        ckpt = torch.load(a.init, map_location='cpu', weights_only=False)
        if ckpt['contract'] != CONTRACT: raise ValueError('checkpoint input contract mismatch')
        net = PolicyValue(ckpt['width']); net.load_state_dict(ckpt['state_dict'])
    else: net = PolicyValue(a.width)
    net.to(a.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    train, val = Dataset(a.data, False, a.stream), Dataset(a.data, True, a.stream)
    a.out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    example = next(train.batches(2))[0]
    for epoch in range(a.epochs):
        net.train(); tr = run_epoch(net, train, a, optimizer, rng)
        net.eval(); va = run_epoch(net, val, a)
        print(json.dumps({'epoch': epoch, 'train': tr, 'val': va}), flush=True)
        score = va['value_loss'] + va['policy_loss']
        if score < best:
            best = score
            torch.save({'contract': CONTRACT, 'width': net.width, 'state_dict': net.state_dict()}, a.out / 'model.pt')
    # One export, of the selected checkpoint, rather than recompiling ONNX every epoch.
    ckpt = torch.load(a.out / 'model.pt', map_location='cpu', weights_only=False)
    selected = PolicyValue(ckpt['width']); selected.load_state_dict(ckpt['state_dict'])
    selected.export(example, a.out / 'model.onnx')


if __name__ == '__main__': main()
