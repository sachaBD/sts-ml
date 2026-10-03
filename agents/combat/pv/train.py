"""MCTS bootstrap learner. Streams caches produced by pv.data; saves a PyTorch checkpoint and ONNX model.

Validation membership is a stable hash of run seed, shared across datasets/iterations. Policy: search visits;
value: terminal combat score. These losses are diagnostics, not a replacement for held-out play evaluation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from .data import collate
from .model import CONTRACT, NAMES, VALUE_SCALE, PolicyValue


def validation(seed):
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], 'little') % 10 == 0


def batches(paths, valid, size):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get('contract') != CONTRACT:
                    raise ValueError('cache input/value contract mismatch; regenerate it with pv_worker encode')
                if validation(row['seed']) != valid: continue
                rows.append(row)
                if len(rows) == size: yield rows; rows = []
    if rows: yield rows


def objective(net, rows, device):
    inputs = {k: v.to(device) for k, v in collate(rows).items()}
    value, logits = net(*(inputs[n] for n in NAMES))
    if not torch.isfinite(value).all() or not torch.isfinite(logits).all():
        raise ValueError('network produced non-finite predictions')
    target_value = torch.tensor([r['value_target'] for r in rows], device=device)
    if not torch.isfinite(target_value).all() or (target_value < 0).any():
        raise ValueError('value targets must be finite, nonnegative HP-equivalent points')
    # Balance the two losses without changing the units of predictions or targets.
    value_loss = ((value - target_value) / VALUE_SCALE).square().mean()
    target = torch.zeros_like(logits)
    mask = torch.tensor([r['has_policy'] for r in rows], device=device, dtype=torch.float32)
    for i, row in enumerate(rows):
        policy = row['policy_target']
        if len(policy) != len(row['moves']) or any(not math.isfinite(p) or p < 0 for p in policy):
            raise ValueError('invalid policy target')
        if row['has_policy'] and not math.isclose(sum(policy), 1, abs_tol=1e-6):
            raise ValueError('policy target must sum to one')
        target[i, :len(policy)] = torch.tensor(policy, device=device)
    policy = -(target * logits.log_softmax(-1)).sum(-1)
    policy_loss = (policy * mask).sum() / mask.sum().clamp(min=1)
    loss = value_loss + policy_loss
    if not torch.isfinite(loss): raise ValueError('non-finite training loss')
    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--init', type=Path)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.batch < 1 or a.epochs < 1: p.error('batch and epochs must be positive')
    torch.manual_seed(0); torch.set_num_threads(1)
    if a.init:
        ckpt = torch.load(a.init, map_location='cpu', weights_only=False)
        if ckpt['contract'] != CONTRACT: raise ValueError('checkpoint input contract mismatch')
        net = PolicyValue(ckpt['width']); net.load_state_dict(ckpt['state_dict'])
    else: net = PolicyValue(a.width)
    net.to(a.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    a.out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    example = None
    for epoch in range(a.epochs):
        net.train(); trained = 0
        for rows in batches(a.data, False, a.batch):
            if example is None: example = [rows[0], rows[-1]]
            optimizer.zero_grad(); loss = objective(net, rows, a.device); loss.backward(); optimizer.step()
            trained += len(rows)
        if not trained: raise ValueError('no training states (need more independent run seeds)')
        net.eval(); total = count = 0
        with torch.no_grad():
            for rows in batches(a.data, True, a.batch):
                total += objective(net, rows, a.device).item() * len(rows); count += len(rows)
        if not count: raise ValueError('no validation states (need more independent run seeds)')
        score = total / count
        print(json.dumps({'epoch': epoch, 'train_states': trained, 'val_states': count, 'val_loss': score}), flush=True)
        if score < best:
            best = score
            torch.save({'contract': CONTRACT, 'width': net.width, 'state_dict': net.state_dict()}, a.out / 'model.pt')
    # One export, of the selected checkpoint, rather than recompiling ONNX every epoch.
    ckpt = torch.load(a.out / 'model.pt', map_location='cpu', weights_only=False)
    selected = PolicyValue(ckpt['width']); selected.load_state_dict(ckpt['state_dict'])
    selected.export(collate(example), a.out / 'model.onnx')


if __name__ == '__main__': main()
