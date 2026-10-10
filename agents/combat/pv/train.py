"""MCTS bootstrap learner. Reads shards produced by pv.data; saves a PyTorch checkpoint and ONNX model.

Validation membership is a stable hash of run seed, shared across datasets/iterations. Policy: search visits;
value: 100 × won (scaled MSE). These losses are diagnostics, not a replacement for held-out play evaluation.
Per epoch it logs train/val losses and maximum absolute raw value prediction (win-score units),
and val top-1 agreement with the search's most-visited move. Gradient norm clipping is opt-in.
Loss weights affect optimization and checkpoint selection, not raw logged losses; defaults are both 1.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from .data import Dataset, load_split
from .model import CONTRACT, NAMES, VALUE_SCALE, PolicyValue


def policy_confidence(target, legal):
    """0 at uniform, 1 at one-hot; near-uniform targets provide proportionally little policy loss.

    This is concentration, not a statistically calibrated measure of teacher correctness.
    Count *legal* actions, including zero-visit actions, not padded output slots.
    """
    n = legal.sum(-1).to(target.dtype)
    return ((n * target.max(-1).values - 1) / (n - 1).clamp(min=1)).clamp(0, 1)


def evaluate(net, batch, device, policy_temp=1.0, value_mix=1.0, teacher_root_mix=1.0, prediction_stats=None,
             flat_policy_weighting=False):
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
        if not torch.isfinite(root[searched]).all() or (root[searched] < 0).any():
            raise ValueError('value mixing needs finite, nonnegative PV search root values')
        root = root.clamp(0, 100)  # The softplus head is unbounded; probability targets are not.
        target_value = torch.where(searched, value_mix * target_value + (1 - value_mix) * root, target_value)
    value, logits = net(*(inputs[n] for n in NAMES))
    if not torch.isfinite(value).all() or not torch.isfinite(logits).all():
        raise ValueError('network produced non-finite predictions')
    if prediction_stats is not None:
        prediction_stats['max_abs_value_prediction'] = max(
            prediction_stats.get('max_abs_value_prediction', 0.0), value.detach().abs().max().item())
    if not torch.isfinite(target_value).all() or (target_value < 0).any():
        raise ValueError('value targets must be finite, nonnegative win-score units')
    if (target < 0).any() or not torch.allclose(target.sum(-1)[has_policy], torch.ones((), device=device), atol=1e-5):
        raise ValueError('policy targets must be nonnegative and sum to one')
    confidence = has_policy.to(target.dtype)
    if flat_policy_weighting:
        legal = inputs['actions'][..., 260] != 0
        confidence = confidence * policy_confidence(target, legal)
        if prediction_stats is not None:
            prediction_stats['effective_policy_states'] = prediction_stats.get('effective_policy_states', 0.) + confidence.sum().item()
    if policy_temp != 1.0:  # sharpen flat (e.g. UCB1 teacher) visit distributions: p ∝ visits^(1/T)
        target = target.pow(1 / policy_temp)
        target = target / target.sum(-1, keepdim=True).clamp(min=1e-12)
    # Balance the two losses without changing the units of predictions or targets.
    value_loss = ((value - target_value) / VALUE_SCALE).square().sum()
    policy_loss = (-(target * logits.log_softmax(-1)).sum(-1) * confidence).sum()
    hits = ((logits.argmax(-1) == target.argmax(-1)) & has_policy).sum().item()
    return value_loss, policy_loss, hits, len(value), has_policy.sum().item()


def run_epoch(net, data, a, optimizer=None, rng=None):
    totals = np.zeros(4)  # value, policy, hits, states; policy states separately
    policy_states = 0
    prediction_stats = {'max_abs_value_prediction': 0.0}
    for batch in data.batches(a.batch, rng):
        with torch.set_grad_enabled(optimizer is not None):
            value, policy, hits, n, n_policy = evaluate(
                net, batch, a.device, a.policy_temp, a.value_mix, a.teacher_root_mix, prediction_stats,
                flat_policy_weighting=getattr(a, 'flat_policy_weighting', False))
            loss = a.value_weight * (value / n) + a.policy_weight * (policy / max(n_policy, 1))
            if not torch.isfinite(loss): raise ValueError('non-finite training loss')
            if optimizer:
                optimizer.zero_grad(); loss.backward()
                if a.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), a.grad_clip)
                optimizer.step()
        totals += (value.item(), policy.item(), hits, n); policy_states += n_policy
    if not totals[3]: raise ValueError('no states in this split (need more independent run seeds)')
    return dict(value_loss=totals[0] / totals[3], policy_loss=totals[1] / max(policy_states, 1),
                top1=totals[2] / max(policy_states, 1), states=int(totals[3]), policy_states=policy_states,
                **prediction_stats)


def sum_groups(realized, groups):
    """Realized per-fight state counts -> per-group counts; a fight missing from the groups file is an error."""
    out = {}
    for fid, n in realized.items():
        out[groups[fid]] = out.get(groups[fid], 0) + n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, nargs='+', required=True, help='shard files (rows.parquet)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--init', type=Path)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--value-activation', choices=['softplus', 'sigmoid'], default=None,
                   help='fresh model head; resumes inherit checkpoint activation (legacy default softplus)')
    p.add_argument('--resume-optimizer', action='store_true', help='restore selected checkpoint AdamW moments; use explicit current lr/weight decay')
    p.add_argument('--train-fights', type=Path, help='JSON list of train fight IDs included this update; validation membership unchanged')
    p.add_argument('--states-per-fight', type=int, help='sample this many decision states per training fight/epoch, with replacement')
    p.add_argument('--device', default='cpu')
    p.add_argument('--policy-temp', type=float, default=1.0, help='policy target temperature (<1 sharpens visits)')
    p.add_argument('--flat-policy-weighting', action='store_true', help='downweight near-uniform visit targets; leave value targets unchanged')
    p.add_argument('--split-manifest', type=Path, help='JSON fight_id -> train/val, assigned by source deck; missing fights are errors')
    p.add_argument('--value-mix', type=float, default=1.0,
                   help='mix·100·won + (1-mix)·PV search root clamped to [0,100]; no-root rows keep outcome')
    p.add_argument('--teacher-root-mix', type=float, default=1.0,
                   help='must remain 1: old HP-unit teacher root mixing is unsupported under the win-only contract')
    p.add_argument('--policy-weight', type=float, default=1.0, help='policy loss weight; raw logged losses unchanged')
    p.add_argument('--value-weight', type=float, default=1.0, help='value loss weight; raw logged losses unchanged')
    p.add_argument('--grad-clip', type=float, default=None, help='clip total parameter gradient norm to G; default off')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--groups', type=Path, help='JSON fight_id -> group (deck); logs realized train states per group each epoch')
    p.add_argument('--mix-shards', action='store_true', help='pool sampled states across ALL shards so each minibatch mixes shards (requires non-stream)')
    p.add_argument('--stream', action='store_true', help='re-read one shard at a time per epoch instead of holding all in RAM')
    a = p.parse_args()
    if any(not math.isfinite(w) or w < 0 for w in (a.policy_weight, a.value_weight)):
        p.error('policy-weight and value-weight must be finite and nonnegative')
    if a.policy_weight == 0 and a.value_weight == 0:
        p.error('at least one loss weight must be positive')
    if a.grad_clip is not None and (not math.isfinite(a.grad_clip) or a.grad_clip <= 0):
        p.error('grad-clip must be finite and positive')
    if not 0 <= a.value_mix <= 1:
        p.error('value-mix must be in [0, 1]')
    if a.teacher_root_mix != 1:
        p.error('teacher root mixing uses old HP units and is incompatible with the win-only contract')
    if a.batch < 1 or a.epochs < 1: p.error('batch and epochs must be positive')
    if a.states_per_fight is not None and a.states_per_fight < 1: p.error('states-per-fight must be positive')
    if a.resume_optimizer and not a.init: p.error('resume-optimizer requires --init')
    if a.mix_shards and a.stream: p.error('mix-shards requires non-stream loading')
    torch.manual_seed(0); torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    if a.init:
        ckpt = torch.load(a.init, map_location='cpu', weights_only=False)
        if ckpt['contract'] != CONTRACT: raise ValueError('checkpoint input contract mismatch')
        activation = ckpt.get('value_activation', 'softplus')
        if a.value_activation is not None and a.value_activation != activation:
            raise ValueError('cannot change a checkpoint value activation on resume')
        net = PolicyValue(ckpt['width'], activation); net.load_state_dict(ckpt['state_dict'])
    else: net = PolicyValue(a.width, a.value_activation or 'softplus')
    net.to(a.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    if a.resume_optimizer:
        if 'optimizer_state_dict' not in ckpt:
            raise ValueError('checkpoint has no optimizer state')
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group.update(lr=a.lr, weight_decay=a.weight_decay)
    split = load_split(a.split_manifest)
    train_fights = None
    if a.train_fights:
        selected = json.loads(a.train_fights.read_text())
        if not isinstance(selected, list) or any(not isinstance(fid, str) for fid in selected):
            raise ValueError('train-fights must be a JSON list of fight IDs')
        train_fights = set(selected)
        if split is not None and any(split.get(fid) != 'train' for fid in train_fights):
            raise ValueError('train-fights contains non-training or unknown fight IDs')
    train = Dataset(a.data, False, a.stream, split, train_fights, a.states_per_fight, a.mix_shards)
    val = Dataset(a.data, True, a.stream, split)
    groups = json.loads(a.groups.read_text()) if a.groups else None
    a.out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    example = next(train.batches(2))[0]
    for epoch in range(a.epochs):
        net.train(); tr = run_epoch(net, train, a, optimizer, rng)
        if groups is not None:
            tr['groups'] = dict(sorted(sum_groups(train.realized, groups).items()))
        net.eval(); va = run_epoch(net, val, a)
        print(json.dumps({'epoch': epoch, 'train': tr, 'val': va}), flush=True)
        score = a.value_weight * va['value_loss'] + a.policy_weight * va['policy_loss']
        if score < best:
            best = score
            torch.save({'contract': CONTRACT, 'width': net.width, 'value_activation': net.value_activation,
                        'state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, a.out / 'model.pt')
    # One export, of the selected checkpoint, rather than recompiling ONNX every epoch.
    ckpt = torch.load(a.out / 'model.pt', map_location='cpu', weights_only=False)
    selected = PolicyValue(ckpt['width'], ckpt.get('value_activation', 'softplus')); selected.load_state_dict(ckpt['state_dict'])
    selected.export(example, a.out / 'model.onnx')


if __name__ == '__main__': main()
