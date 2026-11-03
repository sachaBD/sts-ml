"""Train the network on self-play samples: a value loss against the combat's result + policy cross-entropy
against the search's improved policy. Uniform sampling with replacement; a fixed number of steps."""
import time

import torch
import torch.nn.functional as F

from infra.workers.reader import prefetch
from .model import LOSS_BINS, WIN_BINS, win_probability
from .replay import Replay, TARGETS

WON, HP, ENEMY_HP = (TARGETS.index(name) for name in ('won', 'hp', 'enemy_hp'))


def outcome_bin(targets):
    """Margin axis: losses with most enemy HP left first, then wins with least player HP first."""
    loss = LOSS_BINS - 1 - (targets[:, ENEMY_HP] * LOSS_BINS).floor().clamp(0, LOSS_BINS - 1)
    win = LOSS_BINS - 1 + (targets[:, HP] * WIN_BINS).ceil().clamp(1, WIN_BINS)
    return torch.where(targets[:, WON] > 0, win, loss).long()


def losses(net, x, target, y, device):
    """Mean (value, policy, win BCE) of one microbatch; the optimizer sees value and policy.
    Value is win BCE for spec.value == 'win', otherwise outcome-bin cross-entropy."""
    x, target, y = x.to(device), target.to(device), y.to(device)
    with torch.autocast(device, torch.bfloat16):
        value_head, logits = net.heads(x.float())
    value_head = value_head.float()
    win = win_probability(value_head, net.spec).clamp(1e-6, 1 - 1e-6)
    won = y[:, WON]
    win_loss = -(won * win.log() + (1 - won) * (1 - win).log()).mean()
    value_loss = win_loss if net.spec.value == 'win' else F.cross_entropy(value_head, outcome_bin(y))
    policy_loss = -(target * logits.float().log_softmax(-1)).sum(-1).mean()
    return value_loss, policy_loss, win_loss


@torch.no_grad()
def score(net, replay, rows, batch, device='cuda'):
    """Mean (value, policy, win BCE) loss using replay's policy sampling distribution; changes nothing."""
    net.eval()
    totals, count = torch.zeros(3, device=device), 0
    for _ in range(-(-rows // batch)):
        for x, target, w in replay.batches(batch):
            totals += len(w) * torch.stack(losses(net, x, target, w, device))
            count += len(w)
    return tuple((totals / count).tolist())


def train(net, optimizer, tokens, policy, targets, steps, batch=512, device='cuda', log=print, *, value_batch=0):
    """tokens/policy/targets from data.read (any device); trains net (already on `device`) in place.
    Returns the mean (value, policy, win BCE) loss of the last logged window."""
    replay = tokens if isinstance(tokens, Replay) else None
    if value_batch < 0 or (value_batch and replay is None):
        raise ValueError('value_batch must be nonnegative and requires Replay when enabled')
    if value_batch and value_batch > replay.combat_count:
        raise ValueError('value batch exceeds the combat count')
    net.train()
    if replay is None:
        tokens, policy, targets = tokens.to(device), policy.to(device), targets.to(device)

    def prepare(_):
        policy_batches = list(replay.batches(batch))
        value_batches = list(replay.batches(value_batch, combat_balanced=True)) if value_batch else []
        return policy_batches, value_batches

    # One producer owns sampling for both objectives; checkpoint RNG covers both streams.
    ready = prefetch(prepare, steps) if replay is not None else None
    start, totals = time.time(), torch.zeros(3, device=device)
    try:
        for step in range(1, steps + 1):
            if replay is not None:
                batches, value_batches = next(ready)
            else:
                i = torch.randint(len(targets), (batch,), device=device)
                batches, value_batches = [(tokens[i], policy[i], targets[i])], []
            optimizer.zero_grad(set_to_none=True)
            for x, target, y in batches:
                value_loss, policy_loss, win_loss = losses(net, x, target, y, device)
                weight = len(y) / batch
                if value_batch:
                    (policy_loss * weight).backward()
                    totals[1] += weight * policy_loss.detach()
                else:
                    ((value_loss + policy_loss) * weight).backward()
                    totals += weight * torch.stack((value_loss.detach(), policy_loss.detach(), win_loss.detach()))
            for x, target, y in value_batches:
                value_loss, _, win_loss = losses(net, x, target, y, device)
                weight = len(y) / value_batch
                (value_loss * weight).backward()
                totals[0] += weight * value_loss.detach()
                totals[2] += weight * win_loss.detach()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 500 == 0 or step == steps:
                v, p, w = (totals / ((step - 1) % 500 + 1)).tolist()
                log(f'step {step}/{steps}: value {v:.4f} policy {p:.4f} win {w:.4f} '
                    f'({step * batch / (time.time() - start):.0f} states/s)')
                totals.zero_()
    finally:
        if ready is not None:
            ready.close()
    return v, p, w
