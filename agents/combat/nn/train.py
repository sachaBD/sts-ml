"""Train the network on self-play samples: value BCE against the game result + policy cross-entropy against
the search's improved policy. Uniform sampling with replacement; a fixed number of steps."""
from dataclasses import dataclass
import time

import torch

from infra.workers.reader import prefetch
from .replay import Replay, TARGETS

WON = TARGETS.index('won')


@dataclass(frozen=True)
class TrainingCounters:
    step: int
    total: int
    seconds: float
    value_loss_sum: float   # sum of optimizer-step mean losses since training began
    policy_loss_sum: float


def losses(net, x, target, y, device):
    """Mean value BCE and policy cross-entropy of one microbatch, as the optimizer sees them.
    y holds replay.TARGETS per row; the value trains on the won column."""
    x, target, w = x.to(device), target.to(device), y.to(device)[:, WON]
    with torch.autocast(device, torch.bfloat16):
        value, logits = net(x.float())
    v = value.float().clamp(1e-6, 1 - 1e-6)
    value_loss = -(w * v.log() + (1 - w) * (1 - v).log()).mean()
    policy_loss = -(target * logits.float().log_softmax(-1)).sum(-1).mean()
    return value_loss, policy_loss


@torch.no_grad()
def score(net, replay, rows, batch, device='cuda'):
    """Mean (value, policy) loss using replay's policy sampling distribution; changes nothing."""
    net.eval()
    totals, count = torch.zeros(2, device=device), 0
    for _ in range(-(-rows // batch)):
        for x, target, w in replay.batches(batch):
            totals += len(w) * torch.stack(losses(net, x, target, w, device))
            count += len(w)
    return tuple((totals / count).tolist())


def train(net, optimizer, tokens, policy, targets, steps, batch=512, device='cuda', log=print, *, value_batch=0, on_counters=None):
    """tokens/policy/targets from data.read (any device); trains net (already on `device`) in place.
    Returns the mean (value, policy) loss of the last logged window."""
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
    start, totals = time.monotonic(), torch.zeros(2, device=device)
    next_report = start + 1
    completed = torch.zeros_like(totals) if on_counters else None
    try:
        if on_counters:
            on_counters(TrainingCounters(0, steps, 0., 0., 0.))
        for step in range(1, steps + 1):
            if replay is not None:
                batches, value_batches = next(ready)
            else:
                i = torch.randint(len(targets), (batch,), device=device)
                batches, value_batches = [(tokens[i], policy[i], targets[i])], []
            optimizer.zero_grad(set_to_none=True)
            for x, target, w in batches:
                value_loss, policy_loss = losses(net, x, target, w, device)
                weight = len(w) / batch
                if value_batch:
                    (policy_loss * weight).backward()
                    totals[1] += weight * policy_loss.detach()
                else:
                    ((value_loss + policy_loss) * weight).backward()
                    totals += weight * torch.stack((value_loss.detach(), policy_loss.detach()))
            for x, target, w in value_batches:
                value_loss, _ = losses(net, x, target, w, device)
                weight = len(w) / value_batch
                (value_loss * weight).backward()
                totals[0] += weight * value_loss.detach()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if on_counters and (step == steps or time.monotonic() >= next_report):
                value_sum, policy_sum = (completed + totals).tolist()
                now = time.monotonic()
                on_counters(TrainingCounters(step, steps, now - start, value_sum, policy_sum))
                next_report = now + 1
            if step % 500 == 0 or step == steps:
                v, p = (totals / ((step - 1) % 500 + 1)).tolist()
                log(f'step {step}/{steps}: value {v:.4f} policy {p:.4f} ({step * batch / (time.monotonic() - start):.0f} states/s)')
                if on_counters:
                    completed += totals
                totals.zero_()
    finally:
        if ready is not None:
            ready.close()
    return v, p
