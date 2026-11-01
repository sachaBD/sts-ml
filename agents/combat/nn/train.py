"""Train the network on self-play samples: value BCE against the game result + policy cross-entropy against
the search's improved policy. Uniform sampling with replacement; a fixed number of steps."""
import time

import torch

from infra.workers.reader import prefetch
from .replay import Replay


def losses(net, x, target, w, device):
    """Mean value BCE and policy cross-entropy of one microbatch, as the optimizer sees them."""
    x, target, w = x.to(device), target.to(device), w.to(device)
    with torch.autocast(device, torch.bfloat16):
        value, logits = net(x.float())
    v = value.float().clamp(1e-6, 1 - 1e-6)
    value_loss = -(w * v.log() + (1 - w) * (1 - v).log()).mean()
    policy_loss = -(target * logits.float().log_softmax(-1)).sum(-1).mean()
    return value_loss, policy_loss


@torch.no_grad()
def score(net, replay, rows, batch, device='cuda'):
    """Mean (value, policy) loss over at least `rows` rows sampled like training batches; changes nothing."""
    net.eval()
    totals, count = torch.zeros(2, device=device), 0
    for _ in range(-(-rows // batch)):
        for x, target, w in replay.batches(batch):
            totals += len(w) * torch.stack(losses(net, x, target, w, device))
            count += len(w)
    return tuple((totals / count).tolist())


def train(net, optimizer, tokens, policy, won, steps, batch=512, device='cuda', log=print):
    """tokens/policy/won from data.read (any device); trains net (already on `device`) in place.
    Returns the mean (value, policy) loss of the last logged window."""
    net.train()
    replay = tokens if isinstance(tokens, Replay) else None
    if replay is None:
        tokens, policy, won = tokens.to(device), policy.to(device), won.to(device)
    # Readers build the next steps' microbatches on CPU threads while this thread trains on the GPU.
    ready = prefetch(lambda _: list(replay.batches(batch)), steps) if replay is not None else None
    start, totals = time.time(), torch.zeros(2, device=device)
    for step in range(1, steps + 1):
        if replay is not None:
            batches = next(ready)
        else:
            i = torch.randint(len(won), (batch,), device=device)
            batches = [(tokens[i], policy[i], won[i])]
        optimizer.zero_grad(set_to_none=True)
        for x, target, w in batches:
            value_loss, policy_loss = losses(net, x, target, w, device)
            weight = len(w) / batch
            ((value_loss + policy_loss) * weight).backward()
            totals += weight * torch.stack((value_loss.detach(), policy_loss.detach()))
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 500 == 0 or step == steps:
            v, p = (totals / ((step - 1) % 500 + 1)).tolist()
            log(f'step {step}/{steps}: value {v:.4f} policy {p:.4f} ({step * batch / (time.time() - start):.0f} states/s)')
            totals.zero_()
    return v, p
