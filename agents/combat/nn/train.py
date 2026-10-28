"""Train the network on self-play samples: value BCE against the game result + policy cross-entropy against
the search's improved policy. Uniform sampling with replacement; a fixed number of steps."""
import time

import torch

from .replay import Replay


def train(net, optimizer, tokens, policy, won, steps, batch=512, device='cuda', log=print):
    """tokens/policy/won from data.read (any device); trains net (already on `device`) in place."""
    net.train()
    replay = tokens if isinstance(tokens, Replay) else None
    if replay is None:
        tokens, policy, won = tokens.to(device), policy.to(device), won.to(device)
    start, totals = time.time(), torch.zeros(2, device=device)
    for step in range(1, steps + 1):
        if replay is not None:
            batches = replay.batches(batch)
        else:
            i = torch.randint(len(won), (batch,), device=device)
            batches = [(tokens[i], policy[i], won[i])]
        optimizer.zero_grad(set_to_none=True)
        for x, target, w in batches:
            x, target, w = x.to(device), target.to(device), w.to(device)
            with torch.autocast(device, torch.bfloat16):
                value, logits = net(x.float())
            v = value.float().clamp(1e-6, 1 - 1e-6)
            value_loss = -(w * v.log() + (1 - w) * (1 - v).log()).mean()
            policy_loss = -(target * logits.float().log_softmax(-1)).sum(-1).mean()
            weight = len(w) / batch
            ((value_loss + policy_loss) * weight).backward()
            totals += weight * torch.stack((value_loss.detach(), policy_loss.detach()))
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 500 == 0 or step == steps:
            v, p = (totals / ((step - 1) % 500 + 1)).tolist()
            log(f'step {step}/{steps}: value {v:.4f} policy {p:.4f} ({step * batch / (time.time() - start):.0f} states/s)')
            totals.zero_()
