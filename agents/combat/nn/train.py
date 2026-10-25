"""Train the network on self-play samples: value BCE against the game result + policy cross-entropy against
the search's improved policy. Uniform sampling with replacement; a fixed number of steps."""
import time

import torch


def train(net, optimizer, tokens, policy, won, steps, batch=512, device='cuda', log=print):
    """tokens/policy/won from data.read (any device); trains net (already on `device`) in place."""
    net.train()
    tokens, policy, won = tokens.to(device), policy.to(device), won.to(device)
    start, totals = time.time(), torch.zeros(2, device=device)
    for step in range(1, steps + 1):
        i = torch.randint(len(won), (batch,), device=device)
        with torch.autocast(device, torch.bfloat16):  # bf16 halves memory traffic: ~2x on this small net
            value, logits = net(tokens[i].float())
        v, w = value.float().clamp(1e-6, 1 - 1e-6), won[i]
        value_loss = -(w * v.log() + (1 - w) * (1 - v).log()).mean()
        policy_loss = -(policy[i] * logits.float().log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        (value_loss + policy_loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        totals += torch.stack((value_loss.detach(), policy_loss.detach()))
        if step % 500 == 0 or step == steps:
            v, p = (totals / ((step - 1) % 500 + 1)).tolist()
            log(f'step {step}/{steps}: value {v:.4f} policy {p:.4f} ({step * batch / (time.time() - start):.0f} states/s)')
            totals.zero_()
