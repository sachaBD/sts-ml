"""Train the network on self-play samples: a value loss against the combat's result + policy cross-entropy
against the search's improved policy. Uniform sampling with replacement; a fixed number of steps."""
from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from infra.workers.reader import prefetch
from .model import AUXILIARY, LOSS_BINS, WIN_BINS, value_width, win_probability
from .replay import Replay, TARGETS

WON, HP, ENEMY_HP = (TARGETS.index(name) for name in ('won', 'hp', 'enemy_hp'))
MAX_TURNS = 50  # Fight::max_turns; scales the turns-remaining target
AUXILIARY_WEIGHT = 0.25

# losses() order. Policy batches train policy and the per-round heads, which differ for every state;
# value batches train value and turns, which are shared by a whole combat. Win BCE is only measured.
LOSSES = ('value', 'policy', 'win', 'round', 'turns')
OBJECTIVE = torch.tensor([1., 1., 0., AUXILIARY_WEIGHT, AUXILIARY_WEIGHT])
POLICY_SIDE = torch.tensor([0., 1., 0., 1., 0.])
VALUE_SIDE = 1 - POLICY_SIDE


@dataclass(frozen=True)
class TrainingCounters:
    step: int
    total: int
    seconds: float
    value_loss_sum: float   # sum of optimizer-step mean losses since training began
    policy_loss_sum: float


def outcome_bin(targets):
    """Margin axis: losses with most enemy HP left first, then wins with least player HP first."""
    loss = LOSS_BINS - 1 - (targets[:, ENEMY_HP] * LOSS_BINS).floor().clamp(0, LOSS_BINS - 1)
    win = LOSS_BINS - 1 + (targets[:, HP] * WIN_BINS).ceil().clamp(1, WIN_BINS)
    return torch.where(targets[:, WON] > 0, win, loss).long()


def losses(net, x, target, y, device):
    """Mean LOSSES of one microbatch. Value is win BCE for spec.value == 'win', otherwise outcome-bin
    cross-entropy; round (HP and enemy HP lost) and turns are MSE, zero without spec.auxiliary."""
    x, target, y = x.to(device), target.to(device), y.to(device)
    with torch.autocast(device, torch.bfloat16):
        context, logits = net.heads(x.float())
    context = context.float()
    win = win_probability(context, net.spec).clamp(1e-6, 1 - 1e-6)
    won = y[:, WON]
    win_loss = -(won * win.log() + (1 - won) * (1 - win).log()).mean()
    width = value_width(net.spec)
    value_loss = win_loss if net.spec.value == 'win' else F.cross_entropy(context[:, :width], outcome_bin(y))
    policy_loss = -(target * logits.float().log_softmax(-1)).sum(-1).mean()
    round_loss = turns_loss = torch.zeros((), device=device)
    if net.spec.auxiliary:
        predicted = context[:, width:]
        scale = torch.tensor([MAX_TURNS if name == 'turns_remaining' else 1. for name in AUXILIARY], device=device)
        squared = dict(zip(AUXILIARY, (predicted - y[:, [TARGETS.index(name) for name in AUXILIARY]] / scale)
                           .square().mean(0)))
        round_loss, turns_loss = squared['hp_lost'] + squared['enemy_hp_lost'], squared['turns_remaining']
    return torch.stack((value_loss, policy_loss, win_loss, round_loss, turns_loss))


@torch.no_grad()
def score(net, replay, rows, batch, device='cuda'):
    """Mean LOSSES using replay's policy sampling distribution; changes nothing."""
    net.eval()
    totals, count = torch.zeros(len(LOSSES), device=device), 0
    for _ in range(-(-rows // batch)):
        for x, target, y in replay.batches(batch):
            totals += len(y) * losses(net, x, target, y, device)
            count += len(y)
    return tuple((totals / count).tolist())


def train(net, optimizer, tokens, policy, targets, steps, batch=512, device='cuda', log=print, *, value_batch=0,
          on_counters=None, loss_fn=losses):
    """tokens/policy/targets from data.read (any device); trains net (already on `device`) in place.
    Returns the mean LOSSES of the last logged window."""
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
    objective, policy_side, value_side = OBJECTIVE.to(device), POLICY_SIDE.to(device), VALUE_SIDE.to(device)
    if not value_batch:
        policy_side = torch.ones_like(policy_side)  # one shared batch trains and measures everything

    def accumulate(x, target, y, side, weight):
        measured = side * loss_fn(net, x, target, y, device)
        ((objective * measured).sum() * weight).backward()
        totals.add_(weight * measured.detach())

    start, totals = time.monotonic(), torch.zeros(len(LOSSES), device=device)
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
            for x, target, y in batches:
                accumulate(x, target, y, policy_side, len(y) / batch)
            for x, target, y in value_batches:
                accumulate(x, target, y, value_side, len(y) / value_batch)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if on_counters and (step == steps or time.monotonic() >= next_report):
                value_sum, policy_sum = (completed + totals)[:2].tolist()
                now = time.monotonic()
                on_counters(TrainingCounters(step, steps, now - start, value_sum, policy_sum))
                next_report = now + 1
            if step % 500 == 0 or step == steps:
                mean = (totals / ((step - 1) % 500 + 1)).tolist()
                log(f'step {step}/{steps}: ' + ' '.join(f'{k} {v:.4f}' for k, v in zip(LOSSES, mean))
                    + f' ({step * batch / (time.monotonic() - start):.0f} states/s)')
                if on_counters:
                    completed += totals
                totals.zero_()
    finally:
        if ready is not None:
            ready.close()
    return tuple(mean)
