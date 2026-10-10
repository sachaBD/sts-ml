"""Synthetic checks for the Phase-2 deliberate-fit diagnostic helpers."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import correction_absorption as C  # noqa: E402
import correction_overfit as O  # noqa: E402
from agents.combat.pv.train import evaluate as trainer_evaluate  # noqa: E402


def test_token_classes_and_class_metrics():
    t = np.zeros((3, 261), np.float32); t[:2, 8] = 33; t[2, 8] = 65
    assert C.token_classes(t) == [[0, 1], [2]]
    r = dict(moves=[0, 1, 2], policy_target=[.2, .45, .35], tokens=t, won=False, has_policy=True)
    m = C.state_metrics(r, 0.0, np.array([1.0, 1.0, 0.0]), played=1)
    assert m['logit_tie_at_max'] and not m['top_hit']        # equal logits, index tiebreak picks move 0 (not in top)
    assert m['class_top_hit'] and abs(m['class_top_prob'] - 2 * np.e / (2 * np.e + 1)) < 1e-9


def test_loss_matches_trainer_weighted_objective():
    torch.manual_seed(0)
    net = C.PolicyValue(8, 'sigmoid')
    b = ({'context': torch.randn(3, 65), 'cards': torch.zeros(3, 2, 18), 'monsters': torch.zeros(3, 2, 29),
          'potions': torch.zeros(3, 2, 19), 'relics': torch.zeros(3, 2, 4), 'actions': torch.zeros(3, 3, 261)},
         torch.tensor([0., 100., 0.]), torch.tensor([[.6, .4, 0], [1., 0, 0], [.5, .5, 0]]),
         torch.tensor([True, True, False]), torch.full((3,), float('nan')))
    b[0]['actions'][:, :2, 260] = 1; b[0]['actions'][:, :2, 8] = torch.tensor([3., 4.])
    _, pol, _, _, npol = trainer_evaluate(net, b, 'cpu', flat_policy_weighting=True)
    assert abs(O.loss_fn(net, b, True).item() - pol.item() / npol) < 1e-6
    _, pol, _, _, npol = trainer_evaluate(net, b, 'cpu')
    assert abs(O.loss_fn(net, b, False).item() - pol.item() / npol) < 1e-6



def test_identical_action_tokens_get_identical_logits_regardless_of_position():
    torch.manual_seed(1)
    net = C.PolicyValue(8, 'sigmoid')
    a = torch.zeros(1, 4, 261); a[..., 260] = 1; a[0, :, 8] = torch.tensor([33., 65., 33., 11.]); a[0, :, 12] = 1
    inputs = [torch.randn(1, 65), torch.zeros(1, 2, 18), torch.zeros(1, 2, 29), torch.zeros(1, 2, 19), torch.zeros(1, 2, 4), a]
    _, l = net(*inputs)
    assert l[0, 0] == l[0, 2] and l[0, 0] != l[0, 1]
    _, l2 = net(*inputs[:-1], a[:, [2, 1, 0, 3]])          # permuting moves permutes logits only
    assert torch.allclose(l2[0], l[0, [2, 1, 0, 3]])


def test_joint_floor_matches_brute_force():
    class R: pass
    t = np.zeros((3, 261), np.float32); t[:2, 8] = 33; t[2, 8] = 65
    rows = R(); rows.meta = [dict(policy_target=[.5, .1, .4], tokens=t), dict(policy_target=[.2, .2, .6], tokens=t)]
    groups = {'same_input': [0, 1]}
    f = O.joint_floor(rows, groups, False)
    best = min(np.mean([sum(pi * np.log(pi / qi) for pi, qi in zip(p, (x / 2, x / 2, 1 - x)))
                        for p in ([.5, .1, .4], [.2, .2, .6])]) for x in np.linspace(.001, .999, 9999))
    assert abs(f - best) < 1e-6


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'): fn(); print('ok', name)
