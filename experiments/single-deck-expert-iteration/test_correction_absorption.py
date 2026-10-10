"""Focused checks of the absorption diagnostic's metric/sampler helpers (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import correction_absorption as C  # noqa: E402
from agents.combat.pv.train import policy_confidence  # noqa: E402


def row(policy, moves=None, tokens=None, won=True, has_policy=True):
    n = len(policy)
    t = np.zeros((n, 261), np.float32) if tokens is None else tokens
    for i in range(n):
        if tokens is None: t[i, 8] = 100 + i
    return dict(moves=list(range(n)) if moves is None else moves, policy_target=policy, tokens=t, won=won,
                has_policy=has_policy)


def test_confidence_matches_trainer():
    p = np.array([.5, .3, .2], np.float32)
    ref = policy_confidence(torch.tensor(p)[None], torch.ones(1, 3, dtype=torch.bool)).item()
    assert abs(C.confidence(p, 3) - ref) < 1e-6
    assert C.confidence(np.full(4, .25), 4) == 0 and C.confidence(np.array([1., 0]), 2) == 1


def test_tied_target_set_and_margins():
    r = row([.4, .4, .2])
    m = C.state_metrics(r, 50.0, np.array([1.0, 3.0, 2.0, -1e9]), played=0)  # padded logit ignored
    assert m['n_top'] == 2 and m['top_hit'] and m['top_rank'] == 1
    assert abs(m['top_margin'] - 1.0) < 1e-9          # best of tie set (3) minus best outside (2)
    assert m['played_rank'] == 3 and abs(m['played_margin'] + 2.0) < 1e-9
    q = np.exp([1, 3, 2]) / np.exp([1, 3, 2]).sum()
    assert abs(m['top_prob'] - q[:2].sum()) < 1e-9 and abs(m['value_sq'] - .25) < 1e-12
    assert abs(m['kl'] - (m['ce'] + (np.array([.4, .4, .2]) * np.log([.4, .4, .2])).sum())) < 1e-9


def test_identical_tokens_form_indistinguishable_class_only():
    t = np.zeros((3, 261), np.float32); t[:, 8] = 33; t[2, 8] = 65
    m = C.state_metrics(row([.5, .25, .25], tokens=t), 0.0, np.array([0.0, 0.0, 0.0]), played=1)
    assert abs(m['played_class_prob'] - 2 / 3) < 1e-9 and abs(m['played_prob'] - 1 / 3) < 1e-9


def test_fight_balanced_vs_decision():
    rec = [dict(fight_id='a', x=dict(v=1.0)), dict(fight_id='a', x=dict(v=1.0)), dict(fight_id='a', x=dict(v=1.0)),
           dict(fight_id='b', x=dict(v=0.0))]
    s, n = C.fight_table(rec, lambda r: r['x']['v'], ['a', 'b'])
    assert C.agg(s, n) == (0.75, 0.5)
    assert C.agg(s, n, np.array([1, 1])) == (0.0, 0.0)


def test_stratum():
    t = np.zeros((2, 261), np.float32); t[:, 0] = 2; t[:, 1] = C.TASK_DUAL_WIELD
    assert C.stratum(t, (2, C.TASK_DUAL_WIELD)) == 'dual_wield_select'
    t = np.zeros((2, 261), np.float32); t[0, 8] = 107; t[1, 0] = 4
    assert C.stratum(t, (0, 0)) == 'setup_legal'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'): fn(); print('ok', name)
