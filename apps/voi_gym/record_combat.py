import argparse
from collections import Counter
from itertools import islice
import json
from math import isfinite
from pathlib import Path
import subprocess

from agents.combat.search.voi_gym.evaluator import prior
from agents.combat.search.voi_gym.model import Belief, adaptive_belief
from agents.combat.search.voi_gym.root import (
    CombatDecision, DEFAULT_INITIAL_SD_CAP, DEFAULT_PRIOR_SCALE, DEFAULT_PROBE_SD, DEFAULT_SHORTLIST,
)
from apps.voi_combat.benchmark import exchange

from .decision_view import DecisionView


def action_labels(actions):
    names = [' '.join(action.split()) for action in actions]
    counts = Counter(names)
    return [name if counts[name] == 1 else f'{name} [{i+1}]' for i, name in enumerate(names)]


def record_decision(proc, root, price, budget, **selector_settings):
    decision = CombatDecision(root, **selector_settings)
    labels = action_labels(root['actions'])
    view = DecisionView(decision, labels, root['value'], prior(root['logits']), budget, price,
                        selector_settings.get('prior_scale', DEFAULT_PRIOR_SCALE),
                        selector_settings.get('initial_sd_cap', DEFAULT_INITIAL_SD_CAP))
    q_calls, s_calls = root['requests'], 0
    observation = None
    frames = []
    for spent in range(budget+1):
        m = decision.select(price, budget-spent)
        frames.append(dict(state=view.snapshot(spent, q_calls, s_calls), rows=view.candidates(spent),
                           selected=m.name if m else None, observation=observation))
        if m is None:
            break
        predicted = decision.belief.prediction(m)[0]
        result = exchange(proc, {'pair': decision.pair(m)})
        decision.belief.observe(m, result['difference'])
        branches = [dict(branch, move=labels[branch['action']]) for branch in result['branches']]
        q_calls += result['requests']
        s_calls += sum(len(branch['path']) for branch in branches)
        observation = dict(id=m.name, observation=result['difference'], predicted=predicted, branches=branches)
    shortlisted = set(decision.actions)
    excluded = [label for i, label in enumerate(labels) if i not in shortlisted]
    return (dict(turn=root['turn'], step=root['step'], frames=frames, excluded=excluded, combat=root['combat']),
            decision.move(), q_calls)


def record_fight(proc, row, price, budget, **selector_settings):
    start = row['start']
    root = exchange(proc, start)
    decisions = []
    requests = 0
    while 'win' not in root:
        replay, move, used = record_decision(proc, root, price, budget, **selector_settings)
        decisions.append(replay)
        requests += used
        root = exchange(proc, {'play': move})
    if requests != root['requests']:
        raise ValueError('decision counts do not match fight request total')
    return dict(id=row['fight_id'], encounter=start['encounter'], decisions=decisions, **root)


def main():
    parser = argparse.ArgumentParser()
    for name in ('bridge', 'model', 'starts', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--price', type=float, default=.006)
    parser.add_argument('--budget', type=int, default=3)
    parser.add_argument('--shortlist', type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument('--probe-sd', type=float, default=DEFAULT_PROBE_SD)
    parser.add_argument('--prior-scale', type=float, default=DEFAULT_PRIOR_SCALE)
    parser.add_argument('--initial-sd-cap', type=float, default=DEFAULT_INITIAL_SD_CAP)
    parser.add_argument('--adaptive-scale', action='store_true')
    args = parser.parse_args()
    if args.limit < 1 or not 0 <= args.budget <= 64 or not isfinite(args.price) or args.price < 0:
        parser.error('invalid limit, budget, or price')
    if not (1 <= args.shortlist <= 40 and 0 < args.probe_sd <= 1
            and 0 <= args.prior_scale <= 1 and 0 <= args.initial_sd_cap <= .5):
        parser.error('invalid selector settings')
    with args.starts.open() as starts, args.out.open('x') as out:
        with subprocess.Popen([args.bridge, args.model, '--trace'], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as proc:
            try:
                for line in islice(starts, args.limit):
                    fight = record_fight(proc, json.loads(line), args.price, args.budget,
                                         shortlist=args.shortlist, probe_sd=args.probe_sd,
                                         prior_scale=args.prior_scale, initial_sd_cap=args.initial_sd_cap,
                                         belief_factory=adaptive_belief if args.adaptive_scale else Belief)
                    print(json.dumps(fight, allow_nan=False), file=out, flush=True)
            finally:
                proc.stdin.close()
            if proc.wait():
                raise RuntimeError('bridge failed')


if __name__ == '__main__':
    main()
