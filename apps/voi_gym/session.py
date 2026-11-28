from dataclasses import replace
from functools import cache
from math import log

import numpy as np

from agents.combat.search.voi_gym.evaluator import Q
from agents.combat.search.voi_gym.model import Belief
from agents.combat.search.voi_gym.root import CombatDecision, DEFAULT_INITIAL_SD_CAP, DEFAULT_PRIOR_SCALE
from environments.combat.card_duel import Duel, INTENTS, OPEN, RESPOND, SHOCKS, S, actions, margin, value


class Session:
    def __init__(self, seed=0, budget=3, price=.006, *, belief_factory=Belief, prior_scale=DEFAULT_PRIOR_SCALE,
                 initial_sd_cap=DEFAULT_INITIAL_SD_CAP):
        if budget < 0 or price < 0:
            raise ValueError('negative budget or price')
        self.seed, self.budget, self.price = seed, budget, price
        self.prior_scale = prior_scale
        self.initial_sd_cap = initial_sd_cap
        self.game = Duel(seed)
        self.evaluate = cache(Q)
        self.s_calls = 0
        self.root_value, self.prior = self.evaluate(self.game.root)
        v, p = self.root_value, self.prior
        self.decision = CombatDecision(dict(value=v, logits=[log(probability) for probability in p]), belief_factory=belief_factory,
                                       prior_scale=prior_scale, initial_sd_cap=initial_sd_cap)
        self.history = []

    @property
    def spent(self):
        return len(self.history)

    def snapshot(self):
        d = self.decision
        b = d.belief
        weights = np.exp(b.log_weights)
        scale_variance = float(weights @ b.scales**2)
        return dict(seed=self.seed, spent=self.spent, budget=self.budget, price=self.price,
                    move=OPEN[d.move()], root_q={OPEN[a]: float(v) for a, v in zip(d.actions, b.mean)},
                    actions=[OPEN[a] for a in d.actions],
                    root_value=self.root_value, prior_scale=self.prior_scale, initial_sd_cap=self.initial_sd_cap,
                    prior={a: p for a, p in zip(OPEN, self.prior)},
                    scales=b.scales.tolist(), weights=weights.tolist(), covariance=b.cov.tolist(),
                    sd={OPEN[a]: float(v) for a, v in zip(d.actions, np.sqrt(scale_variance*b.cov.diagonal()))},
                    q_calls=self.evaluate.cache_info().misses, s_calls=self.s_calls)

    def candidates(self):
        d = self.decision
        b = d.belief
        rows = d.candidates(self.price, self.budget-self.spent)
        for row in rows:
            m = d.measurement(row['id'])
            left, right = d.pair(m)
            center, base_sd, shift = b.prediction(m)
            mixture_sd = base_sd*np.sqrt(np.exp(b.log_weights) @ b.scales**2)
            row.update(left=OPEN[left], right=OPEN[right], predicted=center, base_sd=base_sd,
                       predictive_sd=float(mixture_sd), noise_var=m.noise_var,
                       shift=shift.tolist(), gains=b.gain_by_scale(m, d.reaches))
        return rows

    def rollout(self, s, action):
        path = []
        while True:
            self.s_calls += 1
            s = S(s, action)
            path.append(dict(action=action, state=s.intent if s.phase == 'Respond' else s.phase))
            if s.phase == 'Terminal':
                return value(s.score), path, s.score
            if s.phase == 'Next turn':
                return self.evaluate(s)[0], path, s.score
            legal = actions(s)
            if len(legal) == 1:
                action, = legal
            else:
                _, p = self.evaluate(replace(s, world=None))
                action = max(zip(legal, p), key=lambda row: row[1])[0]

    def step(self):
        d = self.decision
        m = d.select(self.price, self.budget-self.spent)
        if m is None:
            return None
        world = self.game.sample()
        branches = []
        for a in d.pair(m):
            endpoint, path, score = self.rollout(world, OPEN[a])
            branches.append(dict(move=OPEN[a], value=endpoint, path=path, score=score))
        y = branches[0]['value'] - branches[1]['value']
        predicted = d.belief.prediction(m)[0]
        d.belief.observe(m, y)
        row = dict(id=m.name, observation=y, predicted=predicted, branches=branches)
        self.history.append(row)
        return row

    def grade(self):
        attack = self.game.root.attack_probability
        truth = [sum(p * value(max(margin(a, i, r) for r in RESPOND) + h + self.game.matchup[j]) / len(SHOCKS)
                     for i, p in zip(INTENTS, (attack, 1-attack)) for h in SHOCKS)
                 for j, a in enumerate(OPEN)]
        d = self.decision
        return dict(no_search=truth[d.actions[0]], search=truth[d.move()], oracle=max(truth))
