"""In-process GUI backend. Public methods return JSON-compatible records."""
from copy import deepcopy

import numpy as np

from agents.combat.search.voi_gym.model import Belief, Chance, Choice, Leaf, Measurement, policies, rank_measurements, buy
from environments.combat.card_duel import Duel, OPEN, INTENTS, RESPOND


class Session:
    def __init__(self, seed=0, budget=16, price=.001, *, adaptive=True):
        if budget < 0 or price < 0:
            raise ValueError('negative budget or price')
        self.game = g = Duel(seed)
        n = len(g.labels)
        scales, weights = g.scales, g.scale_probabilities
        if not adaptive:  # Moment-match the initial predictive variance.
            scales, weights = [np.sqrt(np.dot(weights, np.square(scales)))], [1.]
        self.belief = Belief(g.prior, np.eye(n)*g.prior_sd**2, scales, weights)
        self.tree = Choice({a: Chance({i: (p, Choice({b: Leaf(g.labels.index((a, i, b))) for b in RESPOND}))
                                    for i, p in zip(INTENTS, (g.attack_prob, 1-g.attack_prob))}) for a in OPEN})
        plans, reaches = zip(*policies(self.tree, n))
        self.plans, self.reaches = plans, np.stack(reaches)
        self.measurements = {}
        basis = np.eye(n)
        for j, path in enumerate(g.labels):
            name = '/'.join(path)
            self.measurements[name] = Measurement(name, basis[j], g.scout_sd**2, 1)
            # Compare responses in the same matchup with a shared scout draw.
            # Per-endpoint noise correlation .75 -> difference variance = .5 R.
            if path[-1] == RESPOND[1]:
                key = name + ' minus Strike'
                self.measurements[key] = Measurement(key, basis[j]-basis[j-1], .5*g.scout_sd**2, 2)
        self.budget, self.price, self.spent = budget, price, 0
        self.history = []

    def snapshot(self):
        b = self.belief
        values = self.reaches @ b.mean
        i = int(values.argmax())
        variance = (np.exp(b.log_weights) @ b.scales**2) * b.cov.diagonal()
        return dict(seed=self.game.seed, price=self.price, spent=self.spent, budget=self.budget,
                    plan=deepcopy(self.plans[i]), value=float(values[i]), scales=b.scales.tolist(),
                    scale_probabilities=np.exp(b.log_weights).tolist(), attack_probability=self.game.attack_prob,
                    leaves={'/'.join(p): float(v) for p, v in zip(self.game.labels, b.mean)},
                    root_q={a: float(max(v for p,v in zip(self.plans, values) if a in p)) for a in OPEN},
                    leaf_sd={'/'.join(p): float(sd) for p, sd in zip(self.game.labels, np.sqrt(variance))},
                    history=deepcopy(self.history))

    def candidates(self):
        rows = rank_measurements(self.belief, self.reaches, self.measurements.values(),
                                 self.price, self.budget - self.spent)
        for row in rows:
            h = self.measurements[row['id']].h
            row['terms'] = {'/'.join(p): float(w) for p, w in zip(self.game.labels, h) if w}
        return rows

    def step(self, candidate_id=None):
        if candidate_id is None:
            candidate_id = buy(self.candidates())
            if candidate_id is None:
                return None
        m = self.measurements[candidate_id]
        if self.spent+m.cost > self.budget:
            raise ValueError('measurement exceeds remaining budget')
        y = self.game.scout(m, sum(r['id'] == candidate_id for r in self.history))
        predicted, sd, _ = self.belief.prediction(m)
        predictive_sd = sd*np.sqrt(np.exp(self.belief.log_weights) @ self.belief.scales**2)
        self.belief.observe(m, y)
        self.spent += m.cost
        row = dict(id=m.name, observation=y, predicted=predicted, predictive_sd=float(predictive_sd), spent=self.spent)
        self.history.append(row)
        return row.copy()

    def grade(self):
        """Oracle-only evaluation, never consulted by selection. No sampling of win/loss needed."""
        truth = self.reaches @ self.game.truth
        return dict(no_search=float(truth[(self.reaches @ self.game.prior).argmax()]),
                    search=float(truth[(self.reaches @ self.belief.mean).argmax()]), oracle=float(truth.max()))
