from itertools import combinations

import numpy as np

from .model import Belief, Measurement, buy, rank_measurements


DEFAULT_PRIOR_SCALE = .03
DEFAULT_INITIAL_SD_CAP = .06
DEFAULT_SHORTLIST = 4
DEFAULT_PROBE_SD = .08


class CombatDecision:
    def __init__(self, root, *, belief_factory=Belief, prior_scale=DEFAULT_PRIOR_SCALE,
                 initial_sd_cap=DEFAULT_INITIAL_SD_CAP, shortlist=DEFAULT_SHORTLIST, probe_sd=DEFAULT_PROBE_SD):
        if shortlist < 1 or not np.isfinite(probe_sd) or probe_sd <= 0:
            raise ValueError("invalid shortlist or probe SD")
        self.shortlist, self.probe_sd = shortlist, probe_sd
        logits = np.array(root['logits'])
        self.actions = np.argsort(-logits, kind='stable')[:shortlist]
        v = root['value']
        means = v + prior_scale*(logits[self.actions] - logits[self.actions[0]])
        self.reaches = np.eye(len(means))  # choose a root action, then fixed continuation
        prior_sd = min(initial_sd_cap, v, 1-v)
        self.belief = belief_factory(means, self.reaches * prior_sd**2)
        self.pairs = [(i, j) for j, i in combinations(range(len(means)), 2)]
        self.probes = [Measurement(str(k), self.reaches[i]-self.reaches[j], probe_sd**2, 1)
                       for k, (i, j) in enumerate(self.pairs)]

    def measurement(self, key):
        return self.probes[int(key)]

    def candidates(self, price, remaining):
        return rank_measurements(self.belief, self.reaches, self.probes, price, remaining)

    def select(self, price, remaining):
        if not remaining:
            return None
        key = buy(self.candidates(price, remaining))
        return None if key is None else self.measurement(key)

    def pair(self, m):
        return [int(self.actions[i]) for i in self.pairs[int(m.name)]]

    def move(self):
        return int(self.actions[self.belief.mean.argmax()])
