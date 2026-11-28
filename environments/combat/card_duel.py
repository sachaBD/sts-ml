"""Two plays: open, see enemy intent, respond, then resolve a noisy final duel."""
from math import exp

import numpy as np

OPEN = ('Strike', 'Bash', 'Defend')
INTENTS = ('attack', 'brace')
RESPOND = ('Strike', 'Defend')


class Duel:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.attack_prob = float(rng.uniform(.25, .75))
        self.labels = [(a, i, b) for a in OPEN for i in INTENTS for b in RESPOND]
        # Strike deals 4; Bash deals 2 and doubles the follow-up strike.
        # Defend blocks 5; a braced enemy blocks 3 on the second play.
        def nominal(a, i, b):
            damage = {'Strike': 4, 'Bash': 2, 'Defend': 0}[a]
            damage += max(0, 4*(1+(a == 'Bash'))*(b == 'Strike') - 3*(i == 'brace'))
            incoming = max(0, (9 if i == 'attack' else 2) - 5*((a == 'Defend') + (b == 'Defend')))
            return 1/(1+exp(-(damage-incoming)/7))
        self.prior = np.array([nominal(*path) for path in self.labels])
        self.prior_sd = .09
        self.scout_sd = .12
        self.scales, self.scale_probabilities = [1., 3.], [.9, .1]
        self.scale = rng.choice(self.scales, p=self.scale_probabilities)
        # Fixed hidden matchup effects. Clipping makes truth a probability, while
        # the search deliberately uses an unbounded Gaussian approximation.
        self.truth = np.clip(self.prior + rng.normal(0, self.scale*self.prior_sd, len(self.labels)), 0, 1)
        self.seed = seed

    def scout(self, m, repetition):
        # Per-measurement streams: comparing allocators doesn't reshuffle their evidence.
        rng = np.random.default_rng([self.seed, 938, repetition, *map(int, m.h+1)])
        return float(m.h @ self.truth + rng.normal(0, self.scale*np.sqrt(m.noise_var)))
