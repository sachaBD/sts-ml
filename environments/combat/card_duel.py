"""A tiny combat state machine: opener, response, resolve, then score the next turn."""
from dataclasses import dataclass, replace
from math import exp

import numpy as np

OPEN = ('Strike', 'Bash', 'Defend')
RESPOND = ('Strike', 'Defend')
INTENTS = ('attack', 'brace')
SHOCKS = (-2, 0, 2)


@dataclass(frozen=True)
class World:
    intent: str
    shock: int
    matchup: tuple[int, ...]


@dataclass(frozen=True)
class State:
    attack_probability: float
    phase: str = 'Open'
    opener: str = ''
    intent: str = ''
    response: str = ''
    score: int = 0
    world: World | None = None


def actions(s):
    return {'Open': OPEN, 'Respond': RESPOND, 'Resolve': ('Resolve',)}.get(s.phase, ())


def margin(opener, intent, response):
    damage = {'Strike': 4, 'Bash': 2, 'Defend': 0}[opener]
    damage += max(0, 4*(1+(opener == 'Bash'))*(response == 'Strike') - 3*(intent == 'brace'))
    incoming = max(0, (9 if intent == 'attack' else 2) - 5*((opener == 'Defend') + (response == 'Defend')))
    return damage-incoming


def value(score):
    return 0. if score <= -12 else 1. if score >= 12 else 1/(1+exp(-score/7))


def S(s, action):
    if action not in actions(s) or s.world is None:
        raise ValueError('invalid simulated action')
    if s.phase == 'Open':
        return replace(s, phase='Respond', opener=action, intent=s.world.intent)
    if s.phase == 'Respond':
        return replace(s, phase='Resolve', response=action)
    score = margin(s.opener, s.intent, s.response) + s.world.shock + s.world.matchup[OPEN.index(s.opener)]
    return State(s.attack_probability, phase='Terminal' if abs(score) >= 12 else 'Next turn', score=score)


class Duel:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.root = State(float(rng.uniform(.25, .75)))
        self.matchup = tuple(map(int, rng.integers(-3, 4, len(OPEN))))
        self.random = rng

    def sample(self):
        intent = 'attack' if self.random.random() < self.root.attack_probability else 'brace'
        return replace(self.root, world=World(intent, int(self.random.choice(SHOCKS)), self.matchup))
