from math import exp

from environments.combat.card_duel import INTENTS, OPEN, RESPOND, margin, value


def prior(logits):
    weights = [exp(x - max(logits)) for x in logits]
    return [w / sum(weights) for w in weights]


def Q(s):
    if s.phase == 'Open':
        attack = s.attack_probability
        estimates = [sum(p * max(value(margin(a, i, r)) for r in RESPOND)
                         for i, p in zip(INTENTS, (attack, 1-attack))) for a in OPEN]
        return max(estimates), prior([v / .15 for v in estimates])
    if s.phase == 'Respond':
        scores = [margin(s.opener, s.intent, r) for r in RESPOND]
        return value(max(scores)), prior(scores)
    if s.phase == 'Next turn':
        return value(s.score), []
    raise ValueError('no evaluation at a forced or terminal state')
