"""Combat training selection: smoothed win uncertainty mixed with uniform sampling."""
import math


def deck_distribution(counts, prior_strength, fraction):
    """Smoothed win uncertainty mixed with uniform selection; retain unobserved decks."""
    decks = {}
    for deck, c in counts.items():
        attempts, wins = c['attempts'], c['wins']
        if not 0 <= wins <= attempts:
            raise ValueError(f'{deck}: require 0 <= wins <= attempts')
        # Halving first avoids overflow for large priors; separate loss rate avoids cancellation.
        denominator = attempts / 2 + prior_strength
        win_rate = (wins / 2 + prior_strength / 2) / denominator
        loss_rate = ((attempts - wins) / 2 + prior_strength / 2) / denominator
        decks[deck] = dict(c, weight=win_rate * loss_rate)
    total = math.fsum(d['weight'] for d in decks.values())
    for d in decks.values():
        d['probability'] = (1 - fraction) / len(decks) + fraction * d['weight'] / total
    return decks
