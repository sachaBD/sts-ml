"""Deck weights: practise the decks whose outcome is most uncertain (win rate near 50%)."""
import math


def deck_weights(counts, prior_strength, concentration):
    """Concentration 0 is uniform; 1 selects only maximum-uncertainty decks (ties uniform)."""
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
    peak = max(d['weight'] for d in decks.values())
    for d in decks.values():
        # Scale before exponentiation so at least one weight stays 1, even near concentration 1.
        d['weight'] = (float(d['weight'] == peak) if concentration == 1 else
                       (d['weight'] / peak) ** (concentration / (1 - concentration)))
    total = math.fsum(d['weight'] for d in decks.values())
    for d in decks.values():
        d['probability'] = d['weight'] / total
    return decks
