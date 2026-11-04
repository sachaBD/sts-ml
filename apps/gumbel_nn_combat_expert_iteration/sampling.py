"""Adaptive self-play selection from this experiment's previous completed iterations."""
import json
import math

from apps.common.app import write_json
from .deck_stats import iteration_counts


def deck_distribution(counts, prior_strength, fraction):
    """Smoothed win uncertainty mixed with uniform selection; retain unobserved decks."""
    if not math.isfinite(prior_strength) or prior_strength <= 0 or not 0 <= fraction <= 1:
        raise ValueError('positive finite prior_strength and fraction in [0, 1] required')
    if not counts:
        raise ValueError('empty deck roster')
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


def sampling_weights(training, iteration, settings):
    """Persist before collection, outside archived iteration dirs, so recovery reuses the draw policy."""
    path = training / 'sampling' / f'iter{iteration:03d}.json'
    if not path.exists():
        latest = iteration - 1
        counts = iteration_counts(training, latest, settings.history_iterations)
        decks = deck_distribution(counts, settings.win_rate_prior_strength, settings.weighted_sampling_fraction)
        path.parent.mkdir(exist_ok=True)
        write_json(path, dict(first_iteration=max(1, iteration - settings.history_iterations),
                              last_iteration=latest,
                              history_iterations=settings.history_iterations,
                              win_rate_prior_strength=settings.win_rate_prior_strength,
                              weighted_sampling_fraction=settings.weighted_sampling_fraction,
                              decks=decks))
    return {deck: d['probability'] for deck, d in json.loads(path.read_text())['decks'].items()}
