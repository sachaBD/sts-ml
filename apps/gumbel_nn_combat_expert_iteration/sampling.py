"""Which decks self-play trains on: win-uncertainty weights, history windows, persisted policies."""
import json
import math

from apps.gumbel_nn_combat_expert_iteration.files import write_json
from .deck_stats import iteration_counts


def deck_distribution(counts, prior_strength, concentration):
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


def sampling_weights(training, iteration, settings):
    """Persist before collection, outside archived iteration dirs, so recovery reuses the draw policy."""
    path = training / 'sampling' / f'iter{iteration:03d}.json'
    if not path.exists():
        latest = iteration - 1
        counts = iteration_counts(training, latest, settings.history_iterations)
        decks = deck_distribution(counts, settings.win_rate_prior_strength, settings.sampling_concentration)
        path.parent.mkdir(exist_ok=True)
        write_json(path, dict(first_iteration=max(1, iteration - settings.history_iterations),
                              last_iteration=latest,
                              history_iterations=settings.history_iterations,
                              win_rate_prior_strength=settings.win_rate_prior_strength,
                              sampling_concentration=settings.sampling_concentration,
                              decks=decks))
    return {deck: d['probability'] for deck, d in json.loads(path.read_text())['decks'].items()}
