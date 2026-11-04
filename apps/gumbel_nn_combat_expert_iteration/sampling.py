"""App-owned history windows and persisted adaptive self-play policies."""
import json

from agents.combat.sampling import deck_distribution
from apps.common.app import write_json
from .deck_stats import iteration_counts


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
