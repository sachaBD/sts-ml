"""Each iteration's deck sampling policy: past deck counts in, curriculum weights saved before collection."""
import json

from apps.common.app import write_json
from agents.combat.curriculum.deck_weights import deck_weights


def iteration_counts(training, latest: int, recent_most: int):
    """Per-deck counts over an explicit window ending at latest (zero is initialization)."""
    if recent_most < 1 or latest < 0:
        raise ValueError('positive recent_most and nonnegative latest required')
    before = json.loads((training / f'iter{max(0, latest - recent_most):03d}' / 'decks.json').read_text())
    after = json.loads((training / f'iter{latest:03d}' / 'decks.json').read_text())
    return {deck: dict(attempts=stats['fights'] - before[deck]['fights'],
                       wins=stats['wins'] - before[deck]['wins'])
            for deck, stats in after.items()}


def sampling_weights(training, iteration, settings):
    """Persist before collection, outside archived iteration dirs, so recovery reuses the draw policy."""
    path = training / 'sampling' / f'iter{iteration:03d}.json'
    if not path.exists():
        latest = iteration - 1
        counts = iteration_counts(training, latest, settings.history_iterations)
        decks = deck_weights(counts, settings.win_rate_prior_strength, settings.sampling_concentration)
        path.parent.mkdir(exist_ok=True)
        write_json(path, dict(first_iteration=max(1, iteration - settings.history_iterations),
                              last_iteration=latest,
                              history_iterations=settings.history_iterations,
                              win_rate_prior_strength=settings.win_rate_prior_strength,
                              sampling_concentration=settings.sampling_concentration,
                              decks=decks))
    return {deck: d['probability'] for deck, d in json.loads(path.read_text())['decks'].items()}
