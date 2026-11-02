"""Self-play's decks: each fight draws a deck from the pool, then varies its start. Per-deck fights and wins
(`tally`) are what later deck choice can learn from."""
from dataclasses import dataclass
import random
from statistics import NormalDist

from environments.combat.starts import seeded

from .reporting import deck_of

# A normal draw lies beyond this many standard deviations half the time.
MEDIAN_MOVE = NormalDist().inv_cdf(0.75)


@dataclass(frozen=True)
class Decks:
    fights: str  # the pool: one fight per deck, fight_id = deck name
    per_iteration: int
    hp_jitter: float = 0.0  # half of fights move their deck's HP by more than this fraction of max HP

    def draw(self, pool, name):
        """per_iteration fights named name:deck:i, each on a deck drawn uniformly, with replacement, from pool."""
        rng = random.Random(name)
        picks = rng.choices(pool, k=self.per_iteration)
        fights = [seeded(deck, f'{name}:{deck["fight_id"]}:{i}') for i, deck in enumerate(picks)]
        for f in fights:
            self.vary(f['start'], rng)
        return fights

    def vary(self, start, rng):
        """Each variation of a fight's start, in turn."""
        jitter_hp(start, rng, self.hp_jitter)


def jitter_hp(start, rng, median):
    """HP normal around start's, moving by more than median x max HP half the time; kept within [1, max HP]."""
    sd = median / MEDIAN_MOVE * start['max_hp']
    start['hp'] = min(start['max_hp'], max(1, round(rng.gauss(start['hp'], sd))))


def tally(counts, results):
    """A copy of counts ({deck: {fights, wins}}) with results added."""
    counts = {deck: dict(c) for deck, c in counts.items()}
    for r in results:
        c = counts[deck_of(r['fight_id'])]
        c['fights'] += 1
        c['wins'] += bool(r['won'])
    return counts
