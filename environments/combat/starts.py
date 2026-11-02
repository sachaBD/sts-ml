"""Combat starts (combat_v4 `start` objects): fresh-seeded copies, seeded the way the simulator seeds them, and
draws from a pool of decks with their starts varied."""
import copy
from dataclasses import dataclass
import hashlib
import random
from statistics import NormalDist

MASK = (1 << 64) - 1
# A normal draw lies beyond this many standard deviations half the time.
MEDIAN_MOVE = NormalDist().inv_cdf(0.75)


def rng_state(seed):
    """sts::Random(seed) before any draw (libgdx RandomXS128 seeding)."""
    def murmur(x):
        x ^= x >> 33; x = x * (-49064778989728563 & MASK) & MASK
        x ^= x >> 33; x = x * (-4265267296055464877 & MASK) & MASK
        return (x ^ (x >> 33)) & MASK
    first = murmur(seed if seed else 1 << 63)
    return {'counter': 0, 'seed0': first, 'seed1': murmur(first)}


def seeded(base, fight_id):
    """A copy of base ({fight_id, start}) named fight_id, with a fresh seed derived from that name."""
    seed = int.from_bytes(hashlib.sha256(fight_id.encode()).digest()[:8], 'little')
    f = copy.deepcopy(base)
    f['fight_id'] = fight_id
    f['start'].update(seed=seed, misc_rng=rng_state(seed), potion_rng=rng_state(seed))
    return f


def fights(base, name, n):
    """n copies of base with fresh seeds derived from name."""
    return [seeded(base, f'{name}:{i}') for i in range(n)]


@dataclass(frozen=True)
class Starts:
    hp_jitter: float = 0.0  # half of starts move their deck's HP by more than this fraction of max HP

    def draw(self, pool, name, n):
        """n fights named name:deck:i, each on a deck ({fight_id: deck, start}) drawn uniformly, with replacement,
        from pool, then varied."""
        rng = random.Random(name)
        picks = rng.choices(pool, k=n)
        fights = [seeded(deck, f'{name}:{deck["fight_id"]}:{i}') for i, deck in enumerate(picks)]
        for f in fights:
            self.vary(f['start'], rng)
        return fights

    def vary(self, start, rng):
        """Each variation of a start, in turn."""
        jitter_hp(start, rng, self.hp_jitter)


def jitter_hp(start, rng, median):
    """HP normal around start's, moving by more than median x max HP half the time; kept within [1, max HP]."""
    sd = median / MEDIAN_MOVE * start['max_hp']
    start['hp'] = min(start['max_hp'], max(1, round(rng.gauss(start['hp'], sd))))
