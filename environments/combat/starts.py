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

EMPTY_POTION_SLOT, POTION_BELT = 1, 159  # sts::Potion, sts::RelicId
# Ironclad's potions (sts::Potion ids) by rarity weight, as the game draws them; the simulator rerolls Smoke Bomb.
IRONCLAD_POTIONS = {
    65: (4, 5, 6, 7, 9, 12, 16, 20, 22, 23, 24, 34, 36, 39, 41, 42, 43),  # common: Attack .. Weak Potion
    25: (3, 13, 14, 15, 19, 27, 30, 31, 35),  # uncommon: Ancient Potion .. Regen Potion
    10: (10, 17, 21, 26, 29, 38),  # rare: Cultist Potion .. Snecko Oil
}
# Empty potion slots entering Champ, fit by eye to ~8.1k A20 Ironclad wins (Mega Crit run histories).
EMPTY_SLOTS = {0: 0.75, 1: 0.15, 2: 0.07, 3: 0.03}


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
    potions: bool = False  # fill potion slots as a human Ironclad enters Champ

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
        if self.potions:
            fill_potions(start, rng)


def jitter_hp(start, rng, median):
    """HP normal around start's, moving by more than median x max HP half the time; kept within [1, max HP]."""
    sd = median / MEDIAN_MOVE * start['max_hp']
    start['hp'] = min(start['max_hp'], max(1, round(rng.gauss(start['hp'], sd))))


def fill_potions(start, rng):
    """The game's potion slots (2 from Ascension 11, +2 with Potion Belt), all but a few empty ones holding potions
    drawn as the game draws them."""
    belt = any(r['id'] == POTION_BELT for r in start['relics'])
    slots = (2 if start['ascension'] >= 11 else 3) + 2 * belt
    held = max(0, slots - rng.choices(list(EMPTY_SLOTS), weights=list(EMPTY_SLOTS.values()))[0])
    rarities = rng.choices(list(IRONCLAD_POTIONS.values()), weights=list(IRONCLAD_POTIONS), k=held)
    potions = [rng.choice(rarity) for rarity in rarities]
    start.update(potion_capacity=slots, potions=potions + [EMPTY_POTION_SLOT] * (slots - held))
