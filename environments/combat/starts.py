"""Combat starts (combat_v4 `start` objects): fresh-seeded copies, seeded the way the simulator seeds them, and
draws from a pool of decks with their starts varied."""
import copy
from dataclasses import dataclass
import hashlib
import random
from statistics import NormalDist

from sts_lightspeed.enums import CARDS, POTIONS, RELICS

MASK = (1 << 64) - 1
# A normal draw lies beyond this many standard deviations half the time.
MEDIAN_MOVE = NormalDist().inv_cdf(0.75)

EMPTY_POTION_SLOT, POTION_BELT = POTIONS.index('EMPTY_POTION_SLOT'), RELICS.index('POTION_BELT')
# The game's potion rarity odds, each with Ironclad's potions of that rarity; the simulator rerolls Smoke Bomb.
IRONCLAD_POTIONS = (
    (65, ('ATTACK_POTION', 'BLESSING_OF_THE_FORGE', 'BLOCK_POTION', 'BLOOD_POTION', 'COLORLESS_POTION',
          'DEXTERITY_POTION', 'ENERGY_POTION', 'EXPLOSIVE_POTION', 'FEAR_POTION', 'FIRE_POTION', 'FLEX_POTION',
          'POWER_POTION', 'SKILL_POTION', 'SPEED_POTION', 'STRENGTH_POTION', 'SWIFT_POTION', 'WEAK_POTION')),
    (25, ('ANCIENT_POTION', 'DISTILLED_CHAOS', 'DUPLICATION_POTION', 'ELIXIR_POTION', 'ESSENCE_OF_STEEL',
          'GAMBLERS_BREW', 'LIQUID_BRONZE', 'LIQUID_MEMORIES', 'REGEN_POTION')),
    (10, ('CULTIST_POTION', 'ENTROPIC_BREW', 'FAIRY_POTION', 'FRUIT_JUICE', 'HEART_OF_IRON', 'SNECKO_OIL')),
)
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
    excluded_potions: tuple[str, ...] = ()  # never drawn into potion slots
    excluded_relics: tuple[str, ...] = ()  # decks holding any of these are left out of the pool
    excluded_cards: tuple[str, ...] = ()  # decks holding any of these are left out of the pool

    def __post_init__(self):
        for names, known in ((self.excluded_potions, POTIONS), (self.excluded_relics, RELICS), (self.excluded_cards, CARDS)):
            if unknown := sorted(set(names) - set(known)):
                raise ValueError(f'unknown names: {unknown}')

    def draw(self, pool, name, n, weights=None):
        """Draw with replacement (uniformly unless weighted), excluding forbidden decks, then vary starts."""
        if self.excluded_relics or self.excluded_cards:
            kept = [i for i, d in enumerate(pool) if not self.excludes(d['start'])]
            pool = [pool[i] for i in kept]
            if weights is not None:
                weights = [weights[i] for i in kept]
            if not pool:
                raise ValueError('every deck in the pool holds an excluded relic or card')
        rng = random.Random(name)
        picks = rng.choices(pool, weights=weights, k=n)
        fights = [seeded(deck, f'{name}:{deck["fight_id"]}:{i}') for i, deck in enumerate(picks)]
        for f in fights:
            self.vary(f['start'], rng)
        return fights

    def excludes(self, start):
        """Whether start's deck holds an excluded relic or card."""
        relics = {RELICS.index(r) for r in self.excluded_relics}
        cards = {CARDS.index(c) for c in self.excluded_cards}
        return any(r['id'] in relics for r in start['relics']) or any(c['id'] in cards for c in start['deck'])

    def vary(self, start, rng):
        """Each variation of a start, in turn."""
        jitter_hp(start, rng, self.hp_jitter)
        if self.potions:
            fill_potions(start, rng, self.excluded_potions)


def jitter_hp(start, rng, median):
    """HP normal around start's, moving by more than median x max HP half the time; kept within [1, max HP]."""
    sd = median / MEDIAN_MOVE * start['max_hp']
    start['hp'] = min(start['max_hp'], max(1, round(rng.gauss(start['hp'], sd))))


def fill_potions(start, rng, excluded=()):
    """The game's potion slots (2 from Ascension 11, +2 with Potion Belt), all but a few empty ones holding potions
    drawn as the game draws them, never an excluded one."""
    belt = any(r['id'] == POTION_BELT for r in start['relics'])
    slots = (2 if start['ascension'] >= 11 else 3) + 2 * belt
    held = max(0, slots - rng.choices(list(EMPTY_SLOTS), weights=list(EMPTY_SLOTS.values()))[0])
    allowed = [(weight, kept) for weight, rarity in IRONCLAD_POTIONS if (kept := [p for p in rarity if p not in excluded])]
    weights, rarities = zip(*allowed)
    potions = [POTIONS.index(rng.choice(rarity)) for rarity in rng.choices(rarities, weights, k=held)]
    start.update(potion_capacity=slots, potions=potions + [EMPTY_POTION_SLOT] * (slots - held))
