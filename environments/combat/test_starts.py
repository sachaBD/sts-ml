"""Invariants of starts drawn from a pool, for any pool, name and variation."""
from statistics import median
import unittest

from .starts import EMPTY_POTION_SLOT, IRONCLAD_POTIONS, POTION_BELT, Starts

POOL = [dict(fight_id=name, start=dict(hp=hp, max_hp=80, seed=0, ascension=20, relics=relics, potion_capacity=2,
                                       potions=[EMPTY_POTION_SLOT] * 2))
        for name, hp, relics in (('low', 2, []), ('mid', 40, []), ('full', 80, [dict(id=POTION_BELT, data=0)]))]
DECKS = {d['fight_id']: d['start'] for d in POOL}
SEEDS = {'seed', 'misc_rng', 'potion_rng'}
VARIED = SEEDS | {'hp', 'potion_capacity', 'potions'}


def deck(fight):
    return DECKS[fight['fight_id'].split(':')[-2]]


class DrawInvariants(unittest.TestCase):
    def test_fights_are_freshly_seeded_pool_decks(self):
        fights = Starts(hp_jitter=0.1, potions=True).draw(POOL, 'ns:1', 300)
        self.assertEqual(len(fights), 300)
        self.assertEqual(len({f['fight_id'] for f in fights}), len(fights))
        self.assertEqual(len({f['start']['seed'] for f in fights}), len(fights))
        stable = lambda start: {k: v for k, v in start.items() if k not in VARIED}
        for f in fights:
            self.assertEqual(stable(f['start']), stable(deck(f)))

    def test_an_unvaried_start_is_its_deck(self):
        for f in Starts().draw(POOL, 'ns:1', 100):
            self.assertEqual({k: v for k, v in f['start'].items() if k not in SEEDS},
                             {k: v for k, v in deck(f).items() if k not in SEEDS})

    def test_a_name_always_draws_the_same_fights(self):
        self.assertEqual(Starts(hp_jitter=0.1).draw(POOL, 'ns:1', 50), Starts(hp_jitter=0.1).draw(POOL, 'ns:1', 50))

    def test_hp_stays_playable(self):
        for hp_jitter in (0, 0.1, 2):
            for f in Starts(hp_jitter).draw(POOL, 'ns:1', 300):
                self.assertTrue(1 <= f['start']['hp'] <= f['start']['max_hp'])
                if not hp_jitter:
                    self.assertEqual(f['start']['hp'], deck(f)['hp'])

    def test_half_of_fights_move_hp_by_more_than_hp_jitter(self):
        fights = Starts(hp_jitter=0.1).draw(POOL[1:2], 'ns:1', 4000)
        self.assertAlmostEqual(median(abs(f['start']['hp'] - 40) for f in fights), 0.1 * 80, delta=1)

    def test_potion_slots_hold_ironclad_potions_all_but_a_few_full(self):
        ironclad = {p for rarity in IRONCLAD_POTIONS.values() for p in rarity}
        fights = Starts(potions=True).draw(POOL, 'ns:1', 4000)
        for f in fights:
            start = f['start']
            belt = any(r['id'] == POTION_BELT for r in start['relics'])
            self.assertEqual(len(start['potions']), start['potion_capacity'])
            self.assertEqual(start['potion_capacity'], 4 if belt else 2)
            self.assertLessEqual(start['potions'].count(EMPTY_POTION_SLOT), 3)
            self.assertTrue(set(start['potions']) <= ironclad | {EMPTY_POTION_SLOT})
        full = sum(EMPTY_POTION_SLOT not in f['start']['potions'] for f in fights) / len(fights)
        self.assertAlmostEqual(full, 0.75, delta=0.03)


if __name__ == '__main__':
    unittest.main()
