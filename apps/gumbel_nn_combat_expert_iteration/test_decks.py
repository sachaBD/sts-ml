"""Invariants of self-play's fights, for any pool, name and variation."""
from statistics import median
import unittest

from .decks import Decks
from .reporting import deck_of

POOL = [dict(fight_id=name, start=dict(hp=hp, max_hp=80, seed=0)) for name, hp in (('low', 2), ('mid', 40), ('full', 80))]
VARIED = {'seed', 'misc_rng', 'potion_rng', 'hp'}


class DecksInvariants(unittest.TestCase):
    def test_fights_are_freshly_seeded_pool_decks(self):
        decks = {d['fight_id']: d['start'] for d in POOL}
        fights = Decks('unused', 300, hp_jitter=0.1).draw(POOL, 'ns:1')
        self.assertEqual(len(fights), 300)
        self.assertEqual(len({f['fight_id'] for f in fights}), len(fights))
        self.assertEqual(len({f['start']['seed'] for f in fights}), len(fights))
        stable = lambda start: {k: v for k, v in start.items() if k not in VARIED}
        for f in fights:
            self.assertEqual(stable(f['start']), stable(decks[deck_of(f['fight_id'])]))

    def test_a_name_always_draws_the_same_fights(self):
        decks = Decks('unused', 50, hp_jitter=0.1)
        self.assertEqual(decks.draw(POOL, 'ns:1'), decks.draw(POOL, 'ns:1'))

    def test_hp_stays_playable(self):
        decks = {d['fight_id']: d['start'] for d in POOL}
        for hp_jitter in (0, 0.1, 2):
            for f in Decks('unused', 300, hp_jitter).draw(POOL, 'ns:1'):
                start = f['start']
                self.assertTrue(1 <= start['hp'] <= start['max_hp'])
                if not hp_jitter:
                    self.assertEqual(start['hp'], decks[deck_of(f['fight_id'])]['hp'])

    def test_half_of_fights_move_hp_by_more_than_hp_jitter(self):
        fights = Decks('unused', 4000, hp_jitter=0.1).draw(POOL[1:2], 'ns:1')
        self.assertAlmostEqual(median(abs(f['start']['hp'] - 40) for f in fights), 0.1 * 80, delta=1)


if __name__ == '__main__':
    unittest.main()
