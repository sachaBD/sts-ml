"""Invariants of starts drawn from a pool, for any pool, name and variation."""
from statistics import median
import unittest

from .starts import Starts

POOL = [dict(fight_id=name, start=dict(hp=hp, max_hp=80, seed=0)) for name, hp in (('low', 2), ('mid', 40), ('full', 80))]
DECKS = {d['fight_id']: d['start'] for d in POOL}
VARIED = {'seed', 'misc_rng', 'potion_rng', 'hp'}


def deck(fight):
    return DECKS[fight['fight_id'].split(':')[-2]]


class DrawInvariants(unittest.TestCase):
    def test_fights_are_freshly_seeded_pool_decks(self):
        fights = Starts(hp_jitter=0.1).draw(POOL, 'ns:1', 300)
        self.assertEqual(len(fights), 300)
        self.assertEqual(len({f['fight_id'] for f in fights}), len(fights))
        self.assertEqual(len({f['start']['seed'] for f in fights}), len(fights))
        stable = lambda start: {k: v for k, v in start.items() if k not in VARIED}
        for f in fights:
            self.assertEqual(stable(f['start']), stable(deck(f)))

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


if __name__ == '__main__':
    unittest.main()
