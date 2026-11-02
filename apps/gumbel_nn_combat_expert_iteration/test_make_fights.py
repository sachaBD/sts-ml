"""Invariants of a deck split, for any decks and namespace."""
import unittest

from sts_lightspeed.enums import POTIONS

from .make_fights import split
from .reporting import deck_of

EMPTY = POTIONS.index('EMPTY_POTION_SLOT')
DECKS = [dict(fight_id=f'mc-{i:04x}', start=dict(ascension=20, hp=40, max_hp=80, seed=0, relics=[], potion_capacity=2,
                                                 potions=[EMPTY] * 2, deck=[dict(id=i % 7, upgraded=False, misc=0)]))
         for i in range(50)]


class SplitInvariants(unittest.TestCase):
    def test_every_deck_is_in_exactly_one_part(self):
        training, validation, final = split(DECKS, 'ns', 5, 3)
        parts = [{f['fight_id'] for f in training}, *({deck_of(f['fight_id']) for f in p} for p in (validation, final))]
        self.assertEqual([len(p) for p in parts], [40, 5, 5])
        self.assertEqual(set().union(*parts), {d['fight_id'] for d in DECKS})
        self.assertTrue(all(d in DECKS for d in training))  # training decks are unchanged

    def test_held_out_decks_play_per_deck_fresh_fights_with_potions(self):
        _, validation, final = split(DECKS, 'ns', 5, 3)
        for part in (validation, final):
            self.assertEqual(len({f['fight_id'] for f in part}), 15)
            self.assertEqual(len({f['start']['seed'] for f in part}), 15)
            self.assertTrue(all(len(f['start']['potions']) == f['start']['potion_capacity'] for f in part))
        self.assertTrue(any(EMPTY not in f['start']['potions'] for f in validation))

    def test_the_split_follows_the_namespace(self):
        self.assertEqual(split(DECKS, 'ns', 5, 3), split(DECKS, 'ns', 5, 3))
        self.assertNotEqual(split(DECKS, 'ns', 5, 3)[1], split(DECKS, 'other', 5, 3)[1])

    def test_too_few_decks_to_hold_out(self):
        with self.assertRaises(ValueError):
            split(DECKS, 'ns', 25, 3)


if __name__ == '__main__':
    unittest.main()
