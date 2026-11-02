"""Invariants of the inputs built from human runs that reached Champ, for any runs and namespace."""
import re
import unittest

from sts_lightspeed.enums import POTIONS, RELICS

from .make_fights import EXCLUDED_RELICS, megacrit_fights
from .reporting import deck_of

EMPTY = POTIONS.index('EMPTY_POTION_SLOT')


def row(i, relics=()):
    start = dict(ascension=20, hp=40, max_hp=80, seed=0, relics=[dict(id=r, data=0) for r in relics],
                 potion_capacity=2, potions=[EMPTY] * 2, deck=[dict(id=i % 7, upgraded=False, misc=0)])
    return [dict(play_id=f'{i:04x}-run', copy=copy, start=start) for copy in (1, 2)]


ROWS = [r for i in range(60) for r in row(i, [RELICS.index('RUNIC_DOME')] if i % 10 == 0 else [])]


class MegacritInvariants(unittest.TestCase):
    def test_every_kept_deck_is_in_exactly_one_split(self):
        training, validation, final = megacrit_fights(ROWS, 'ns', 5, 3)
        held = [{deck_of(f['fight_id']) for f in split} for split in (validation, final)]
        decks = [{f['fight_id'] for f in training}, *held]
        self.assertEqual([len(d) for d in decks], [44, 5, 5])
        self.assertFalse(decks[0] & decks[1] or decks[0] & decks[2] or decks[1] & decks[2])
        kept = {f'mc-{r["play_id"]}' for r in ROWS if not {x['id'] for x in r['start']['relics']} & EXCLUDED_RELICS}
        self.assertEqual(set().union(*decks), kept)
        self.assertTrue(all(re.fullmatch(r'[a-z0-9_-]+', d) for d in kept))

    def test_held_out_decks_play_per_deck_fresh_fights_with_potions(self):
        _, validation, final = megacrit_fights(ROWS, 'ns', 5, 3)
        for split in (validation, final):
            self.assertEqual(len(split), 15)
            self.assertEqual(len({f['fight_id'] for f in split}), 15)
            self.assertEqual(len({f['start']['seed'] for f in split}), 15)
            self.assertTrue(all(len(f['start']['potions']) == f['start']['potion_capacity'] for f in split))
        self.assertTrue(any(EMPTY not in f['start']['potions'] for f in validation))

    def test_the_split_follows_the_namespace(self):
        self.assertEqual(megacrit_fights(ROWS, 'ns', 5, 3), megacrit_fights(ROWS, 'ns', 5, 3))
        self.assertNotEqual(megacrit_fights(ROWS, 'ns', 5, 3)[1], megacrit_fights(ROWS, 'other', 5, 3)[1])

    def test_too_few_decks_to_hold_out(self):
        with self.assertRaises(ValueError):
            megacrit_fights(ROWS, 'ns', 27, 3)


if __name__ == '__main__':
    unittest.main()
