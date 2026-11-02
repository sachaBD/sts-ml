"""Partition invariants; no simulator or outcomes involved."""
from collections import Counter
import unittest

from .prepare import composition, legacy_fights, partition


def deck(i, name=None):
    return dict(fight_id=name or f'deck-{i}', start=dict(hp=40, max_hp=80, deck=[dict(id=i, upgraded=False, misc=0)]))


class PreparationTests(unittest.TestCase):
    def test_compositions_never_cross_parts_and_all_candidates_are_accounted_for(self):
        pool = [deck(i) for i in range(2100)] + [deck(i, f'other-{i}') for i in range(60)]
        legacy = [deck(i, f'legacy-{i}') for i in range(50)]
        train, new, final, withheld, matches = partition(pool, legacy)
        groups = [{composition(f) for f in part} for part in (train, new, final, legacy)]
        self.assertEqual(len(groups[1]), 950)
        self.assertEqual(len(groups[2]), 1000)
        for i, group in enumerate(groups):
            self.assertFalse(any(group & other for other in groups[i + 1:]))
        accounted = [f['fight_id'] for part in (train, new, final, withheld) for f in part]
        accounted += [name for names in matches.values() for name in names]
        self.assertEqual(Counter(accounted), Counter(f['fight_id'] for f in pool))
        self.assertEqual(partition(pool, legacy), partition(list(reversed(pool)), legacy))

    def test_legacy_fights_are_unmodified_recorded_starts(self):
        legacy = [deck(0, 'legacy')]
        recorded = [dict(fight_id=f'history:legacy:validation:{i}', start=dict(hp=i + 1, potions=[1, 1]))
                    for i in range(50)]
        selected = legacy_fights(legacy, recorded)
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(f in recorded for f in selected))
        self.assertEqual(selected, legacy_fights(legacy, list(reversed(recorded))))


if __name__ == '__main__':
    unittest.main()
