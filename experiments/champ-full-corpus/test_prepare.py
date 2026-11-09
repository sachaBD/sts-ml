"""Partition invariants; no simulator or outcomes involved."""
from collections import Counter
import unittest

from .prepare import partition


def deck(i, name=None, upgraded=False):
    return dict(fight_id=name or f'deck-{i}', start=dict(hp=40, max_hp=80, deck=[dict(id=i, upgraded=upgraded, misc=0)]))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.pool = [deck(i) for i in range(2100)] + [deck(i, f'upgraded-{i}', upgraded=True) for i in range(2100)]
        self.legacy = [deck(i, f'legacy-{i}') for i in range(50)]

    def test_compositions_never_cross_partitions_even_when_upgrades_differ(self):
        train, validation, final, _, _ = partition(self.pool, self.legacy)
        groups = [{tuple(sorted(c['id'] for c in f['start']['deck'])) for f in part}
                  for part in (train, validation, final, self.legacy)]
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                self.assertTrue(group.isdisjoint(other))

    def test_every_candidate_is_accounted_for_exactly_once(self):
        train, validation, final, withheld, matches = partition(self.pool, self.legacy)
        accounted = [f['fight_id'] for part in (train, validation, final, withheld) for f in part]
        accounted += [name for names in matches.values() for name in names]
        self.assertEqual(Counter(accounted), Counter(f['fight_id'] for f in self.pool))

    def test_selection_is_independent_of_input_order(self):
        self.assertEqual(partition(self.pool, self.legacy),
                         partition(list(reversed(self.pool)), list(reversed(self.legacy))))


if __name__ == '__main__':
    unittest.main()
