"""Offline checks for combat-loop split isolation and paired promotion accounting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from apps.run_rl import combat_loop as c

ROOT = Path(__file__).resolve().parents[2]


def row(pid, card=1, seed=123, kind='human'):
    start = json.loads((ROOT / 'agents/combat/pv/testdata/champ.json').read_text())['start']
    start['deck'][0]['id'] = card
    start['seed'] = seed
    return dict(fight_id=f'{pid}:{kind}', deck_id=pid, seed_kind=kind, start=start)


class CombatLoop(unittest.TestCase):
    def test_deck_families_and_heldout_never_leak(self):
        a = row('a')
        a2 = row('a2', seed=456)
        a2['start']['hp'] -= 10
        original = [a, a2, row('a', seed=789, kind='fresh'), row('b', card=2), row('d', card=3),
                    row('heldout', card=99), row('heldout-clone', card=88)]
        heldout = [row('heldout', card=88)]
        pool, mapping = c.partition(original, heldout, .5)
        self.assertEqual({r['deck_id'] for r in pool}, {'a', 'a2', 'b', 'd'})
        self.assertEqual(mapping[a['fight_id']], mapping[a2['fight_id']])
        self.assertEqual(set(mapping.values()), {'train', 'val'})
        self.assertEqual(c.partition(original, heldout, .5), (pool, mapping))
        with self.assertRaises(ValueError):
            c.partition([a, a2], [])

    def test_teacher_selects_only_completed_training_fights(self):
        rows = [row(str(i), card=i+1) for i in range(8)]
        results = {r['fight_id']: dict(status='completed', fight={'won': i > 2}) for i, r in enumerate(rows)}
        results[rows[7]['fight_id']]['status'] = 'error'
        selected = c.choose_teacher(rows, results, 4)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(r['fight_id'].endswith(':teacher') for r in selected))
        self.assertNotIn('7', {r['deck_id'] for r in selected})
        self.assertEqual(len({r['deck_id'] for r in selected}), 4)
        self.assertEqual(c.choose_teacher(rows, results, 0), [])

    def test_comparison_requires_all_deck_seeds_and_reports_se(self):
        rows = [row('a', card=1), row('a', card=1, kind='fresh'), row('b', card=2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, outcomes in [('base', [False, True, False]), ('new', [True, True, False])]:
                d = root / arm
                d.mkdir()
                (d / 'results.jsonl').write_text(''.join(json.dumps(dict(fight_id=r['fight_id'],
                    status='completed', fight={'won': won}))+'\n' for r, won in zip(rows, outcomes)))
            result = c.comparison(rows, root / 'base', root / 'new')
            self.assertEqual(result['paired_decks'], 2)
            self.assertEqual(result['difference'], .25)
            self.assertEqual(result['se'], .25)
            results = c.read_results(root / 'new')
            results['a:fresh']['status'] = 'capped'
            (root / 'new/results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results.values()))
            result = c.comparison(rows, root / 'base', root / 'new')
            self.assertEqual(result['paired_decks'], 1)
            self.assertIsNone(result['se'])


if __name__ == '__main__':
    unittest.main()
