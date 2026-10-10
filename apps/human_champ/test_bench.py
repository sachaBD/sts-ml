"""Offline converter and paired-report tests."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from apps.human_champ import bench as b
from apps.megacrit_dump.test_champ_starts import BASE


class TestBench(unittest.TestCase):
    def event(self):
        return copy.deepcopy(BASE)

    def test_convert_preserves_deck_hp_and_no_potions(self):
        ev = self.event()
        row = b.reconstruct(ev)
        start = b.convert(ev, row, 123)
        cards = b.enum_ids('Cards.h', 'CardId')
        self.assertEqual(sorted(c['id'] for c in start['deck']), sorted(cards[c['card']] for c in row['deck']))
        self.assertEqual((start['hp'], start['max_hp']), (50, 93))
        self.assertEqual(start['potions'], [1, 1])
        self.assertEqual(start['encounter'], 39)
        self.assertEqual(start['bottled'], [-1, -1, -1])
        self.assertNotEqual(start['misc_rng']['seed0'], 0)
        self.assertEqual(b.rng_state(123), b.rng_state(123))

    def test_uncertain_and_dome_skip(self):
        for relic, reason in [('Runic Dome', 'runic_dome'), ('Lizard Tail', 'unknown_lizard_tail_usage'),
                              ('Bottled Flame', 'reconstruction')]:
            ev = self.event()
            ev['relics'].append(relic)
            with self.assertRaisesRegex(ValueError, reason):
                b.convert(ev, b.reconstruct(ev), 123)

    def test_misc_and_upgrade(self):
        ev = self.event()
        ev['master_deck'].append('Searing Blow+4')
        row = b.reconstruct(ev)
        start = b.convert(ev, row, 123)
        card = next(c for c in start['deck'] if c['id'] == b.enum_ids('Cards.h', 'CardId')['searing_blow'])
        self.assertEqual(card['misc'], 4)
        self.assertTrue(card['upgraded'])
        ev['master_deck'].append('RitualDagger')
        with self.assertRaises(ValueError):
            b.convert(ev, b.reconstruct(ev), 123)

    def test_curse_count_and_girya(self):
        ev = self.event()
        ev['relics'] += ['Du-Vu Doll', 'Girya']
        ev['master_deck'] += ['AscendersBane', 'Regret']
        ev['campfire_choices'] = [{'floor': 2, 'key': 'LIFT'}, {'floor': 3, 'key': 'LIFT'}]
        start = b.convert(ev, b.reconstruct(ev), 123)
        ids = b.enum_ids('Relics.h', 'RelicId')
        data = {r['id']: r['data'] for r in start['relics']}
        self.assertEqual(data[ids['du_vu_doll']], 2)
        self.assertEqual(data[ids['girya']], 2)

    def test_report_requires_both_seeds_both_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema = b.load('human_champ_bench_v1').STARTS
            start = b.convert(self.event(), b.reconstruct(self.event()), 123)
            rows = [dict(fight_id=f'p:{k}', deck_id='p', human_won=True, seed_kind=k,
                         start=start, demon_form=False, scaling_count=0, hp_band='mid',
                         changed_cards=[], reset_counters=[]) for k in ['human', 'fresh']]
            pq.write_table(pa.Table.from_pylist(rows, schema), root / 'starts.parquet')
            for name in ['teacher', 'pv']:
                (root / name).mkdir()
                (root / name / 'results.jsonl').write_text(''.join(json.dumps(dict(fight_id=r['fight_id'],
                    status='completed', fight={'won': False})) + '\n' for r in rows))
            args = SimpleNamespace(starts=root / 'starts.parquet', teacher=root / 'teacher', pv=root / 'pv', out=root / 'report')
            b.report(args)
            self.assertEqual(pq.read_table(root / 'report/viewer_starts.parquet').num_rows, 2)
            (root / 'pv/results.jsonl').write_text(json.dumps(dict(fight_id='p:human', status='error')) + '\n')
            b.report(args)
            self.assertEqual(json.loads((root / 'report/summary.json').read_text())['paired_decks'], 0)


if __name__ == '__main__':
    unittest.main()
