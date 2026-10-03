import json
from pathlib import Path
import tempfile
import unittest
import pyarrow.parquet as pq
from apps.run_rl.records import contract, write_overworld, write_combat
from agents.overworld.value.core import read_runs

START = {'seed': 7, 'ascension': 20, 'act': 1, 'floor': 1, 'encounter': 1, 'cur_room': 4, 'last_room': 8,
         'burning_elite_buff': -1, 'hp': 68, 'max_hp': 75, 'gold': 99,
         'misc_rng': {'counter': 0, 'seed0': 1, 'seed1': 2}, 'potion_rng': {'counter': 0, 'seed0': 3, 'seed1': 4},
         'potion_capacity': 2, 'potions': [1, 1], 'relics': [{'id': 0, 'data': 0}],
         'deck': [{'id': 321, 'upgraded': False, 'misc': 0}], 'bottled': [-1, -1, -1]}


def record(fight_id, searched):
    fight = {'fight_id': fight_id, 'version': 1, 'start': START, 'actions': [1, 2147483648], 'explored': [False, False],
             'won': True, 'final_hp': 60, 'agent': 'run_rl mcts leaf=guided_rollout sims=500 particles=8'}
    search = [{'fight_id': fight_id, 'step': 0, 'agent': 'mcts leaf=guided_rollout sims=500 particles=8',
               'root_value': .5, 'simulations': 500, 'children': [{'action': 1, 'visits': 300, 'value': .6}]}]
    return {'fight': fight, 'search': search if searched else []}


class RecordingTests(unittest.TestCase):
    def test_overworld_facts_and_agent_annotations_separate(self):
        r={'run_key':'collection:42','seed':42,'status':'died','boss':'slime_boss','floor':5,'final_hp':0,
           'steps':[{'kind':'start','state':{'hp':80}}, {'kind':'pick','choice':1,'source':'explore','values':[.2,.3]}]}
        with tempfile.TemporaryDirectory() as d:
            write_overworld(d,r,42)
            actual=read_runs([d])[0]
            self.assertEqual(actual['steps'][1]['choice'],1)
            self.assertNotIn('values',actual['steps'][1])
            self.assertNotIn('source',actual['steps'][1])
            self.assertEqual(pq.read_table(Path(d)/'agent-42.parquet')['source'].to_pylist(),['explore'])
            write_overworld(d,r,42) # idempotent atomic replacement
            self.assertEqual(len(read_runs([d])),1)

    def test_combat_v4_fights_and_search_tables(self):
        c=contract('combat_v4')
        self.assertEqual(set(c.TABLES), {'fights', 'search'})
        with tempfile.TemporaryDirectory() as d:
            write_combat(d,[record('c:7:0',True), record('c:7:1',False)],7)
            fights=pq.read_table(Path(d)/'fights-7.parquet')
            search=pq.read_table(Path(d)/'search-7.parquet')
            self.assertEqual(fights.schema, c.FIGHTS)
            self.assertEqual(fights['fight_id'].to_pylist(), ['c:7:0','c:7:1'])
            self.assertEqual(search['fight_id'].to_pylist(), ['c:7:0'])

    def test_no_search_rows_no_search_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_combat(d,[record('c:8:0',False)],8)
            self.assertTrue((Path(d)/'fights-8.parquet').exists())
            self.assertFalse((Path(d)/'search-8.parquet').exists())

if __name__=='__main__': unittest.main()
