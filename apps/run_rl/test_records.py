import json
from pathlib import Path
import tempfile
import unittest
import pyarrow.parquet as pq
from apps.run_rl.records import contract, write_overworld, write_combat
from agents.overworld.value.core import read_runs

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

    def test_cache_is_distinct_from_replay_facts(self):
        c=contract('combat_v4')
        self.assertEqual(c.TABLES['steps'].schema.names,['fight_id','step_index','action_bits'])
        self.assertIn('extra_state_json',c.INITIAL_STATE.names)
        self.assertEqual(c.TABLES['training'].schema.metadata[b'derived'],b'encoding-cache-v1')

    def test_empty_combat_batch_has_named_tables(self):
        with tempfile.TemporaryDirectory() as d:
            write_combat(d,[],1)
            for t in contract('combat_v4').TABLES.values():
                self.assertTrue((Path(d)/f'{t.file_prefix}-1.parquet').exists())

if __name__=='__main__': unittest.main()
