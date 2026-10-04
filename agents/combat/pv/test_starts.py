import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from apps.pv import starts

class BenchStartsTest(unittest.TestCase):
    def test_fresh_complete_copies_and_guards(self):
        start=json.loads((Path(__file__).with_name('testdata')/'champ.json').read_text())['start']
        source=[{'fight_id':'a','start':start},{'fight_id':'b','start':copy.deepcopy(start)}]
        source[1]['start']['hp']-=1
        dome=copy.deepcopy(source[0]);dome['fight_id']='dome';dome['start']['relics'].append({'id':57,'data':0})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.parquet';out=Path(tmp)/'fresh.parquet'
            pq.write_table(pa.Table.from_pylist(source+[dome],schema=pa.schema([('fight_id',pa.string()),('start',starts.start_type())])),path)
            a=argparse.Namespace(source=str(path),out=out,encounter=39,bench_copies=5,n=10,seed0=998000000000,
                add_p=0,remove_p=0,hp_p=0,exclude_seed_min=None,exclude_seed_max=None)
            starts.generate(a);rows=pq.ParquetFile(out).read().to_pylist()
            self.assertEqual(len(rows),10);self.assertEqual(len({r['fight_id'] for r in rows}),10)
            self.assertEqual([r['start']['seed'] for r in rows],list(range(a.seed0,a.seed0+10)))
            for src in source:
                copies=[r for r in rows if r['source_fight_id']==src['fight_id']]
                self.assertEqual(len(copies),5)
                for row in copies:
                    row['start']['seed']=src['start']['seed'];self.assertEqual(row['start'],src['start']);self.assertEqual(row['augment'],'')
            a.n=11
            with self.assertRaises(ValueError):starts.generate(a)
            a.n=10;a.add_p=.3
            with self.assertRaises(ValueError):starts.generate(a)
            a.add_p=0;a.exclude_seed_min=981000000000
            with self.assertRaises(ValueError):starts.generate(a)
            a.exclude_seed_min=None;a.bench_copies=None
            with self.assertRaises(ValueError):starts.generate(a)
if __name__=='__main__':unittest.main()
