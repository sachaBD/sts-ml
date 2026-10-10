import json
from pathlib import Path
import tempfile
import unittest

from apps.human_champ.library import CARDS, bucket, catalogue, results, select


class LibraryTests(unittest.TestCase):
    def test_bucket(self):
        reverse={v:k for k,v in CARDS.items()}
        row={'start':{'deck':[{'id':reverse['barricade']},{'id':reverse['demon_form']}]}}
        self.assertEqual(bucket(row),'block')

    def test_catalogue_excludes_errors_and_tolerates_partial_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'play').mkdir()
            (out/'manifest.json').write_text(json.dumps([{'fight_ids':['a','b','c']}]))
            (out/'play/results.jsonl').write_text(json.dumps({'fight_id':'a','status':'completed','fight':{'won':True}})+'\n'+json.dumps({'fight_id':'b','status':'error'})+'\n{"fight_id":')
            row=catalogue(out)[0]
            self.assertEqual((row['logged'],row['completed'],row['wins']),(2,1,1))
            self.assertEqual(row['statuses'],{'completed':1,'error':1})
            self.assertIsNotNone(row['confidence_95'])
            self.assertEqual(len(results(out/'play/results.jsonl')),2)


if __name__=='__main__':unittest.main()
