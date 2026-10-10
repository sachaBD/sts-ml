"""Controller gates; no native gameplay or real training."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from .expert_champ_run import Controller, Halt, Ledger, generate, paired, rng_state
from apps.human_champ.bench import rng_state as reference_rng


class ControllerTests(unittest.TestCase):
    def test_rng_matches_existing_generator(self):
        for s in (0,1,123456789,2**64-1):self.assertEqual(rng_state(s),reference_rng(s))

    def test_seed_splits_disjoint_and_collision_fails(self):
        used=set();base={'start':{'seed':0,'hp':34},'fight_id':'old'}
        a=generate(base,'test','train',3,used);b=generate(base,'test','final',3,used)
        self.assertEqual(len(used),6);self.assertEqual(base['start']['seed'],0)
        self.assertTrue(all(x['start']['hp']==34 for x in a+b))
        with self.assertRaises(Halt):generate(base,'test','train',1,used)

    def test_ledger_reuse_orphan_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger';l=Ledger(p)
            self.assertIsNone(l.reusable('x','h'))
            l.append(dict(kind='intent',key='x',request_sha256='h'))
            with self.assertRaises(Halt):l.reusable('x','h')
            l.append(dict(kind='result',key='x',status='completed',line={'won':True}))
            self.assertIsNotNone(l.reusable('x','h'))
            with self.assertRaises(Halt):l.reusable('x','changed')
            with self.assertRaises(Halt):l.append(dict(kind='intent',key='x'))
            l.f.close();l=Ledger(p);self.assertIsNotNone(l.reusable('x','h'));l.f.close()

    def test_partial_tail_and_caps_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger';p.write_text('{"kind":')
            with self.assertRaises(Halt):Ledger(p)
            p.unlink();l=Ledger(p);l.append(dict(kind='intent',key='x',request_sha256='h'))
            l.append(dict(kind='result',key='x',status='capped'))
            with self.assertRaises(Halt):l.reusable('x','h')
            l.f.close()

    def test_paired_counts_and_complete_requirement(self):
        a={str(i):{'won':w} for i,w in enumerate([1,1,0,0])}
        b={str(i):{'won':w} for i,w in enumerate([1,0,1,1])}
        r=paired(a,b);self.assertEqual((r['n'],r['wins'],r['incumbent_wins'],r['candidate_only'],r['incumbent_only']),(4,3,2,2,1))
        self.assertEqual(r['gap'],.25)
        with self.assertRaises(Halt):paired(a,{'0':b['0']})

    def fake_controller(self,path,obj):
        c=object.__new__(Controller);c.stop=threading.Event();c.ledger=Ledger(path);c.cfg={'game_timeout_seconds':5}
        c.execute=lambda *a,**kw:(0,json.dumps(obj),'')
        return c

    def test_game_intent_result_and_flags(self):
        with tempfile.TemporaryDirectory() as d:
            row={'fight_id':'f','start':{'seed':1}}
            obj={'status':'completed','fight':dict(**row,won=True,final_hp=1,actions=[],agent='pv sims=2000 rollout_mix=0.500000'),'search':[]}
            c=self.fake_controller(Path(d)/'ledger',obj)
            r=c.game('key',['fake'],row,{'stage':'smoke'},'h')
            self.assertEqual(r['status'],'completed');self.assertEqual(len(c.ledger.intents),1)
            self.assertEqual(len(c.ledger.results),1);self.assertFalse(c.stop.is_set());c.ledger.f.close()
            obj['fight']['agent']='pv sims=2000'
            c=self.fake_controller(Path(d)/'bad',obj);r=c.game('key',['fake'],row,{},'h')
            self.assertEqual(r['status'],'error');self.assertTrue(c.stop.is_set());c.ledger.f.close()

    def test_capped_game_not_relabelled_as_loss(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fake_controller(Path(d)/'ledger',{'status':'capped'})
            r=c.game('k',['fake'],{'fight_id':'f','start':{}},{},'h')
            self.assertEqual(r['status'],'capped');self.assertNotIn('line',r);c.ledger.f.close()


if __name__=='__main__':unittest.main()
