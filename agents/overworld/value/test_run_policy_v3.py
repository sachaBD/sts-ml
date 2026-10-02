"""Graph topology invariants, attention masking, checkpoint parity and training integration."""
import copy
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest

import torch

from .core import build_model, encode, load_model, run_score, save_model
from .graph_encoding import NUMERIC
from .learn import batches, to
from .run_policy_v3 import RunPolicyV3, RunPolicyV3Attention
from .policy import Policy


def mkcard(id,upgraded=0):
    return dict(card_id=id,upgraded=upgraded,misc=0,mechanics=dict(type=0,rarity=3,cost=1,base_damage=6,innate=False,ethereal=False,exhaust=False,self_retain=False,x_cost=False))


def state(act=1):
    card = mkcard(25)
    o = dict(version=1,character=0,ascension=20,act=act,floor_in_act=1,
             keys=dict(ruby=False,sapphire=False,emerald=False),card_rarity_factor=5,
             card_rarity={k:dict(common=.6,uncommon=.37,rare=.03) for k in ('hallway','elite','shop')},
             potion_chance_modifier=0,potion_roll_probability=.4,potion_obtain_probability=.4,
             question_base=dict(fight=.1,shop=.03,treasure=.02),
             question_next=dict(fight=.1,shop=.03,treasure=.02,event=.85),
             question_after_shop=dict(fight=.1,shop=0.,treasure=.02,event=.88),
             current_room=4,visible_encounter=0,shop_remove_count=0,shop_remove_cost=75,
             events=dict(normal=[dict(id=6,eligible=True)],shrine=[],one_time=[]),
             relic_candidates=dict(common=[dict(id=20,eligible=True,shop_eligible=True)]),relic_candidates_exact=False,
             encounters=dict(possible_elites=[15,16,17],last_elite=0,hallway_count=1,elite_count=0,
                             hallway_history=[1],next_hallway=[dict(id=2,probability=1.)]))
    nodes = [dict(x=0,y=y,room='MONSTER',edges=[0]) for y in range(15)]
    nodes[1]['room']='REST'; nodes[2]['room']='ELITE'; nodes[-1]['room']='REST'; nodes[-1]['edges']=[]
    return dict(deck=[card,mkcard(104,1)],relics=[dict(relic_id=1,data=0)],potions=[],
                boss='hexaghost' if act==1 else 'the_heart',hp=45,max_hp=75,gold=99,floor=1,potion_capacity=2,
                overworld=o,map=dict(nodes=nodes,paths=[],current=dict(y=0,x=0),burning_elite=None))


def batch(s=None, model=None, opts=None):
    return encode([(s or state(),opts or [])],['hexaghost'],kind=model.KIND if model else RunPolicyV3.KIND)


class TopologyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def models(self):
        return [c(width=8,hidden=16,depth=1,heads=2,dropout=0).eval() for c in (RunPolicyV3,RunPolicyV3Attention)]

    def test_round_trip_and_gradient(self):
        for model in self.models():
            with self.subTest(kind=model.KIND), tempfile.TemporaryDirectory() as d:
                b = batch(model=model,opts=[mkcard(11)])
                expected = model(b)[0]
                self.assertEqual(expected.shape,(1,2))
                self.assertTrue(torch.isfinite(expected).all())
                path=Path(d)/'model.pt'; save_model(model,path,target='heart')
                torch.testing.assert_close(load_model(path)(b)[0],expected,rtol=0,atol=0)
                model.train(); model(b)[0].sum().backward()
                for key in ('node_mlp.0.weight','node_update.0.weight','card_mlp.0.weight','relic_mlp.0.weight'):
                    grad=dict(model.named_parameters())[key].grad
                    self.assertIsNotNone(grad,key); self.assertGreater(grad.abs().sum(),0,key)
                if model.attention:
                    self.assertGreater(model.attention[0].self_attn.in_proj_weight.grad.abs().sum(),0)

    def test_order_is_preserved(self):
        s=state(); t=copy.deepcopy(s)
        t['map']['nodes'][1]['room'],t['map']['nodes'][2]['room']=t['map']['nodes'][2]['room'],t['map']['nodes'][1]['room']
        for model in self.models():
            b,c=batch(s,model),batch(t,model)
            self.assertFalse(torch.equal(b['graph_room'],c['graph_room']))
            self.assertGreater(float((model(b)[0]-model(c)[0]).abs().max().detach()),1e-7)

    def test_roots_connectivity_and_burning(self):
        s=state(); s['map']['nodes'] += [dict(x=1,y=1,room='SHOP',edges=[0])]
        s['map']['nodes'][0]['edges']=[0,1]
        a=copy.deepcopy(s); a['map']['next_xs']=[0]
        b=copy.deepcopy(s); b['map']['next_xs']=[1]
        x,y=batch(a),batch(b)
        self.assertEqual(x['graph_roots'].sum(),1)
        self.assertEqual(y['graph_roots'].sum(),1)
        self.assertEqual(x['graph_mask'][0,1,1],0)
        self.assertEqual(y['graph_mask'][0,1,0],0)
        self.assertEqual(x['graph_mask'][0,15,0],1) # explicit boss sink
        burn=copy.deepcopy(a); burn['map']['burning_elite']=dict(x=0,y=2)
        self.assertEqual(batch(burn)['graph_numeric'][0,2,0,2],1)
        for model in self.models():
            self.assertGreater(float((model(batch(a,model))[0]-model(batch(b,model))[0]).abs().max().detach()),1e-7)

    def test_permutation_padding_and_empty(self):
        for model in self.models():
            s=state(); t=copy.deepcopy(s); t['deck'].reverse(); t['map']['nodes'].reverse()
            torch.testing.assert_close(model(batch(s,model))[0],model(batch(t,model))[0],rtol=1e-5,atol=1e-6)
            opts=[mkcard(11)]
            alone=model(batch(s,model,opts))[0]
            large=copy.deepcopy(s); large['deck']*=3; large['relics']*=3
            large['overworld']['events']['normal']*=3
            combined=encode([(s,opts),(large,opts*3)],['hexaghost']*2,kind=model.KIND)
            value=model(combined)[0][0]
            torch.testing.assert_close(alone[0,0],value[0],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(alone[0,-1],value[-1],rtol=1e-5,atol=1e-6)
            empty=state(); empty['deck']=[];empty['relics']=[];empty['map']['current']=dict(y=15,x=0)
            self.assertTrue(torch.isfinite(model(batch(empty,model))[0]).all())

    def test_heart_and_legacy_contract(self):
        s=state(4); s['map']['current']=dict(y=-1,x=-1)
        s['map']['nodes']=[dict(x=3,y=y,room=r,edges=[3] if y<3 else [])
                           for y,r in enumerate(['REST','SHOP','ELITE','BOSS'])]
        b=batch(s)
        self.assertEqual(b['boss'].item(),9)
        self.assertEqual(b['graph_mask'].sum(),4)
        self.assertEqual(b['graph_room'][0,3,3],6)
        self.assertEqual(b['graph_mask'][0,15,0],0)
        for model in self.models(): self.assertTrue(torch.isfinite(model(batch(s,model))[0]).all())
        legacy=state();del legacy['overworld']
        with self.assertRaisesRegex(ValueError,'recollect'): batch(legacy)
        with self.assertRaisesRegex(ValueError,'map.nodes'):
            bad=state();del bad['map']['nodes'];batch(bad)
        old=build_model(dict(kind='run_policy_v1',width=8,hidden=16,dropout=0)).eval()
        plain=encode([(legacy,[])],['hexaghost']);with_extra=encode([(state(),[])],['hexaghost'])
        torch.testing.assert_close(old(plain)[0],old(with_extra)[0],rtol=0,atol=0)
        self.assertEqual(run_score(dict(heart_cleared=True),'heart'),1)
        self.assertEqual(run_score(dict(status='act_complete',floor=51),'heart'),0)

    def test_policy_dispatch(self):
        for model in self.models():
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'model.pt'; save_model(model,path)
                policy=Policy('net',ckpt=path,target='heart')
                s=state()
                choice,source,values=policy(dict(state=s,options=[mkcard(11)],boss='hexaghost'))
                self.assertIn(choice,(0,1));self.assertEqual(source,'net');self.assertEqual(len(values),2)
                choice,source,values=policy.decide(dict(after=[s,s],options=[{},{}],boss='hexaghost',decision='rest',simple=0))
                self.assertIn(choice,(0,1));self.assertEqual(len(values),2)
                self.assertEqual(policy.terminal(dict(terminal='cleared',floor=51)),0)
                self.assertEqual(policy.terminal(dict(terminal='cleared',floor=56,heart_cleared=True)),1)

    def test_native_observation(self):
        root=Path(__file__).resolve().parents[3]
        binary=root/'build'/os.environ.get('STSRL_BUILD_DIR','main')/'overworld_observation_test'
        if not binary.exists():
            self.skipTest('native observation dump executable not built')
        for flag in ('--dump','--dump4'):
            s=json.loads(subprocess.check_output([binary,flag],text=True))
            for model in self.models():
                b=batch(s,model,opts=s['deck'][:2])
                self.assertTrue(torch.isfinite(model(b)[0]).all())
                self.assertEqual(b['act'].item(),s['act'])
                self.assertGreater(b['graph_roots'].sum(),0)

    def test_learner_encoding(self):
        s=state();m=self.models()[0]
        examples=[(s,[],None,'hexaghost',.5,1,1.,[0.,0.,0.])]
        b,col,y,w,aux=batches(examples,1,'cpu',False,kind=m.KIND)[0]
        self.assertEqual(b['run_numeric'].shape[1],len(NUMERIC))
        self.assertTrue(torch.isfinite(m(to(b,'cpu'))[0]).all())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
