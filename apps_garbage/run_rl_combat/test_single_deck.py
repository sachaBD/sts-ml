"""Fresh fixed-deck curriculum and bounded probability-head tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from agents.combat.pv.model import PolicyValue, NAMES, WIDTHS
from apps.run_rl.single_deck import generate, teacher_count, wilson, score, select_loadout, training_point, plot_training
from apps.run_rl.single_deck_status import stages, journal_counts, training_progress

ROOT=Path(__file__).resolve().parents[2]


class SingleDeck(unittest.TestCase):
    def test_sigmoid_is_bounded_and_legacy_default_unchanged(self):
        inputs=[torch.zeros(2,width) if name=='context' else torch.zeros(2,2,width) for name,width in zip(NAMES,WIDTHS)]
        inputs[-1][...,260]=1
        old=PolicyValue(8);legacy=PolicyValue(8,'softplus');legacy.load_state_dict(old.state_dict())
        self.assertTrue(torch.equal(old(*inputs)[0],legacy(*inputs)[0]))
        fresh=PolicyValue(8,'sigmoid')
        for bias in [-100.,0.,100.]:
            with torch.no_grad():fresh.value.weight.zero_();fresh.value.bias.fill_(bias)
            values,_=fresh(*inputs)
            self.assertTrue(((values>=0)&(values<=100)).all())
        with self.assertRaises(ValueError):PolicyValue(8,'unknown')

    def test_seed_pools_disjoint_and_loadout_unchanged(self):
        start=json.loads((ROOT/'agents/combat/pv/testdata/champ.json').read_text())['start']
        base=dict(fight_id='base',deck_id='d',seed_kind='human',start=start)
        before=copy.deepcopy(base);used={start['seed']}
        a=generate(base,'n','train',20,used);b=generate(base,'n','monitor',20,used)
        self.assertEqual(len(used),41)
        self.assertEqual(base,before)
        self.assertFalse({r['start']['seed'] for r in a}&{r['start']['seed'] for r in b})
        for row in a+b:
            for key in start.keys()-{'seed','misc_rng','potion_rng'}:
                self.assertEqual(row['start'][key],start[key])

    def test_alternative_loadout_preserves_original_hp(self):
        base=dict(deck_id='other',seed_kind='human',start=dict(hp=34,max_hp=52,potions=[1,1]))
        self.assertIs(select_loadout([base],'other'),base)
        bad=copy.deepcopy(base);bad['start']['potions']=[2,1]
        with self.assertRaises(ValueError):select_loadout([bad],'other')

    def test_training_curve_counts_teacher_and_learner_replay(self):
        teacher={'t':dict(status='completed',fight={'won':True})}
        batch={'a':dict(status='completed',fight={'won':True}),
               'b':dict(status='completed',fight={'won':False}),
               'error':dict(status='error')}
        point=training_point(1,batch,['t','a','b'],teacher)
        self.assertEqual(point['batch']['n'],2)
        self.assertEqual(point['batch']['win_rate'],.5)
        self.assertEqual(point['replay']['n'],3)
        self.assertAlmostEqual(point['replay']['win_rate'],2/3)
        with tempfile.TemporaryDirectory() as tmp:
            plot_training(Path(tmp),[point],'Test deck')
            self.assertTrue((Path(tmp)/'training-win-rate.png').is_file())
            self.assertEqual(json.loads((Path(tmp)/'training-win-rate.json').read_text()),[point])

    def test_teacher_tapers_to_zero(self):
        self.assertEqual([teacher_count(i) for i in [1,5,10,11,20]],[200,120,20,0,0])
        lo,hi=wilson(18,20)
        self.assertAlmostEqual(lo,.6989663548)
        self.assertAlmostEqual(hi,.9721335188)

    def test_read_only_dashboard_handles_partial_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'results.jsonl').write_text(json.dumps(dict(status='completed',fight={'won':True}))+'\n{"status":')
            counts=journal_counts(out)
            self.assertEqual(counts['logged'],1)
            self.assertEqual(counts['wins'],1)
            config=dict(bootstrap=200,monitor=100,updates=5,batch_fights=100,eval_every=5)
            self.assertEqual(len(stages(out,config)),22)
            self.assertEqual(training_progress(out,10),'0/10 epochs logged')
            self.assertEqual(len((out/'results.jsonl').read_text().splitlines()),2)

    def test_same_deck_seeds_are_independent_comparison_units(self):
        rows=[dict(fight_id=str(i),deck_id='same') for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,won in [('a',[True,True,False,False]),('b',[True,True,True,False])]:
                p=root/name;p.mkdir()
                (p/'results.jsonl').write_text(''.join(json.dumps(dict(fight_id=str(i),status='completed',fight={'won':w}))+'\n' for i,w in enumerate(won)))
                (p/'summary.json').write_text(json.dumps({'statuses':{'completed':4}}))
            result=score(rows,root/'a',root/'b')
            self.assertEqual(result['n'],4)
            self.assertEqual(result['gap'],.25)
            self.assertEqual(result['gap_se'],.25)


if __name__=='__main__':unittest.main()
