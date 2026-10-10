"""Controller unit tests (no gameplay). The end-to-end smoke is opt-in: STSRL_CORPUS_SMOKE=<decks.json> plus a worker."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.corpus.controller import (DEFAULTS, Controller, allocate, fight_seconds, macro, next_status, smoothed_p,
                                    teacher_count, training_ids)

ROOT = Path(__file__).resolve().parents[2]
START = json.loads((ROOT / 'agents/combat/pv/testdata/champ.json').read_text())['start']


def controller(excluded=()):
    c = object.__new__(Controller)
    c.ns, c.excluded = 'ns', set(excluded)
    c.decks = {d: dict(start=dict(fight_id='b', deck_id=d, seed_kind='human', start=START)) for d in ('A', 'B')}
    return c


def deck(entry=0, learner=(), teacher=50):
    batches = [dict(k=k, kind='learner', attempted=len(ids), completed=len(ids), wins=len(ids) // 2, seconds=2. * len(ids),
                    ids=ids, shard=f's{k}') for k, ids in learner]
    return dict(entry_update=entry, batches=batches, screen_seconds=50., teacher_pool={f't{i}': 'T' for i in range(teacher)})


class Seeds(unittest.TestCase):
    def test_deterministic_and_independent_of_other_decks_and_counts(self):
        c = controller()
        a1 = c.rows('A', 'learner-3', 5)
        c.rows('B', 'learner-3', 9); c.rows('A', 'monitor', 30)  # no shared mutable state
        a2 = c.rows('A', 'learner-3', 8)
        self.assertEqual([r['start']['seed'] for r in a1], [r['start']['seed'] for r in a2[:5]])
        self.assertEqual([r['fight_id'] for r in a1], [f'ns:A:learner-3:{i}' for i in range(5)])
        want = int.from_bytes(hashlib.sha256(b'ns:A:learner-3:0').digest()[:8], 'little')
        self.assertEqual(a1[0]['start']['seed'], want)
        self.assertFalse({r['start']['seed'] for r in a1} & {r['start']['seed'] for r in c.rows('B', 'learner-3', 5)})

    def test_static_excluded_seeds_skipped_without_mutation(self):
        want = int.from_bytes(hashlib.sha256(b'ns:A:final:0').digest()[:8], 'little')
        c = controller({want})
        seeds = c.excluded.copy()
        self.assertNotEqual(c.rows('A', 'final', 3)[0]['start']['seed'], want)
        self.assertEqual(c.excluded, seeds)


class Quotas(unittest.TestCase):
    def test_floor_budget_graduated(self):
        q = allocate(100, 10, [('a', False, 3.), ('b', False, 1.), ('c', True, 99.), ('d', False, 0.)])
        self.assertEqual(sum(q.values()), 100)
        self.assertEqual((q['c'], q['d']), (10, 10))  # graduated exactly the floor; zero weight keeps the floor
        self.assertGreater(q['a'], q['b'])
        self.assertEqual(allocate(10, 10, [('a', False, 1.), ('b', False, 1.)]), {'a': 10, 'b': 10})
        self.assertEqual(allocate(7, 0, [('a', False, 1.), ('b', False, 1.), ('c', False, 1.)]), {'a': 3, 'b': 2, 'c': 2})

    def test_weights_inputs(self):
        batches = [dict(wins=10, completed=10), dict(wins=0, completed=10), dict(wins=5, completed=10)]
        self.assertAlmostEqual(smoothed_p(batches), (5 + 1) / (20 + 2))  # last two batches only
        d = deck(learner=[(1, ['x', 'y'])])
        self.assertAlmostEqual(fight_seconds(d), 2.)
        self.assertAlmostEqual(fight_seconds(deck()), 10.)  # screen seconds / 5


class Training(unittest.TestCase):
    def test_teacher_taper_uses_per_deck_age(self):
        self.assertEqual([teacher_count(a, 50) for a in (0, 1, 9, 10, 11)], [50, 45, 5, 0, 0])
        ctl = dict(DEFAULTS)
        new, old = deck(entry=20), deck(entry=2)
        self.assertEqual(len(training_ids(new, 20, ctl)), 50)   # just entered at a late global update: full teacher
        self.assertEqual(len(training_ids(old, 20, ctl)), 0)    # age 18: tapered away
        self.assertEqual(len(training_ids(new, 23, ctl)), 35)
        self.assertEqual(set(training_ids(new, 23, ctl)), set(training_ids(new, 23, ctl)))  # deterministic

    def test_replay_window_is_per_deck(self):
        ctl = dict(DEFAULTS, replay_fights_per_deck=5, teacher_per_deck=0)
        d = deck(entry=0, learner=[(1, ['a1', 'a2', 'a3']), (2, ['b1', 'b2', 'b3']), (30, ['c1'])], teacher=0)
        self.assertEqual(training_ids(d, 30, ctl), {'a3': 's1', 'b1': 's2', 'b2': 's2', 'b3': 's2', 'c1': 's30'})
        sparse = deck(entry=0, learner=[(1, ['z1', 'z2'])], teacher=0)  # old data of a slow deck survives global age
        self.assertEqual(set(training_ids(sparse, 99, ctl)), {'z1', 'z2'})


class Lifecycle(unittest.TestCase):
    def ev(self, wins, mcts, n=30):
        return dict(n=n, wins=wins, mcts_wins=mcts)

    def test_transitions(self):
        self.assertEqual(next_status('active', self.ev(28, 27), .5, 10, 27)[0], 'graduated')
        self.assertEqual(next_status('active', self.ev(26, 20), .5, 10, 27)[0], 'active')   # below 27
        self.assertEqual(next_status('active', self.ev(28, 29), .5, 10, 27)[0], 'active')   # below MCTS
        self.assertEqual(next_status('graduated', self.ev(28, 29), .5, 10, 27)[0], 'active')
        self.assertEqual(next_status('graduated', None, .5, 10, 27)[0], 'graduated')
        self.assertEqual(next_status('active', None, .04, 150, 27)[0], 'parked')
        self.assertEqual(next_status('active', None, .04, 149, 27)[0], 'active')
        self.assertEqual(next_status('parked', self.ev(30, 30), .04, 200, 27)[0], 'graduated')  # reversible
        self.assertEqual(next_status('software-failed', self.ev(30, 0), .5, 0, 27)[0], 'software-failed')

    def test_macro_cluster_se(self):
        decks = [dict(evals=[dict(n=30, wins=w, mcts_wins=m)]) for w, m in ((30, 27), (15, 21), (24, 24))]
        agg = macro(decks)
        self.assertAlmostEqual(agg['gap'], (3 - 6 + 0) / 90)
        self.assertIsNotNone(agg['gap_se'])


class EncodeFailure(unittest.TestCase):
    def test_entry_marks_deck_failed_and_continues(self):
        import pyarrow as pa
        import apps.corpus.controller as ctl_mod
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c = controller()
            c.out, c.worker, c.state = out, out / 'w', dict(decks={}, model=None)
            c.a = type('A', (), dict(monitor=1, final=1, teacher_sims=1, workers=1))()
            c.order = ['A', 'B']
            c.save = lambda *a, **k: None
            c.log = lambda m: None
            for d in 'AB':
                c.decks[d].update(teacher_runs=['r'], teacher_fight_ids=[f'ns:{d}:screen:0'])
            (out / 'logs').mkdir()
            (out / 'logs/shard-screen-A.log').write_text('x\nunsupported card RAGNAROK\n')
            ref = out / 'ref'
            res = {f'ns:{d}:screen:0': dict(status='completed', seconds=3.) for d in 'AB'}
            res.update({f'ns:{d}:monitor:0': dict(status='completed', seconds=1.) for d in 'AB'})

            def fake_encode(o, name, src, worker):
                if name == 'shard-screen-A':
                    raise ctl_mod.subprocess.CalledProcessError(1, 'enc')
                return o / name / 'rows.parquet'
            with patch.object(ctl_mod, 'play', lambda *a, **k: ref), patch.object(ctl_mod, 'write_starts', lambda *a: None), \
                    patch.object(ctl_mod, 'read_results', lambda p: res), patch.object(ctl_mod, 'subset', lambda *a: None), \
                    patch.object(ctl_mod, 'encode', fake_encode):
                c.entry(0, ['A', 'B'], dict(DEFAULTS, teacher_per_deck=1))
            a, b = c.state['decks']['A'], c.state['decks']['B']
            self.assertEqual((a['status'], b['status']), ('software-failed', 'active'))
            self.assertIn('RAGNAROK', a['reasons'][0])
            manifest = json.loads((out / 'split.json').read_text())
            self.assertFalse([f for f in manifest if ':A:' in f])
            self.assertTrue([f for f in manifest if ':B:' in f])
            self.assertEqual(c.entered(), ['B'])


@unittest.skipUnless(os.environ.get('STSRL_CORPUS_SMOKE'), 'opt-in gameplay smoke (needs <=2 idle workers)')
class Smoke(unittest.TestCase):
    def test_two_decks_two_updates(self):
        from apps.run_rl.combat_loop import active_play_workers
        self.assertEqual(active_play_workers(), 0)
        out = Path(os.environ.get('STSRL_CORPUS_SMOKE_OUT', 'scratch/corpus-smoke'))
        out.mkdir(parents=True, exist_ok=True)
        (out / 'control.json').write_text(json.dumps(dict(
            DEFAULTS, fights_per_update=4, min_fights_per_deck=2, wave_every=1, wave_size=1, eval_every=1,
            teacher_per_deck=4, epochs=1)))
        cmd = [sys.executable, '-m', 'apps.corpus.controller', 'run', '--decks', os.environ['STSRL_CORPUS_SMOKE'],
               '--worker', os.environ['STSRL_CORPUS_WORKER'], '--out', str(out), '--workers', '2', '--device', 'cpu',
               '--sims', '16', '--teacher-sims', '64', '--monitor', '2', '--final', '2', '--updates', '2']
        subprocess.run(cmd, check=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
        state = json.loads((out / 'state.json').read_text())
        self.assertEqual(state['k'], 2)
        self.assertEqual(len(state['decks']), 2)


if __name__ == '__main__':
    unittest.main()
