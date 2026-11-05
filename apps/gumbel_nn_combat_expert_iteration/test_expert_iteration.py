"""Invariants of the recipe on a prepared run. nn_play, replay and ONNX export are faked; checkpoints are real."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from agents.combat.nn import learner
from agents.combat.nn.model import ModelSpec
from agents.combat.nn.play import model_files
from . import expert_iteration
from .config import load
from .reporting import deck_of
from .run_control import RunControl

EVALUATION = """
[[evaluation.sets]]
name = "validation"
gumbel_scale = 0.0
rollout_mix = 0.0
particles = 3
sims = 0
at = [0]
every = 0
fights = "unused"
"""

CONFIG = """
[run]
id = "test"
worker = "nn_play"
seed_namespace = "test"
iterations = 2

[starts]
fights = "unused"
per_iteration = 5

[search]
sims = 4
gumbel_scale = 1.0
rollout_mix = 0.5
particles = 2

[training]
steps = 1
batch = 4
lr = 0.001
replay_window = 1
max_train_rows = 0
balanced_replay = false
prune_replay = true
device = "cpu"

[model]
kind = "transformer"
dim = 8
layers = 1
heads = 1
gpu_tokens = 16
export_reuse = false

[workers]
actors = 2
slots = 1
gpu_batch = 512
gpu_bank = [256, 32]
gpu_wait_us = 100
timeout_ms = 321
shutdown_timeout_ms = 123
""" + EVALUATION


class ExpertIterationInvariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        (self.out / 'inputs').mkdir()
        (self.out / 'config.toml').write_text(CONFIG)
        self.decks = [dict(fight_id=name, start=dict(hp=40, max_hp=80)) for name in 'abc']
        (self.out / 'inputs/starts.json').write_text(json.dumps(self.decks))
        (self.out / 'inputs/evaluation').mkdir()
        (self.out / 'inputs/evaluation/validation.jsonl').write_text(json.dumps(dict(fight_id='t', start={})) + '\n')
        self.plays = []
        fakes = [(expert_iteration, 'play', self.play), (expert_iteration, 'Replay', self.replay),
                 (learner.Learner, 'fit', lambda *a, **k: (.1, .2, .1, 0., 0.)),
                 (learner.Learner, 'score', lambda *a: (.3, .4, .3, 0., 0.)),
                 (expert_iteration.ModelExporter, 'export', self.export)]
        for target, name, fake in fakes:
            patcher = patch.object(target, name, side_effect=fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def export(self, net, directory, batch, bank):
        files = model_files(directory, bank)
        for path in (files['dynamic'], files['gpu'], *files['bank']):
            path.touch()

    def play(self, binary, models, fights, directory, search, workers, progress,
             gpu_attended=0, gpu_wide_attended=0, target_association=False, encoding='legacy'):
        results = [dict(fight_id=f['fight_id'], won=i % 2 == 0, capped=False, finished_unix=1.0)
                   for i, f in enumerate(fights)]
        self.plays.append(dict(models=models, directory=directory, search=search, fights=fights, results=results,
                               encoding=encoding))
        directory.mkdir(parents=True)
        (directory / 'samples-0.bin').touch()
        return results

    def replay(self, *args, **kwargs):
        replay = MagicMock()
        replay.__enter__.return_value = replay
        replay.__len__.return_value = 7
        replay.groups = {'a': []}
        replay.available = replay.selected = replay.drawn = {'a': 1}
        return replay

    def run_loop(self, resume_from=None, config=None):
        """As main(): the run reads OUT/config.toml."""
        if config:
            (self.out / 'config.toml').write_text(config)
        training = self.out / 'training'
        training.mkdir(exist_ok=True)
        control = RunControl(training)
        with control.lock():
            expert_iteration.ExpertIteration(self.out, load((self.out / 'config.toml').read_text()), control).run(resume_from)
        return control

    def self_play(self):
        return [p for p in self.plays if 'test' not in p['directory'].relative_to(self.out / 'training').parts]

    def evaluations(self):
        return [p for p in self.plays if 'test' in p['directory'].relative_to(self.out / 'training').parts]

    def test_model_kind_selects_the_encoder_for_training_and_evaluation(self):
        self.run_loop(config=CONFIG.replace('kind = "transformer"', 'kind = "inventory_v1"'))
        self.assertTrue(self.self_play())
        self.assertTrue(self.evaluations())
        self.assertTrue(all(p['encoding'] == 'inventory_v1' for p in self.plays))

    def test_each_iteration_plays_the_previous_network(self):
        control = self.run_loop()
        for p in self.self_play():
            it = int(p['directory'].name[4:])
            self.assertEqual(p['models'], control.iteration(it - 1))

    def test_deck_counts_are_every_self_play_result_so_far(self):
        self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        control = self.run_loop(resume_from=1, config=CONFIG)
        played = {p['directory']: p['results'] for p in self.self_play()}
        expected = {d['fight_id']: dict(fights=0, wins=0) for d in self.decks}
        for it in range(3):
            for r in played.get(control.iteration(it), []):
                expected[deck_of(r['fight_id'])]['fights'] += 1
                expected[deck_of(r['fight_id'])]['wins'] += r['won']
            self.assertEqual(json.loads((control.iteration(it) / 'decks.json').read_text()), expected)

    def test_adaptive_sampling_survives_resume_without_changing_evaluation(self):
        config = CONFIG.replace('per_iteration = 5',
            'per_iteration = 5\nhistory_iterations = 1\nsampling_concentration = 0.5')
        self.run_loop(config=config.replace('iterations = 2', 'iterations = 1'))
        control = self.run_loop(resume_from=1, config=config)
        for p in self.self_play():
            it = int(p['directory'].name[4:])
            artifact = json.loads((control.directory / 'sampling' / f'iter{it:03d}.json').read_text())
            counts = artifact['decks']
            starts = load(config).starts
            expected = starts.draw(self.decks, f'test:{it}', starts.per_iteration,
                                   weights=[counts[d['fight_id']]['probability'] for d in self.decks])
            self.assertEqual(p['fights'], expected)
            self.assertTrue(all(0 <= c['wins'] <= c['attempts'] for c in counts.values()))
            if it == 1:
                self.assertTrue(all(c['attempts'] == 0 for c in counts.values()))
            else:
                self.assertEqual(sum(c['attempts'] for c in counts.values()), starts.per_iteration)
        for p in self.evaluations():
            self.assertEqual(p['fights'], [dict(fight_id='t', start={})])

    def test_evaluation_keeps_its_mix_while_self_play_decays(self):
        config = (CONFIG.replace('every = 0', 'every = 1')
                  .replace('rollout_mix = 0.5', 'rollout_mix = 0.5\nrollout_mix_decay = 2'))
        self.run_loop(config=config)
        self.assertEqual({p['search'].rollout_mix for p in self.evaluations()}, {0})
        mixes = [p['search'].rollout_mix for p in self.self_play()]
        self.assertEqual(mixes, sorted(mixes, reverse=True))
        self.assertTrue(mixes[0] > 0 and mixes[-1] == 0)

    def test_sets_keep_their_search_schedule_and_results_separate(self):
        (self.out / 'inputs/evaluation/retention.jsonl').write_bytes(
            (self.out / 'inputs/evaluation/validation.jsonl').read_bytes())
        retention = (EVALUATION.replace('name = "validation"', 'name = "retention"')
                     .replace('sims = 0', 'sims = 12').replace('at = [0]', 'at = [2]'))
        config = CONFIG.replace('every = 0', 'every = 1') + retention
        control = self.run_loop(config=config)
        evaluations = self.evaluations()
        self.assertEqual([(p['directory'].parent.name, p['directory'].name) for p in evaluations],
                         [('iter000', 'validation'), ('iter001', 'validation'),
                          ('iter002', 'validation'), ('iter002', 'retention')])
        self.assertEqual([p['search'].sims for p in evaluations], [0, 0, 0, 12])
        self.assertEqual([(r['test'], r['set']) for r in control.history() if 'test' in r],
                         [(0, 'validation'), (1, 'validation'), (2, 'validation'), (2, 'retention')])

    def test_recovery_retries_only_the_uncommitted_set(self):
        (self.out / 'inputs/evaluation/retention.jsonl').write_bytes(
            (self.out / 'inputs/evaluation/validation.jsonl').read_bytes())
        config = CONFIG + EVALUATION.replace('name = "validation"', 'name = "retention"')

        def interrupted(*args, **kwargs):
            directory = args[3]
            if directory.name == 'retention':
                directory.mkdir(parents=True)
                (directory / 'worker.log').write_text('interrupted')
                raise RuntimeError('interrupted evaluation')
            return self.play(*args, **kwargs)

        with patch.object(expert_iteration, 'play', side_effect=interrupted), \
                self.assertRaisesRegex(RuntimeError, 'interrupted evaluation'):
            self.run_loop(config=config)
        control = self.run_loop(resume_from=0, config=config)
        archived = list((control.directory / 'test/iter000').glob('.retention-interrupted-*/worker.log'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), 'interrupted')
        self.assertEqual([(r['test'], r['set']) for r in control.history() if 'test' in r],
                         [(0, 'validation'), (0, 'retention')])

    def test_a_logged_iteration_has_its_checkpoint_and_models(self):
        control = self.run_loop()
        logged = [r['iteration'] for r in control.history() if 'iteration' in r]
        self.assertEqual(logged, [0, 1, 2])
        for it in logged:
            for name in ('model.pt', 'model.onnx', 'model-gpu.onnx', 'model-gpu-b256.onnx', 'model-gpu-b32.onnx'):
                self.assertTrue((control.iteration(it) / name).exists(), f'iter{it}/{name}')

    def test_samples_outside_the_replay_window_are_pruned(self):
        control = self.run_loop()
        self.assertEqual([k for k in range(3) if (control.iteration(k) / 'samples-0.bin').exists()], [2])
        self.assertFalse(list((control.directory / 'test').rglob('samples-*.bin')))

    def test_resume_continues_the_same_sequence(self):
        self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        control = self.run_loop(resume_from=1, config=CONFIG)
        self.assertEqual([r['iteration'] for r in control.history() if 'iteration' in r], [0, 1, 2])

    def test_resume_never_changes_the_experiment(self):
        self.run_loop()
        different = CONFIG.replace('sims = 4', 'sims = 8').replace('iterations = 2', 'iterations = 3')
        with self.assertRaisesRegex(ValueError, r'\[search\]\.sims'):
            self.run_loop(resume_from=2, config=different)

    def test_resume_never_appends_to_a_partial_iteration(self):
        self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        (self.out / 'training/iter002').mkdir()
        with self.assertRaises(FileExistsError):
            self.run_loop(resume_from=1)

    def test_stop_leaves_only_complete_iterations(self):
        (self.out / 'training').mkdir()
        (self.out / 'training/STOP').touch()
        control = self.run_loop()
        self.assertEqual(sorted(p.name for p in control.directory.glob('iter*')), ['iter000'])
        self.assertEqual(json.loads((control.directory / 'status.json').read_text())['phase'], 'stopped')

    def test_initial_checkpoint_is_iteration_zero(self):
        initial = learner.Learner(ModelSpec('transformer', 8, 1, 1), 1e-3, 'cpu', seed=1)
        initial.save(self.out / 'inputs/initial-model.pt')
        control = self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        saved = torch.load(control.iteration(0) / 'model.pt', weights_only=True)['net']
        for name, value in initial.net.state_dict().items():
            torch.testing.assert_close(saved[name], value)


if __name__ == '__main__':
    unittest.main()
