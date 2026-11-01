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
from .run_control import RunControl

CONFIG = """
[run]
id = "test"
worker = "nn_play"
seed_namespace = "test"
games_per_deck = 2
iterations = 2

[search]
sims = 4
gumbel_scale = 1.0
rollout_mix = 0.5
particles = 2

[training]
fights = "unused"
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

[evaluation]
sims = 0
at = [0]
every = 0
wall_share = 0
fights = "unused"
"""


class ExpertIterationInvariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        (self.out / 'inputs').mkdir()
        (self.out / 'config.toml').write_text(CONFIG)
        (self.out / 'inputs/starts.json').write_text(json.dumps([dict(fight_id='deck', start={})]))
        (self.out / 'inputs/test.jsonl').write_text(json.dumps(dict(fight_id='t', start={})) + '\n')
        self.plays = []
        fakes = [(expert_iteration, 'play', self.play), (expert_iteration, 'Replay', self.replay),
                 (learner, 'train', lambda *a, **k: None),
                 (expert_iteration.ModelExporter, 'export', self.export)]
        for target, name, fake in fakes:
            patcher = patch.object(target, name, side_effect=fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def export(self, net, directory, batch, bank):
        files = model_files(directory, bank)
        for path in (files['dynamic'], files['gpu'], *files['bank']):
            path.touch()

    def play(self, binary, models, games, directory, search, workers, progress):
        self.plays.append(dict(models=models, directory=directory, search=search))
        directory.mkdir(parents=True)
        (directory / 'samples-0.bin').touch()
        return [dict(won=True, capped=False, finished_unix=1.0) for _ in games]

    def replay(self, *args, **kwargs):
        replay = MagicMock()
        replay.__enter__.return_value = replay
        replay.__len__.return_value = 7
        replay.groups = {'deck': []}
        replay.available = replay.selected = replay.drawn = {'deck': 1}
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
        return [p for p in self.plays if p['directory'].parent.name != 'test']

    def evaluations(self):
        return [p for p in self.plays if p['directory'].parent.name == 'test']

    def test_each_iteration_plays_the_previous_network(self):
        control = self.run_loop()
        for p in self.self_play():
            it = int(p['directory'].name[4:])
            self.assertEqual(p['models'], control.iteration(it - 1))

    def test_evaluation_never_explores(self):
        self.run_loop(config=CONFIG.replace('every = 0', 'every = 1'))
        self.assertEqual(len(self.evaluations()), 3)
        for p in self.evaluations():
            self.assertEqual((p['search'].gumbel_scale, p['search'].sims), (0.0, 0))
        for p in self.self_play():
            self.assertEqual((p['search'].gumbel_scale, p['search'].sims), (1.0, 4))

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

    def test_entity_runs_through_the_same_recipe(self):
        config = CONFIG.replace('kind = "transformer"', 'kind = "entity"\ngpu_attended = 8')
        control = self.run_loop(config=config)
        checkpoint = torch.load(control.iteration(2) / 'model.pt', weights_only=True)
        self.assertEqual(checkpoint['spec']['kind'], 'entity')
        self.assertEqual([r['iteration'] for r in control.history() if 'iteration' in r], [0, 1, 2])

    def test_initial_checkpoint_is_iteration_zero(self):
        initial = learner.Learner(ModelSpec('transformer', 8, 1, 1), 1e-3, 'cpu', seed=1)
        initial.save(self.out / 'inputs/initial-model.pt')
        control = self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        saved = torch.load(control.iteration(0) / 'model.pt', weights_only=True)['net']
        for name, value in initial.net.state_dict().items():
            torch.testing.assert_close(saved[name], value)


if __name__ == '__main__':
    unittest.main()
