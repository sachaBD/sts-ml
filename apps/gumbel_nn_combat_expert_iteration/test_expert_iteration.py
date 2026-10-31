"""The recipe on a prepared run, with nn_play, replay and ONNX export faked: what is played, saved, logged and refused."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from agents.combat.nn import learner
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
window = 1
max_train_rows = 0
balanced_replay = false
prune_replay = true
device = "cpu"

[model]
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


class ExpertIterationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        (self.out / 'inputs').mkdir()
        (self.out / 'config.toml').write_text(CONFIG)
        (self.out / 'inputs/starts.json').write_text(json.dumps([dict(fight_id='deck', start={})]))
        (self.out / 'inputs/test.jsonl').write_text(json.dumps(dict(fight_id='t', start={})) + '\n')
        self.plays = []
        for target, fake in [(expert_iteration, 'play'), (expert_iteration, 'Replay'), (learner, 'train'),
                             (learner, 'export'), (learner, 'export_static')]:
            patcher = patch.object(target, fake, side_effect=getattr(self, f'fake_{fake.lower()}', None))
            setattr(self, fake.lower(), patcher.start())
            self.addCleanup(patcher.stop)

    def fake_play(self, binary, models, games, directory, search, workers, progress):
        self.plays.append(dict(models=models.name, games=len(games), directory=directory, search=search))
        directory.mkdir(parents=True)
        (directory / 'samples-0.bin').touch()
        return [dict(won=True, capped=False, finished_unix=1.0) for _ in games]

    def fake_replay(self, *args, **kwargs):
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
        training.mkdir(exist_ok=resume_from is not None)
        control = RunControl(training)
        with control.lock():
            expert_iteration.ExpertIteration(self.out, load((self.out / 'config.toml').read_text()), control).run(resume_from)
        return control

    def test_fresh_run_plays_trains_exports_and_evaluates(self):
        control = self.run_loop()
        self.assertEqual([(p['models'], p['directory'].name) for p in self.plays],
                         [('iter000', 'iter000'), ('iter000', 'iter001'), ('iter001', 'iter002')])
        evaluation, self_play = self.plays[0]['search'], self.plays[1]['search']
        self.assertEqual((evaluation.sims, evaluation.gumbel_scale), (0, 0.0))
        self.assertEqual((self_play.sims, self_play.gumbel_scale), (4, 1.0))
        self.assertEqual(self.plays[1]['games'], 2)
        self.assertEqual([c.args[2] for c in self.export_static.call_args_list], [512, 256, 32] * 3)
        rows = control.history()
        self.assertEqual([r.get('iteration', r.get('test')) for r in rows], [0, 0, 1, 2])
        self.assertEqual(rows[-1]['states'], 7)
        self.assertEqual(json.loads((control.directory / 'status.json').read_text())['phase'], 'done')
        self.assertFalse((control.iteration(1) / 'samples-0.bin').exists())  # window 1 prunes older samples
        self.assertTrue((control.iteration(2) / 'samples-0.bin').exists())
        self.assertIn('rng', torch.load(control.iteration(2) / 'model.pt', weights_only=True))

    def test_resume_continues_from_the_checkpoint(self):
        self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        self.plays.clear()
        control = self.run_loop(resume_from=1, config=CONFIG)  # iterations may change on resume
        self.assertEqual([p['directory'].name for p in self.plays], ['iter002'])
        self.assertEqual(control.history()[-1]['iteration'], 2)

    def test_resume_refuses_a_different_experiment(self):
        self.run_loop()
        different = CONFIG.replace('sims = 4', 'sims = 8').replace('iterations = 2', 'iterations = 3')
        with self.assertRaisesRegex(ValueError, 'search.sims'):
            self.run_loop(resume_from=2, config=different)

    def test_resume_refuses_a_partial_iteration(self):
        self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        (self.out / 'training/iter002').mkdir()
        with self.assertRaises(FileExistsError):
            self.run_loop(resume_from=1)

    def test_stop_ends_at_an_iteration_boundary(self):
        (self.out / 'training').mkdir()
        (self.out / 'training/STOP').touch()
        control = RunControl(self.out / 'training')
        with control.lock():
            expert_iteration.ExpertIteration(self.out, load((self.out / 'config.toml').read_text()), control).run()
        self.assertEqual([p['directory'].name for p in self.plays], ['iter000'])
        self.assertEqual(json.loads((control.directory / 'status.json').read_text())['phase'], 'stopped')

    def test_initial_checkpoint_is_where_training_starts(self):
        initial = learner.Learner(dict(dim=8, layers=1, heads=1), 1e-3, 'cpu', seed=1)
        initial.save(self.out / 'inputs/initial-model.pt')
        control = self.run_loop(config=CONFIG.replace('iterations = 2', 'iterations = 1'))
        self.assertIn('initial_checkpoint', control.history()[0])
        saved = torch.load(control.iteration(0) / 'model.pt', weights_only=True)['net']
        for name, value in initial.net.state_dict().items():
            torch.testing.assert_close(saved[name], value)

if __name__ == '__main__':
    unittest.main()
