"""Recipe invariants for both models; only native gameplay and ONNX export are stubbed."""
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from agents.combat.nn._test_support import combat_tokens
from agents.combat.nn.learner import Learner
from agents.combat.nn.play import Search, model_files
from agents.combat.nn.tokens import TokenKind, TokenRows
from . import expert_iteration
from ._test_support import CONFIG, POOLED_CONFIG
from .config import load
from .run_control import RunControl


@dataclass(frozen=True)
class PlayedStage:
    models: Path
    directory: Path
    search: Search



class RecipeContract:
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.inputs = combat_tokens()
        self.prepare_run(self.directory / 'initial')
        self.plays = []
        self.enterContext(patch.object(expert_iteration, 'play', side_effect=self.play))
        self.enterContext(patch.object(expert_iteration.ModelExporter, 'export', side_effect=self.export))

    def prepare_run(self, directory):
        self.out = directory
        (directory / 'inputs').mkdir(parents=True)
        (directory / 'config.toml').write_text(self.config_text)
        (directory / 'inputs/starts.json').write_text(json.dumps([dict(fight_id='deck', start={})]))
        (directory / 'inputs/test.jsonl').write_text(json.dumps(dict(fight_id='test', start={})) + '\n')

    def export(self, net, directory, batch, bank):
        files = model_files(directory, bank)
        for path in (files['dynamic'], files['gpu'], *files['bank']):
            path.touch()

    def play(self, binary, models, games, directory, search, workers, progress):
        self.plays.append(PlayedStage(models, directory, search))
        directory.mkdir(parents=True)
        tokens = self.inputs[0]
        action_count = int(TokenRows(tokens).is_kind(TokenKind.ACTION).sum())
        policy = torch.full((action_count,), 1 / action_count)
        sample = struct.pack('<iif', len(tokens), action_count, 1.) + tokens.numpy().tobytes() + policy.numpy().tobytes()
        (directory / 'samples-0.bin').write_bytes(sample * len(games))
        results = [dict(fight_id=game['fight_id'], won=True, capped=False, decisions=1, finished_unix=1.)
                   for game in games]
        (directory / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results))
        return results

    def run_loop(self, resume_from=None, config=None):
        if config is not None:
            (self.out / 'config.toml').write_text(config)
        settings = load((self.out / 'config.toml').read_text())
        training = self.out / 'training'
        training.mkdir(exist_ok=True)
        control = RunControl(training)
        with control.lock():
            expert_iteration.ExpertIteration(self.out, settings, control).run(resume_from)
        return control

    def collections(self):
        return [stage for stage in self.plays if stage.directory.parent.name != 'test']

    def evaluations(self):
        return [stage for stage in self.plays if stage.directory.parent.name == 'test']

    def test_collection_uses_the_previous_checkpoint(self):
        control = self.run_loop()

        self.assertTrue(self.collections())
        for stage in self.collections():
            iteration = int(stage.directory.name.removeprefix('iter'))
            self.assertEqual(stage.models, control.iteration(iteration - 1))

    def test_evaluation_never_explores(self):
        self.run_loop(config=self.config_text.replace('every = 0', 'every = 1'))

        self.assertTrue(self.evaluations())
        for stage in self.evaluations():
            self.assertEqual(stage.search.gumbel_scale, 0)

    def test_evaluation_uses_its_own_search_budget(self):
        self.run_loop(config=self.config_text.replace('every = 0', 'every = 1'))

        self.assertTrue(self.evaluations())
        for stage in self.evaluations():
            self.assertEqual(stage.search.sims, 0)  # fixture: evaluate without search, collect with four sims

    def test_logged_iterations_have_loadable_checkpoints_and_deployable_files(self):
        control = self.run_loop()
        settings = load(self.config_text)
        restored = Learner(settings.model.architecture(), lr=1e-3, device='cpu')

        logged = [row for row in control.history() if 'iteration' in row]
        self.assertTrue(logged)
        for row in logged:
            directory = control.iteration(row['iteration'])
            restored.load(directory / 'model.pt')
            for name in ('model.onnx', 'model-gpu.onnx', 'model-gpu-b256.onnx', 'model-gpu-b32.onnx'):
                self.assertTrue((directory / name).exists(), name)

    def test_expired_training_samples_are_pruned(self):
        control = self.run_loop()

        self.assertFalse((control.iteration(1) / 'samples-0.bin').exists())
        self.assertTrue((control.iteration(2) / 'samples-0.bin').exists())

    def test_evaluation_samples_are_not_retained_for_training(self):
        control = self.run_loop()

        self.assertTrue(self.evaluations())
        self.assertFalse(list((control.directory / 'test').rglob('samples-*.bin')))

    def test_resume_continues_without_repeating_completed_iterations(self):
        self.run_loop(config=self.config_text.replace('iterations = 2', 'iterations = 1'))

        control = self.run_loop(resume_from=1, config=self.config_text)

        completed = [row['iteration'] for row in control.history() if 'iteration' in row]
        self.assertEqual(completed, [0, 1, 2])

    def checkpoint_predictions(self, control, iteration):
        architecture = load(self.config_text).model.architecture()
        restored = Learner(architecture, lr=1e-3, device='cpu')
        restored.load(control.iteration(iteration) / 'model.pt')
        return restored.net.eval()(self.inputs)

    def test_training_updates_the_collected_model(self):
        control = self.run_loop()
        initial_values, _ = self.checkpoint_predictions(control, 0)
        trained_values, _ = self.checkpoint_predictions(control, 2)

        self.assertFalse(torch.equal(initial_values, trained_values))

    def test_resumed_training_matches_uninterrupted_training(self):
        uninterrupted = self.run_loop()
        expected = self.checkpoint_predictions(uninterrupted, 2)
        self.prepare_run(self.directory / 'resumed')
        self.run_loop(config=self.config_text.replace('iterations = 2', 'iterations = 1'))

        resumed = self.run_loop(resume_from=1, config=self.config_text)

        torch.testing.assert_close(self.checkpoint_predictions(resumed, 2), expected, rtol=0, atol=0)

    def test_resume_rejects_a_different_experiment(self):
        self.run_loop()
        different_search = self.config_text.replace('sims = 4', 'sims = 8')
        different_search = different_search.replace('iterations = 2', 'iterations = 3')

        with self.assertRaises(ValueError):
            self.run_loop(resume_from=2, config=different_search)

    def test_resume_does_not_append_to_partial_output(self):
        self.run_loop(config=self.config_text.replace('iterations = 2', 'iterations = 1'))
        (self.out / 'training/iter002').mkdir()

        with self.assertRaises(FileExistsError):
            self.run_loop(resume_from=1)

    def test_stop_does_not_start_another_iteration(self):
        (self.out / 'training').mkdir()
        (self.out / 'training/STOP').touch()

        control = self.run_loop()

        self.assertTrue((control.iteration(0) / 'model.pt').exists())
        self.assertFalse(control.iteration(1).exists())

    def test_warm_start_is_the_initial_model(self):
        architecture = load(self.config_text).model.architecture()
        initial = Learner(architecture, lr=1e-3, device='cpu', seed=1)
        inputs = combat_tokens()
        expected = initial.net.eval()(inputs)
        initial.save(self.out / 'inputs/initial-model.pt')

        control = self.run_loop()
        restored = Learner(architecture, lr=1e-3, device='cpu')
        restored.load(control.iteration(0) / 'model.pt')

        torch.testing.assert_close(restored.net.eval()(inputs), expected, rtol=0, atol=0)


class TransformerRecipeTests(RecipeContract, unittest.TestCase):
    config_text = CONFIG


class PooledTransformerRecipeTests(RecipeContract, unittest.TestCase):
    config_text = POOLED_CONFIG


if __name__ == '__main__':
    unittest.main()
