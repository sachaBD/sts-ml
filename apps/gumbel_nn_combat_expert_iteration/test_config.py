"""Invariants of the config: a mistake stops the run before anything is written; resume may only change resumable settings."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from .config import experiment_changes, load
from .test_expert_iteration import CONFIG, EVALUATION

PRESETS = Path(__file__).with_name('config')


class ConfigTests(unittest.TestCase):
    def rejects(self, text, message):
        with patch('sys.exit', side_effect=SystemExit) as exit, self.assertRaises(SystemExit):
            load(text)
        self.assertIn(message, exit.call_args.args[0])

    def test_the_presets_are_valid(self):
        for preset in sorted(PRESETS.glob('*.toml')):
            with self.subTest(preset=preset.name):
                load(preset.read_text())

    def test_a_mistake_stops_the_run(self):
        self.rejects(CONFIG.replace('steps = 1', 'stepz = 1'), "unknown [training] keys: ['stepz']")
        self.rejects(CONFIG.replace('lr = 0.001\n', ''), 'missing [training].lr')
        self.rejects(CONFIG.replace('actors = 2', 'actors = "2"'), '[workers].actors must be int')
        self.rejects(CONFIG.replace('batches = [4, 2]', 'batches = [4, 4]'), 'distinct positive batch sizes')
        self.rejects(CONFIG.replace('deployments = [{ tokens = 16, batches = [4, 2] }]',
                                    'deployments = [{ tokens = 16, batches = [4] }, { tokens = 8, batches = [4] }]'),
                     'strictly increasing')
        self.rejects(CONFIG.replace('rollout_mix = 0.5', 'rollout_mix = 2'), 'rollout_mix')
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nhp_jitter = -0.1'), 'hp_jitter')
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nexcluded_cards = ["PURTY"]'), "['PURTY']")
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nexcluded_relics = ["RUNIC_DOOM"]'), "['RUNIC_DOOM']")
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nexcluded_potions = [15]'), 'list of str')
        self.rejects(CONFIG.replace('rollout_mix = 0.5', 'rollout_mix = 0.5\nrollout_mix_decay = -1'), 'rollout_mix_decay')

    def test_sampling_policy_is_validated_and_cannot_change_on_resume(self):
        base = load(CONFIG)
        for key, value in [('history_iterations', 3), ('win_rate_prior_strength', 2.0),
                           ('sampling_concentration', 0.5)]:
            changed = CONFIG.replace('per_iteration = 5', f'per_iteration = 5\n{key} = {value}')
            self.assertEqual(experiment_changes(base, load(changed)), [f'[starts].{key}'])
        for key, values in [('history_iterations', (0, -1)),
                            ('win_rate_prior_strength', (0, -1, 'inf', 'nan')),
                            ('sampling_concentration', (-0.1, 1.1, 'nan'))]:
            for value in values:
                self.rejects(CONFIG.replace('per_iteration = 5', f'per_iteration = 5\n{key} = {value}'), key)
        self.assertEqual(base.starts.sampling_concentration, 0)

    def test_evaluation_resume_preserves_search_but_allows_scheduling_changes(self):
        base = load(CONFIG)
        evaluation = base.evaluation.sets[0]
        for changed, expected in ((replace(evaluation, every=7, at=(0, 7)), []),
                                  (replace(evaluation, particles=6), ['[evaluation].sets'])):
            config = replace(base, evaluation=replace(base.evaluation, sets=(changed,)))
            self.assertEqual(experiment_changes(base, config), expected)

    def test_evaluation_names_cannot_collide_or_escape_the_run(self):
        for name in ('', '.', '..', '../outside', '/tmp/outside'):
            self.rejects(CONFIG.replace('name = "validation"', f'name = "{name}"'), 'set name')
        duplicate = CONFIG + EVALUATION
        self.rejects(duplicate, 'set names must be unique')

    def test_only_resumable_settings_may_change_on_resume(self):
        base = load(CONFIG)
        self.assertEqual(experiment_changes(base, load(CONFIG.replace('iterations = 2', 'iterations = 9'))), [])
        self.assertEqual(experiment_changes(base, load(CONFIG.replace('lr = 0.001', 'lr = 0.01'))), ['[training].lr'])

if __name__ == '__main__':
    unittest.main()
