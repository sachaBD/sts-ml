"""Invariants of the config: the presets load; resume may only change resumable settings."""
from dataclasses import replace
from pathlib import Path
import unittest

from .config import experiment_changes, load
from .test_expert_iteration import CONFIG

PRESETS = Path(__file__).with_name('config')


class ConfigTests(unittest.TestCase):
    def test_the_presets_are_valid(self):
        for preset in sorted(PRESETS.glob('*.toml')):
            with self.subTest(preset=preset.name):
                load(preset.read_text())

    def test_sampling_policy_cannot_change_on_resume(self):
        base = load(CONFIG)
        for key, value in [('history_iterations', 3), ('win_rate_prior_strength', 2.0),
                           ('sampling_concentration', 0.5)]:
            changed = CONFIG.replace('per_iteration = 5', f'per_iteration = 5\n{key} = {value}')
            self.assertEqual(experiment_changes(base, load(changed)), [f'[starts].{key}'])

    def test_evaluation_resume_preserves_search_but_allows_scheduling_changes(self):
        base = load(CONFIG)
        evaluation = base.evaluation.sets[0]
        for changed, expected in ((replace(evaluation, every=7, at=(0, 7)), []),
                                  (replace(evaluation, search={**evaluation.search, 'particles': 6}), ['[evaluation].sets'])):
            config = replace(base, evaluation=replace(base.evaluation, sets=(changed,)))
            self.assertEqual(experiment_changes(base, config), expected)

    def test_only_resumable_settings_may_change_on_resume(self):
        base = load(CONFIG)
        self.assertEqual(experiment_changes(base, load(CONFIG.replace('iterations = 2', 'iterations = 9'))), [])
        self.assertEqual(experiment_changes(base, load(CONFIG.replace('lr = 0.001', 'lr = 0.01'))), ['[training].lr'])

if __name__ == '__main__':
    unittest.main()
