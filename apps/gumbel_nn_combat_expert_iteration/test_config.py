"""Invariants of the config: a mistake stops the run before anything is written; resume may only change resumable settings."""
from pathlib import Path
import unittest
from unittest.mock import patch

from .config import experiment_changes, load
from .test_expert_iteration import CONFIG

PRESET = Path(__file__).with_name('config') / 'champ_ten.toml'


class ConfigTests(unittest.TestCase):
    def rejects(self, text, message):
        with patch('sys.exit', side_effect=SystemExit) as exit, self.assertRaises(SystemExit):
            load(text)
        self.assertIn(message, exit.call_args.args[0])

    def test_the_preset_is_valid(self):
        config = load(PRESET.read_text())
        self.assertEqual(config.starts.fights, 'runs/benchmark/champ-ten/decks.jsonl')
        self.assertEqual(config.workers.gpu_bank, (256, 32))

    def test_a_mistake_stops_the_run(self):
        self.rejects(CONFIG.replace('steps = 1', 'stepz = 1'), "unknown [training] keys: ['stepz']")
        self.rejects(CONFIG.replace('lr = 0.001\n', ''), 'missing [training].lr')
        self.rejects(CONFIG.replace('actors = 2', 'actors = "2"'), '[workers].actors must be int')
        self.rejects(CONFIG.replace('gpu_bank = [256, 32]', 'gpu_bank = [256, 600]'), 'gpu_bank')
        self.rejects(CONFIG.replace('rollout_mix = 0.5', 'rollout_mix = 2'), 'rollout_mix')
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nhp_jitter = -0.1'), 'hp_jitter')
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nexcluded_cards = ["PURTY"]'), "['PURTY']")
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nexcluded_relics = ["RUNIC_DOOM"]'), "['RUNIC_DOOM']")
        self.rejects(CONFIG.replace('per_iteration = 5', 'per_iteration = 5\nexcluded_potions = [15]'), 'list of str')
        self.rejects(CONFIG.replace('rollout_mix = 0.5', 'rollout_mix = 0.5\nrollout_mix_decay = -1'), 'rollout_mix_decay')

    def test_value_sampling_is_opt_in_and_defines_the_experiment(self):
        base = load(CONFIG)
        explicit_default = load(CONFIG.replace('steps = 1', 'steps = 1\nvalue_batch = 0'))
        self.assertEqual(base, explicit_default)
        sparse = load(CONFIG.replace('steps = 1', 'steps = 1\nvalue_batch = 128'))
        self.assertEqual(experiment_changes(base, sparse), ['[training].value_batch'])
        self.rejects(CONFIG.replace('steps = 1', 'steps = 1\nvalue_batch = -1'), 'value_batch')
        self.rejects(CONFIG.replace('steps = 1', 'steps = 1\nvalue_batch = 1.5'), 'value_batch')
        self.rejects(CONFIG.replace('steps = 1', 'steps = 1\nvalue_batch = 128')
                     .replace('max_train_rows = 0', 'max_train_rows = 500000'), 'max_train_rows')

    def test_rollout_mix_decays_to_zero_and_stays(self):
        for decay in (0, 1, 7, 30):
            search = load(CONFIG.replace('rollout_mix = 0.5', f'rollout_mix = 0.5\nrollout_mix_decay = {decay}')).search
            mixes = [search.at(it).rollout_mix for it in range(1, 70)]
            self.assertTrue(all(0 <= b <= a <= 0.5 for a, b in zip(mixes, mixes[1:])))
            self.assertEqual(mixes[-1], 0 if decay else 0.5)
            if decay:
                self.assertEqual(search.at(decay).rollout_mix, 0)
                self.assertGreater(search.at(decay - 1).rollout_mix, 0)

    def test_only_resumable_settings_may_change_on_resume(self):
        base = load(CONFIG)
        self.assertEqual(experiment_changes(base, load(CONFIG.replace('iterations = 2', 'iterations = 9'))), [])
        self.assertEqual(experiment_changes(base, load(CONFIG.replace('lr = 0.001', 'lr = 0.01'))), ['[training].lr'])

if __name__ == '__main__':
    unittest.main()
