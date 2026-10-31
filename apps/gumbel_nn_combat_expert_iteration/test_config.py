"""Config mistakes stop the run before anything is written."""
from pathlib import Path
import unittest
from unittest.mock import patch

from .config import Config, changed, load
from .test_expert_iteration import CONFIG

PRESET = Path(__file__).with_name('config') / 'champ_ten.toml'


class ConfigTests(unittest.TestCase):
    def rejects(self, text, message):
        with patch('sys.exit', side_effect=SystemExit) as exit, self.assertRaises(SystemExit):
            load(text)
        self.assertIn(message, exit.call_args.args[0])

    def test_preset_loads(self):
        config = load(PRESET.read_text())
        self.assertEqual(len(config.decks), 10)
        self.assertEqual(config.workers.gpu_bank, (256, 32))

    def test_typos_missing_and_mistyped_settings(self):
        self.rejects(CONFIG.replace('steps = 1', 'stepz = 1'), "unknown [training] keys: ['stepz']")
        self.rejects(CONFIG.replace('lr = 0.001\n', ''), 'missing [training].lr')
        self.rejects(CONFIG.replace('actors = 2', 'actors = "2"'), '[workers].actors must be int')
        self.rejects(CONFIG.replace('gpu_bank = [256, 32]', 'gpu_bank = [256, 600]'), 'gpu_bank')
        self.rejects(CONFIG.replace('rollout_mix = 0.5', 'rollout_mix = 2'), 'rollout_mix')

    def test_resumable_changes(self):
        base = load(CONFIG)
        self.assertEqual(changed(base, load(CONFIG.replace('iterations = 2', 'iterations = 9'))), {'run.iterations'})
        self.assertFalse(changed(base, load(CONFIG.replace('lr = 0.001', 'lr = 0.01'))) <= Config.RESUMABLE)


if __name__ == '__main__':
    unittest.main()
