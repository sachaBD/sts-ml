"""Bad inputs fail at load; resume cannot change the experiment's meaning."""
from pathlib import Path
import unittest

from ._test_support import CONFIG, POOLED_CONFIG
from .config import experiment_changes, load

PRESET = Path(__file__).with_name('config') / 'champ_ten.toml'


class ConfigTests(unittest.TestCase):
    def test_maintained_preset_loads(self):
        load(PRESET.read_text())

    def test_unknown_fields_are_rejected(self):
        typo = CONFIG.replace('steps = 1', 'stepz = 1')

        with self.assertRaisesRegex(SystemExit, 'stepz'):
            load(typo)

    def test_required_fields_cannot_be_omitted(self):
        missing_learning_rate = CONFIG.replace('lr = 0.001\n', '')

        with self.assertRaisesRegex(SystemExit, 'lr'):
            load(missing_learning_rate)

    def test_wrong_field_types_are_rejected(self):
        wrong_type = CONFIG.replace('actors = 2', 'actors = "two"')

        with self.assertRaisesRegex(SystemExit, 'actors'):
            load(wrong_type)

    def test_bank_batches_cannot_exceed_the_main_batch(self):
        oversized_bank = CONFIG.replace('gpu_bank = [256, 32]', 'gpu_bank = [1024]')

        with self.assertRaisesRegex(SystemExit, 'gpu_bank'):
            load(oversized_bank)

    def test_rollout_mixing_must_be_a_probability(self):
        invalid_mix = CONFIG.replace('rollout_mix = 0.5', 'rollout_mix = 2')

        with self.assertRaisesRegex(SystemExit, 'rollout_mix'):
            load(invalid_mix)

    def test_unknown_model_kind_is_rejected(self):
        unknown_model = CONFIG.replace('kind = "transformer"', 'kind = "unknown"')

        with self.assertRaisesRegex(SystemExit, 'kind'):
            load(unknown_model)

    def test_attention_pooling_capacity_requires_a_pooled_model(self):
        unpooled = CONFIG.replace('kind = "transformer"', 'kind = "transformer"\ngpu_attended = 8')

        with self.assertRaisesRegex(SystemExit, 'gpu_attended'):
            load(unpooled)

    def test_attention_capacity_cannot_exceed_token_capacity(self):
        oversized_attention = POOLED_CONFIG.replace('gpu_attended = 8', 'gpu_attended = 17')

        with self.assertRaisesRegex(SystemExit, 'gpu_attended'):
            load(oversized_attention)

    def test_attention_capacity_must_be_positive(self):
        empty_attention = POOLED_CONFIG.replace('gpu_attended = 8', 'gpu_attended = 0')

        with self.assertRaisesRegex(SystemExit, 'gpu_attended'):
            load(empty_attention)

    def test_iteration_limit_can_change_on_resume(self):
        original = load(CONFIG)
        longer = load(CONFIG.replace('iterations = 2', 'iterations = 9'))

        self.assertEqual(experiment_changes(original, longer), [])

    def test_learning_rate_cannot_change_on_resume(self):
        original = load(CONFIG)
        faster = load(CONFIG.replace('lr = 0.001', 'lr = 0.01'))

        self.assertIn('[training].lr', experiment_changes(original, faster))

    def test_model_kind_cannot_change_on_resume(self):
        self.assertIn('[model].kind', experiment_changes(load(CONFIG), load(POOLED_CONFIG)))

    def test_deployment_capacity_cannot_change_on_resume(self):
        original = load(POOLED_CONFIG)
        larger = load(POOLED_CONFIG.replace('gpu_attended = 8', 'gpu_attended = 9'))

        self.assertIn('[model].gpu_attended', experiment_changes(original, larger))


if __name__ == '__main__':
    unittest.main()
