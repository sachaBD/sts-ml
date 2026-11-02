"""The native worker receives the capacities of the models it actually loads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from .play import Search, Workers, play


class PlayDeploymentTests(unittest.TestCase):
    def test_attention_limits_and_wide_model_are_optional_and_match_the_deployment(self):
        workers = Workers(actors=1, slots=1, gpu_batch=4, gpu_bank=(2,), gpu_wait_us=0,
                          timeout_ms=1000, shutdown_timeout_ms=1000)
        for narrow, wide in ((0, 0), (4, 0), (4, 8)):
            with self.subTest(narrow=narrow, wide=wide), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                process = MagicMock()
                process.__enter__.return_value = process
                process.stdout = ['{"fight_id":"test"}\n']
                process.returncode = 0
                with patch('agents.combat.nn.play.subprocess.Popen', return_value=process) as start:
                    results = play(root / 'nn_play', root / 'models', [dict(fight_id='test')], root / 'play',
                                   Search(0, 0, 0, 1), workers, sample_devices=False,
                                   gpu_attended=narrow, gpu_wide_attended=wide)
                command = start.call_args.args[0]
                options = dict(zip(command[4::2], command[5::2]))
                self.assertEqual(len(results), 1)
                self.assertEqual('--gpu-attended' in options, bool(narrow))
                self.assertEqual('--gpu-wide-model' in options, bool(wide))
                self.assertEqual('--gpu-wide-attended' in options, bool(wide))
                if narrow:
                    self.assertEqual(int(options['--gpu-attended']), narrow)
                if wide:
                    self.assertEqual(int(options['--gpu-wide-attended']), wide)
                    self.assertEqual(Path(options['--gpu-wide-model']), root / 'models/wide/model-gpu.onnx')


if __name__ == '__main__':
    unittest.main()
