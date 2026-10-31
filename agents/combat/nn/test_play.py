"""The nn_play command: every setting reaches the native process, and finished fights are journaled."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from . import play


class PlayTests(unittest.TestCase):
    def test_command_and_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = [json.dumps(dict(fight_id='f', won=True)) + '\n']
            process.returncode = 0
            search = play.Search(sims=0, gumbel_scale=0.0, rollout_mix=0.5, particles=8)
            workers = play.Workers(actors=2, slots=4, gpu_batch=512, gpu_bank=(32,), gpu_wait_us=100,
                                   timeout_ms=321, shutdown_timeout_ms=123)
            with patch.object(play.subprocess, 'Popen', return_value=process) as popen:
                results = play.play(Path('nn_play'), root, [{'fight_id': 'f', 'start': {}}], root / 'out', search,
                                    workers, sample_devices=False)
            self.assertEqual(results, [dict(fight_id='f', won=True)])
            self.assertEqual((root / 'out/results.jsonl').read_text().strip(), json.dumps(dict(fight_id='f', won=True)))
            command = popen.call_args.args[0]
            options = dict(zip(command[4::2], command[5::2]))
            self.assertEqual(command[:4], ['nn_play', 'play', str(root / 'model.onnx'), str(root / 'out/samples-0.bin')])
            self.assertEqual(options, {'--threads': '2', '--parallel': '4', '--gpu-model': str(root / 'model-gpu.onnx'),
                                       '--gpu-batch': '512', '--gpu-wait-us': '100', '--timeout-ms': '321',
                                       '--shutdown-timeout-ms': '123', '--sims': '0', '--gumbel-scale': '0.0',
                                       '--rollout-mix': '0.5', '--particles': '8',
                                       '--gpu-bank': str(root / 'model-gpu-b32.onnx')})

    def test_refuses_to_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'samples-0.bin').touch()
            with self.assertRaises(FileExistsError):
                play.play(Path('nn_play'), directory, [], directory, None, None)


if __name__ == '__main__':
    unittest.main()
