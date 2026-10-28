"""Failure injection for the bounded CUDA supervisor; no real GPU/job needed."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import supervisor


class SupervisorTests(unittest.TestCase):
    def fixture(self, root, message='workers: failed: CUDA: unknown error'):
        out = root / 'out'; training = out / 'training'; training.mkdir(parents=True)
        (training / 'iter001').mkdir(); (training / 'iter001/model.pt').touch()
        (training / 'log.jsonl').write_text('{"iteration": 1}\n')
        partial = training / 'iter002'; partial.mkdir()
        (partial / 'worker.log').write_text(message); (partial / 'samples-0.bin').write_bytes(b'partial')
        (training / 'status.json').write_text(json.dumps(dict(directory=str(partial), phase='play')))
        command = ['python', '-m', 'fixture']
        (out / 'command.json').write_text(json.dumps(command))
        return out, command, partial

    def test_cuda_recovery_archives_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, command, partial = self.fixture(Path(tmp))
            with patch.object(supervisor.subprocess, 'run', side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as run, patch.object(supervisor.time, 'sleep'):
                supervisor.launch(command, out)
            self.assertFalse(partial.exists())
            self.assertEqual(len(list((out / 'training').glob('.interrupted-*/samples-0.bin'))), 1)
            self.assertEqual(run.call_args_list[-1].args[0][-2:], ['--resume-from', '1'])
            self.assertEqual(len(supervisor.jsonl(out / 'recovery.jsonl')), 1)

    def test_unknown_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, command, partial = self.fixture(Path(tmp), 'workers: failed: unsupported card')
            with patch.object(supervisor.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run, self.assertRaises(subprocess.CalledProcessError):
                supervisor.launch(command, out)
            self.assertEqual(run.call_count, 1); self.assertTrue(partial.exists())

    def test_stop_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, command, partial = self.fixture(Path(tmp)); (out / 'training/STOP').touch()
            with patch.object(supervisor.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run, self.assertRaises(subprocess.CalledProcessError):
                supervisor.launch(command, out)
            self.assertEqual(run.call_count, 1); self.assertTrue(partial.exists())

    def test_retries_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, command, partial = self.fixture(Path(tmp))
            def fail(*args, **kwargs):
                partial.mkdir(exist_ok=True)
                (partial / 'worker.log').write_text('workers: failed: CUDA: unknown error')
                (partial / 'samples-0.bin').write_bytes(b'partial')
                return SimpleNamespace(returncode=1)
            with patch.object(supervisor.subprocess, 'run', side_effect=fail) as run, patch.object(supervisor.time, 'sleep'), self.assertRaises(subprocess.CalledProcessError):
                supervisor.launch(command, out, retries=2)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(len(supervisor.jsonl(out / 'recovery.jsonl')), 2)


if __name__ == '__main__': unittest.main()
