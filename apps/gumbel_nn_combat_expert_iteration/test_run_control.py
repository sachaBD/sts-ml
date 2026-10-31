"""One owner per run, resumption only from the last completed iteration, pruning keeps checkpoints."""
from pathlib import Path
import tempfile
import unittest

from .run_control import RunControl


class RunControlTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.control = RunControl(Path(tmp.name))

    def test_one_owner(self):
        with self.control.lock(), self.assertRaises(BlockingIOError):
            with RunControl(self.control.directory).lock():
                pass

    def test_resume_only_from_the_last_completed_iteration(self):
        self.control.log(iteration=0)
        self.control.log(iteration=1)
        with self.assertRaises(ValueError):
            self.control.check_resume(0)
        self.control.check_resume(1)
        self.control.iteration(2).mkdir()
        with self.assertRaises(FileExistsError):
            self.control.check_resume(1)

    def test_prune_keeps_checkpoints(self):
        for k in range(3):
            self.control.iteration(k).mkdir()
            for name in ('samples-0.bin', 'samples-0.index.npz', 'model.pt'):
                (self.control.iteration(k) / name).touch()
            (self.control.iteration(k) / 'tensorrt').mkdir()
        self.control.prune(keep_from=2)
        for k in range(3):
            kept = sorted(p.name for p in self.control.iteration(k).iterdir())
            self.assertEqual(kept, ['model.pt'] if k < 2 else ['model.pt', 'samples-0.bin', 'samples-0.index.npz', 'tensorrt'])


if __name__ == '__main__':
    unittest.main()
