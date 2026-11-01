"""Preview is harmless, deletion is selective, and active writers are protected."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from .cleanup import main
from .run_control import RunControl


class CleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.training = self.run / 'out/training'
        iteration = self.training / 'iter001'
        self.cache = iteration / 'tensorrt/network.engine'
        self.replay = iteration / 'samples-0.bin'
        self.index = iteration / 'samples-0.index.npz'
        self.evidence = [iteration / 'model.pt', iteration / 'model.onnx', iteration / 'results.jsonl']
        self.files = [self.cache, self.replay, self.index, *self.evidence]
        for path in self.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'keep unless explicitly disposable')
        (self.training / '.lock').touch()
        (self.run / 'run.json').write_text(json.dumps(dict(schema='gumbel_nn_combat_v1', status='done')))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_preview_does_not_change_data(self):
        before = {path: path.read_bytes() for path in self.files}

        main([str(self.run), '--drop-replay'])

        self.assertEqual({path: path.read_bytes() for path in self.files}, before)

    def test_apply_removes_only_the_explicitly_disposable_data(self):
        evidence = {path: path.read_bytes() for path in self.evidence}

        main([str(self.run), '--apply'])
        self.assertFalse(self.cache.exists())
        self.assertTrue(self.replay.exists())
        main([str(self.run), '--apply', '--drop-replay'])

        self.assertFalse(self.replay.exists())
        self.assertFalse(self.index.exists())
        self.assertEqual({path: path.read_bytes() for path in self.evidence}, evidence)

    def test_active_run_cannot_be_cleaned(self):
        with RunControl(self.training).lock(), self.assertRaises(SystemExit):
            main([str(self.run), '--apply', '--drop-replay'])

        self.assertTrue(all(path.exists() for path in self.files))


if __name__ == '__main__':
    unittest.main()
