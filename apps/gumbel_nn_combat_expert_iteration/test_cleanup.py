"""A retained iteration survives cleanup in full, not just its checkpoint."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from .cleanup import main


class CleanupTests(unittest.TestCase):
    def test_preview_preserves_everything_and_apply_preserves_kept_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'run.json').write_text(json.dumps(dict(schema='gumbel_nn_combat_v1', status='done')))
            training = run / 'out/training'
            training.mkdir(parents=True)
            (training / '.lock').touch()
            original = {}
            for iteration in (69, 70):
                for name in ('samples-0.bin', 'samples-0.index.npz', 'model.pt', 'results.jsonl',
                             'tensorrt/model.engine', 'tensorrt/model.profile'):
                    path = training / f'iter{iteration:03d}' / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    original[path] = f'{iteration}/{name}'.encode()
                    path.write_bytes(original[path])
            options = [str(run), '--drop-replay', '--keep-iteration', '70']
            with contextlib.redirect_stdout(io.StringIO()):
                main(options)
            self.assertTrue(all(path.read_bytes() == data for path, data in original.items()))
            with contextlib.redirect_stdout(io.StringIO()):
                main(options + ['--apply'])
            for path, data in original.items():
                if 'iter070' in path.parts or path.name in ('model.pt', 'results.jsonl'):
                    self.assertEqual(path.read_bytes(), data)
                else:
                    self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
