"""Named evaluation inputs are included in lineage and frozen independently."""
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from . import prepare
from .test_expert_iteration import CONFIG, EVALUATION


class PrepareTests(unittest.TestCase):
    def test_freezes_each_named_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second, worker = (root / name for name in ('first.jsonl', 'second.jsonl', 'nn_play'))
            first.write_text('{"fight_id": "first", "start": {}}\n')
            second.write_text('{"fight_id": "second", "start": {}}\n')
            worker.touch()
            text = CONFIG.replace('worker = "nn_play"', f'worker = "{worker}"').replace('"unused"', f'"{first}"')
            text += EVALUATION.replace('"validation"', '"retention"').replace('"unused"', f'"{second}"')
            config = root / 'config.toml'
            config.write_text(text)
            self.assertEqual(set(prepare.inputs(tomllib.loads(text))), {str(first), str(second)})
            out = root / 'out'
            with patch.object(prepare.subprocess, 'run'):
                prepare.run(None, config, out)
            for name, source in (('validation', first), ('retention', second)):
                original = source.read_bytes()
                source.write_text('changed after launch')
                self.assertEqual((out / 'inputs/evaluation' / f'{name}.jsonl').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
