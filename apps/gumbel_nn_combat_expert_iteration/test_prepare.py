"""Prepared runs own their evaluation inputs, independently of mutable source files."""
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from . import prepare
from .test_expert_iteration import CONFIG


class PrepareTests(unittest.TestCase):
    def test_all_sets_are_frozen_and_recorded_as_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'nn_play'
            worker.write_text('worker')
            decks = root / 'decks.jsonl'
            decks.write_text(json.dumps(dict(fight_id='deck', start={})) + '\n')
            first, second, final = (root / name for name in ('first.jsonl', 'second.jsonl', 'final.jsonl'))
            for i, path in enumerate((first, second, final)):
                path.write_text(json.dumps(dict(fight_id=f'ns:deck:test:{i}', start={'hp': i + 1})) + '\n')
            text = (CONFIG.replace('worker = "nn_play"', f'worker = "{worker}"')
                    .replace('fights = "unused"', f'fights = "{decks}"', 1)
                    .replace('fights = "unused"', f'fights = "{first}"')
                    .replace('[[evaluation.sets]]', f'[evaluation]\nfinal_fights = "{final}"\n\n[[evaluation.sets]]'))
            text += ('[[evaluation.sets]]' + CONFIG.split('[[evaluation.sets]]')[1]).replace(
                'name = "validation"', 'name = "retention"').replace('fights = "unused"', f'fights = "{second}"')
            config = root / 'config.toml'
            config.write_text(text)
            expected = {str(path) for path in (decks, first, second, final)}
            self.assertEqual(set(prepare.inputs(tomllib.loads(text))), expected)
            out = root / 'out'
            with patch.object(prepare.subprocess, 'run'):
                prepare.run(None, config, out)
            for name, source in (('validation', first), ('retention', second)):
                frozen = out / 'inputs/evaluation' / f'{name}.jsonl'
                original = source.read_bytes()
                source.write_text('changed after launch')
                self.assertEqual(frozen.read_bytes(), original)
                self.assertEqual(frozen.stat().st_mode & 0o222, 0)
            self.assertEqual((out / 'inputs/FINAL_TEST_DO_NOT_USE_FOR_SELECTION.jsonl').read_bytes(), final.read_bytes())
            self.assertEqual((out / 'config.toml').read_text(), text)


if __name__ == '__main__':
    unittest.main()
