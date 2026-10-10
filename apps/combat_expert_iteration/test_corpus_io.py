import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa

from .corpus_io import freeze_json, journal_rows, play_stage


class CorpusIOTest(unittest.TestCase):
    def test_frozen_inputs_and_torn_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            freeze_json(path, {'x': 1})
            freeze_json(path, {'x': 1})
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                freeze_json(path, {'x': 2})
            journal = Path(tmp) / 'results.jsonl'
            good = json.dumps({'fight_id': 'a', 'status': 'capped'}) + '\n'
            journal.write_text(good + '{"fight_id":')
            self.assertEqual(set(journal_rows(journal, repair=True)), {'a'})
            self.assertEqual(journal.read_text(), good)

    def test_resume_does_not_replay_and_full_start_changes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'worker'
            worker.write_text('frozen')
            starts = [{'fight_id': 'a', 'start': {'seed': 1, 'hp': 10}}]
            schema = SimpleNamespace(FIGHTS=pa.schema([('won', pa.bool_())]), SEARCH=pa.schema([('step', pa.int32())]))
            result = dict(fight_id='a', status='completed', fight={'won': True}, search=[])
            with patch('apps.combat_expert_iteration.corpus_io.play_one', return_value=result) as play, patch('apps.combat_expert_iteration.corpus_io.load', return_value=schema):
                command = [str(worker), 'teacher', '20000']
                play_stage(root / 'stage', starts, command, 1)
                play_stage(root / 'stage', starts, command, 1)
                self.assertEqual(play.call_count, 1)
                starts[0]['start']['hp'] = 11
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    play_stage(root / 'stage', starts, command, 1)

    def test_cached_error_is_not_retried_or_counted_as_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'worker'
            worker.write_text('frozen')
            starts = [{'fight_id': 'a'}]
            with patch('apps.combat_expert_iteration.corpus_io.play_one', return_value={'fight_id': 'a', 'status': 'error', 'error': 'timeout'}):
                with self.assertRaisesRegex(RuntimeError, 'worker error'):
                    play_stage(root / 'stage', starts, [str(worker), 'teacher', '1'], 1)
                with self.assertRaisesRegex(RuntimeError, 'cached infrastructure'):
                    play_stage(root / 'stage', starts, [str(worker), 'teacher', '1'], 1)


if __name__ == '__main__':
    unittest.main()
