"""Setup/dashboard safety tests, no gameplay or training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .expert_champ import prepare
from .expert_champ_status import render


def config():
    return dict(status='SETUP_ONLY',workers=10,updates=2,batch_fights=200,optimizer_steps=1000,
                anchor_states=32,online_states=32,half_life_updates=1.,sims=2000,rollout_mix=.5,
                seed_manifest=None)


class ExpertChampTests(unittest.TestCase):
    def test_missing_status_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'missing'; self.assertIn('NOT CONFIGURED',render(p));self.assertFalse(p.exists())

    def test_prepare_no_launch_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d,patch('apps.run_rl.expert_champ.draft_config',return_value=config()):
            p=Path(d)/'run';prepare(p)
            self.assertIn('SETUP_ONLY',render(p))
            self.assertIn('not launch-ready',render(p))
            self.assertFalse((p/'out/ledger.jsonl').exists())
            before=(p/'out/config.json').read_bytes()
            with self.assertRaises(ValueError):prepare(p)
            self.assertEqual(before,(p/'out/config.json').read_bytes())

    def test_partial_ledger_and_error_are_not_wins(self):
        with tempfile.TemporaryDirectory() as d,patch('apps.run_rl.expert_champ.draft_config',return_value=config()):
            p=Path(d)/'run';prepare(p)
            rows=[dict(kind='intent',key='a'),dict(kind='result',key='a',stage='collect',update=1,status='error',won=True),dict(kind='intent',key='b')]
            path=p/'out/ledger.jsonl';raw='\n'.join(json.dumps(r) for r in rows)+'\n{"kind":';path.write_text(raw)
            text=render(p)
            self.assertIn('wins 0, caps 0, errors 1',text)
            self.assertIn('Unresolved dispatch intents: 1',text)
            self.assertEqual(path.read_text(),raw)


if __name__=='__main__':
    unittest.main()
