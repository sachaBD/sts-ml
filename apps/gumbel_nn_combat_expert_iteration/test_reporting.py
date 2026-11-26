import json
from pathlib import Path
import tempfile
import unittest

from .reporting import bosses, by_boss


class BossTest(unittest.TestCase):
    def test_fights_are_grouped_by_their_decks_encounter(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'inputs/evaluation').mkdir(parents=True)
            (out / 'inputs/starts.json').write_text(json.dumps([
                dict(fight_id='champ-deck', start=dict(encounter=39)),
                dict(fight_id='collector-deck', start=dict(encounter=38))]))
            (out / 'inputs/evaluation/validation.jsonl').write_text(
                json.dumps(dict(fight_id='bench:held-out-deck:validation:0', start=dict(encounter=37))) + '\n')
            rates = by_boss({'ns:4:champ-deck:0': True, 'ns:4:champ-deck:1': False, 'ns:4:collector-deck:2': True,
                             'bench:held-out-deck:validation:0': False}, bosses(out))
        self.assertEqual({name: (r['w'], r['n']) for name, r in rates.items()},
                         {'CHAMP': (1, 2), 'COLLECTOR': (1, 1), 'AUTOMATON': (0, 1), 'all': (2, 4)})


if __name__ == '__main__':
    unittest.main()
