"""Evaluation trends only pair common starts within each deck."""
from pathlib import Path
import unittest
from unittest.mock import patch

from . import server as web


class EvaluationTrendTests(unittest.TestCase):
    def test_grouping_preserves_paired_rates_and_intervals(self):
        old = {'ns:alpha:test:0': False, 'ns:alpha:test:1': True,
               'ns:beta:test:0': True, 'ns:beta:test:1': False,
               'ns:old_only:test:0': True}
        new = {'ns:alpha:test:0': True, 'ns:alpha:test:1': True,
               'ns:beta:test:0': False, 'ns:beta:test:1': False,
               'ns:new_only:test:0': True}
        results = {0: old, 1: old, 2: new, 3: new}
        snapshot = dict(progress={}, state='completed', config={'run': {'id': 'test'}})
        with patch.object(web, 'snapshot', return_value=snapshot), \
             patch.object(web, 'jsonl', return_value=[{'test': i} for i in results]), \
             patch.object(web, 'finished', side_effect=lambda p: results[int(p.parent.name[4:])]), \
             patch.object(web, 'losses', return_value={}), \
             patch.object(web, 'system', return_value={}), \
             patch.object(web, 'comparison', return_value=None):
            response = web.api(Path('/unused/out'))
        for row in response['trend']:
            deck = row['deck']
            keys = [k for k in new if k in old and (deck == 'all' or web.deck_of(k) == deck)]
            self.assertEqual(row['eval'], web.rate(sum(new[k] for k in keys), len(keys)))
            self.assertEqual(row['eval_d'], web.paired(new, old, keys))


if __name__ == '__main__':
    unittest.main()
