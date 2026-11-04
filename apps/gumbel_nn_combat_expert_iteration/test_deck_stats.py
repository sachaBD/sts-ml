import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import deck_stats


class DeckStatsTest(unittest.TestCase):
    def test_iteration_window_subtracts_baseline_in_newest_experiment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(deck_stats, 'RUNS', Path(tmp)):
            def run(name, started, snapshots):
                root = Path(tmp) / 'schema=gumbel_nn_combat_v1/date=2026-01-01' / f'id={name}'
                root.mkdir(parents=True)
                (root / 'run.json').write_text(json.dumps(dict(started=started)))
                for iteration, counts in snapshots.items():
                    path = root / 'out/training' / f'iter{iteration:03d}' / 'decks.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(counts))

            run('z-old', '2026-01-01T00:00:00+00:00', {1: {'old': dict(fights=99, wins=99)}})
            run('a-new', '2026-01-02T00:00:00+00:00', {
                0: {'b': dict(fights=0, wins=0), 'a': dict(fights=0, wins=0)},
                998: {'b': dict(fights=2, wins=1), 'a': dict(fights=0, wins=0)},
                999: {'b': dict(fights=5, wins=3), 'a': dict(fights=0, wins=0)},
                1000: {'b': dict(fights=9, wins=5), 'a': dict(fights=0, wins=0)},
            })
            for n, attempts, wins in [(1, 4, 2), (2, 7, 4), (2000, 9, 5)]:
                self.assertEqual(deck_stats.newest_runs(n).to_pylist(), [
                    dict(deck_id='a', attempts=0, wins=0), dict(deck_id='b', attempts=attempts, wins=wins),
                ])
            run('empty', '2026-01-03T00:00:00+00:00', {})
            self.assertEqual(deck_stats.newest_runs(1).num_rows, 0)

    def test_empty_table_has_typed_columns_and_invalid_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(deck_stats, 'RUNS', Path(tmp)):
            table = deck_stats.newest_runs(3)
            self.assertEqual(table.column_names, ['deck_id', 'attempts', 'wins'])
            self.assertEqual(table.num_rows, 0)
            for limit in (0, -1):
                with self.assertRaises(ValueError):
                    deck_stats.newest_runs(limit)
