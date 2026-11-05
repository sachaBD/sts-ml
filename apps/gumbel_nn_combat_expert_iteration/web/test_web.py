"""CPU fallback counts each inference request once, including GPU-to-CPU retries."""
import unittest

from .server import comparison, cpu_fallback


class FallbackTests(unittest.TestCase):
    def test_missing_or_empty_counters_are_unknown(self):
        self.assertIsNone(cpu_fallback({}))
        self.assertIsNone(cpu_fallback(dict(leaves=0, cpu_leaves=0, cpu_precision_retries=0)))

    def test_all_cpu_and_all_gpu_requests(self):
        for count in (1, 32, 10000):
            self.assertEqual(cpu_fallback(dict(leaves=0, cpu_leaves=count, cpu_precision_retries=0)), 1)
            self.assertEqual(cpu_fallback(dict(leaves=count, cpu_leaves=0, cpu_precision_retries=0)), 0)

    def test_retries_are_not_new_requests(self):
        for count in (1, 32, 10000):
            self.assertEqual(cpu_fallback(dict(leaves=count, cpu_leaves=0, cpu_precision_retries=count)), 1)

    def test_scaling_counters_preserves_rate(self):
        counts = dict(leaves=1234, cpu_leaves=17, cpu_precision_retries=9)
        expected = cpu_fallback(counts)
        self.assertTrue(0 < expected < 1)
        for scale in (2, 32, 10000):
            self.assertEqual(cpu_fallback({k: v * scale for k, v in counts.items()}), expected)


class ComparisonTests(unittest.TestCase):
    def test_coverage_is_explicit_and_shared_fights_are_matched_by_id(self):
        import json
        from pathlib import Path
        import tempfile

        a, b, c = ('ns:alpha:test:0', 'ns:alpha:test:1', 'ns:beta:test:0')
        reference = {'PUCT': {a: True, b: False, c: True}, 'Gumbel': {a: False}}
        with tempfile.TemporaryDirectory() as tmp:
            for name in (None, 'core'):
                out = Path(tmp) / str(name)
                directory = out / 'training/test/iter003' / (name or 'in-sample')
                directory.mkdir(parents=True)
                (directory / 'results.jsonl').write_text(''.join(
                    json.dumps(dict(fight_id=k, won=v)) + '\n' for k, v in {a: False, b: True}.items()))
                row = dict(test=3, **({'set': name} if name else {}))
                table, = comparison(out, [row], reference)
                all_decks = table['rows'][0]
                current = table['columns'][-1]
                self.assertEqual([all_decks['rates'][k]['n'] for k in table['columns']], [3, 1, 2])
                self.assertEqual(all_decks['rates'][current]['w'], 1)
                self.assertEqual([all_decks['matched'][k]['n'] for k in table['columns']], [1, 1, 1])
                self.assertEqual(all_decks['matched'][current]['w'], 0)
                beta = next(r for r in table['rows'] if r['deck'] == 'beta')
                self.assertIsNone(beta['rates']['Gumbel']['p'])


class LossApiTests(unittest.TestCase):
    def test_structured_losses_override_legacy_and_missing_heldout_stays_missing(self):
        from pathlib import Path
        from unittest.mock import patch
        from . import server as web

        history = [dict(iteration=1, games=10, train_value_loss=0.3, train_policy_loss=1.2,
                        heldout_value_loss=0.4, heldout_policy_loss=1.3),
                   dict(iteration=2, games=10),
                   dict(iteration=3, games=10, heldout_value_loss=0.0)]
        snapshot = dict(progress={}, state='completed', config={'run': {'id': 'test'}})
        with patch.object(web, 'snapshot', return_value=snapshot), \
             patch.object(web, 'jsonl', return_value=history), \
             patch.object(web, 'finished', return_value={}), \
             patch.object(web, 'losses', return_value={1: dict(value=9, policy=9),
                                                     2: dict(value=0.5, policy=1.5)}), \
             patch.object(web, 'system', return_value={}), \
             patch.object(web, 'comparison', return_value=None):
            rows = web.api(Path('/unused/out'))['train']
        self.assertEqual(rows[0]['loss'], dict(value=0.3, policy=1.2,
                                             heldout_value=0.4, heldout_policy=1.3))
        self.assertEqual(rows[1]['loss'], dict(value=0.5, policy=1.5))
        self.assertEqual(rows[2]['loss'], dict(heldout_value=0.0))


if __name__ == '__main__':
    unittest.main()
