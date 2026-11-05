"""CPU fallback counts each inference request once, including GPU-to-CPU retries."""
import unittest

from .server import cpu_fallback


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


class LossApiTests(unittest.TestCase):
    def test_named_sets_do_not_become_empty_legacy_evaluation_points(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from . import server as web

        with tempfile.TemporaryDirectory() as tmp, patch.object(web, 'system', return_value={}):
            out = Path(tmp)
            (out / 'training').mkdir()
            (out / 'config.toml').write_text('[run]\nid = "test"\n')
            (out / 'training/log.jsonl').write_text('{"test": 0, "set": "retention", "games": 10, "wins": 7, "seconds": 1}\n')
            data = web.api(out)
            self.assertEqual(data['test'], [])
            self.assertTrue(all('eval' not in row for row in data['trend']))

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
