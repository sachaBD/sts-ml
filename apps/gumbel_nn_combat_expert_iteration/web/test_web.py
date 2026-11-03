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


if __name__ == '__main__':
    unittest.main()
