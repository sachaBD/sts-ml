"""Mock scenarios remain synthetic and mark missing/stale values explicitly."""
import unittest


class MockBackendTests(unittest.TestCase):
    def test_mock_rates_preserve_request_accounting(self):
        from .health_mock import MockHealth
        for scenario in ('normal', 'fallback'):
            data = MockHealth(scenario).view()
            metric = lambda key: data['metrics'][key]['current']
            self.assertAlmostEqual(metric('nn_s'), metric('gpu_s') + metric('cpu_oversized_s'))
            self.assertAlmostEqual(metric('fallback') * metric('nn_s') / 100,
                                   metric('cpu_oversized_s') + metric('precision_retry_s'))
            self.assertTrue(data['mock'])

    def test_mock_stale_and_missing_are_not_zero_fallback(self):
        from .health_mock import MockHealth
        self.assertTrue(MockHealth('stale').view()['metrics']['fallback']['stale'])
        self.assertNotIn('fallback', MockHealth('missing').view()['metrics'])


if __name__ == '__main__':
    unittest.main()
