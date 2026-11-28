import unittest

from agents.combat.search.voi_gym.model import adaptive_belief

from .explorer import replay
from .session import Session


class ExplorerTests(unittest.TestCase):
    def test_frames_match_automatic_session(self):
        seed, budget, price = 4, 4, .001
        session = Session(seed=seed, budget=budget, price=price)
        observation = None
        frames = replay(seed, budget, price)['frames']
        for frame in frames:
            self.assertEqual(frame['state'], session.snapshot())
            self.assertEqual(frame['rows'], session.candidates())
            self.assertEqual(frame['observation'], observation)
            observation = session.step()
            self.assertEqual(frame['selected'], observation['id'] if observation else None)
        self.assertIsNone(observation)
        self.assertEqual(frames, replay(seed, budget, price)['frames'])

    def test_stop_does_not_hide_unaffordable_measurements(self):
        initial, = replay(0, 0, 0)['frames']
        self.assertIsNone(initial['selected'])
        self.assertIsNone(initial['observation'])
        self.assertTrue(initial['rows'])
        self.assertFalse(any(row['affordable'] for row in initial['rows']))

    def test_prior_scale_changes_value_gaps_not_policy_or_uncertainty(self):
        def initial(scale):
            return replay(4, 0, .001, prior_scale=scale)['frames'][0]['state']

        flat, default, wider = map(initial, (0, .03, .06))
        for frame in (flat, wider):
            self.assertEqual(frame['prior'], default['prior'])
            self.assertEqual(frame['covariance'], default['covariance'])
            self.assertEqual(frame['root_value'], default['root_value'])
        for action in flat['actions']:
            self.assertEqual(flat['root_q'][action], flat['root_value'])
            self.assertAlmostEqual(wider['root_q'][action] - flat['root_value'],
                                   2*(default['root_q'][action] - flat['root_value']))

    def test_zero_initial_sd_cap_keeps_probe_noise_but_eliminates_gain(self):
        frame, = replay(4, 4, 0, initial_sd_cap=0)['frames']
        default = replay(4, 4, 0)['frames'][0]
        self.assertEqual(frame['state']['root_q'], default['state']['root_q'])
        self.assertTrue(all(sd == 0 for sd in frame['state']['sd'].values()))
        self.assertIsNone(frame['selected'])
        for row in frame['rows']:
            self.assertEqual(row['gain'], 0)
            self.assertAlmostEqual(row['base_sd']**2, row['noise_var'])
            self.assertGreater(row['predictive_sd'], 0)

    def test_adaptive_belief_is_opt_in(self):
        fixed = replay(4, 4, 0)['frames']
        mixed = replay(4, 4, 0, belief_factory=adaptive_belief)['frames']
        for frame in fixed:
            self.assertEqual(frame['state']['scales'], [1.])
            self.assertEqual(frame['state']['weights'], [1.])
            for row in frame['rows']:
                self.assertAlmostEqual(row['predictive_sd'], row['base_sd'])
        self.assertEqual(mixed[0]['state']['scales'], [1., 3.])
        self.assertEqual(fixed[0]['state']['root_q'], mixed[0]['state']['root_q'])
        self.assertNotEqual(mixed[-1]['state']['weights'], mixed[0]['state']['weights'])

    def test_shortlist_and_noise_are_effective_parameters(self):
        frame = replay(4, 0, 0, shortlist=2, probe_sd=.12)['frames'][0]
        s = frame['state']
        self.assertEqual(s['actions'], sorted(s['prior'], key=s['prior'].get, reverse=True)[:2])
        row, = frame['rows']
        self.assertAlmostEqual(row['noise_var'], .12**2)
        self.assertEqual(s['shortlist'], 2)
        self.assertEqual(s['probe_sd'], .12)
        only, = replay(4, 4, 0, shortlist=1)['frames']
        self.assertEqual(only['rows'], [])
        self.assertIsNone(only['selected'])
