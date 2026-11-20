"""Physical boundaries and evaluator cache behavior for stop-to-stop running."""
import unittest
import numpy as np
from rail_gen.services import running_hours, Service, ServiceNetwork
from rail_gen.evaluator import Evaluator
from rail_gen.config import load_config
from test_railgen import config, graph_fixture


class RunningTimeTests(unittest.TestCase):
    def test_triangular_cruise_and_boundary(self):
        # 36 km/h = 10 m/s; a=1 gives a 100 m threshold.
        np.testing.assert_allclose(running_hours(np.array([0, .025, .1, .2]), 36, 1),
                                   np.array([0, 10, 20, 30]) / 3600)
        self.assertAlmostEqual(float(running_hours(.1 - 1e-9, 36, 1)),
                               float(running_hours(.1 + 1e-9, 36, 1)), places=9)

    def test_short_hsr_loses_advantage(self):
        self.assertGreater(running_hours(5, 230, .4), running_hours(5, 140, .7))
        self.assertLess(running_hours(50, 230, .4), running_hours(50, 140, .7))

    def test_cache_respects_stock_and_infrastructure_and_rebuilds(self):
        c = config()
        e = Evaluator(graph_fixture(), c)
        self.assertTrue(np.all(np.isinf(e.running_times['CR'][:, 2])))
        self.assertGreater(e.running_times['HSR'][0, 1], e.running_times['CR'][0, 1])
        cache = e.running_times['HSR']
        for frequency, formation in [(1, 1), (2, 2)]:
            network = ServiceNetwork(e, np.array([2, 0, 0]),
                [Service((0, 1), 'HSR', frequency, formation)], np.array([2, 0, 0]))
            self.assertAlmostEqual(network.times[0], cache[0, 2] + c['operations']['dwell_hours'])
            self.assertIs(e.running_times['HSR'], cache)
        c['rolling_stock']['HSR']['effective_acceleration_mps2'] = .2
        rebuilt = Evaluator(graph_fixture(), c)
        self.assertGreater(rebuilt.running_times['HSR'][0, 2], cache[0, 2])
        c['rolling_stock']['HSR']['speed_kmh'] = 100
        capped = Evaluator(graph_fixture(), c)
        self.assertAlmostEqual(capped.running_times['HSR'][0, 1],
            running_hours(capped.lengths[0, 1], 100, .2))

    def test_invalid_acceleration(self):
        for value in [0, -1, float('nan'), float('inf'), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_config(overrides={'rolling_stock': {'HSR': {'effective_acceleration_mps2': value}}})
