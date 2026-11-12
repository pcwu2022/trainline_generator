"""Infrastructure choices buy capacity independently of train formations."""
import unittest
import numpy as np
from test_railgen import config, graph_fixture
from rail_gen.evaluator import Evaluator
from rail_gen.incremental import neighbours
from rail_gen.config import load_config
from rail_gen.report import geojson


class TrackLayoutTests(unittest.TestCase):
    def test_cost_frequency_and_formation_limits(self):
        c = config()
        c['demand']['k'] = 10000
        e = Evaluator(graph_fixture(), c)
        for state, count, frequency, cost in [(1, 1, 1, .5), (1, 2, 6, 1),
                                             (1, 4, 12, 2), (2, 2, 12, 1), (2, 4, 24, 2)]:
            with self.subTest(state=state):
                result = e.evaluate([state, 0, 0], details=True, track_counts=[count, 0, 0])
                self.assertEqual(result.frequency[0], frequency)
                baseline = e.graph.track_capital(np.array([state, 0, 0]))
                self.assertAlmostEqual(result.metrics['track_capital_eur'], baseline * cost * 1e6)
                self.assertLessEqual(result.train_units[0], c['operations']['maximum_train_units'])
                self.assertEqual(geojson(e.graph, result)['features'][0]['properties']['track_count'], count)
                self.assertAlmostEqual(result.edge_net.sum(), result.metrics['net_eur'], delta=1e-5)

    def test_shared_tiles_pay_for_largest_required_infrastructure(self):
        e = Evaluator(graph_fixture(), config())
        for x, counts, expected in [([1, 1, 0], [1, 2, 0], 1500), ([1, 1, 0], [2, 4, 0], 3000), ([1, 2, 0], [4, 2, 0], 5200)]:
            self.assertAlmostEqual(e.graph.track_capital(np.array(x), track_counts=counts), expected)
            r = e.evaluate(x, details=True, track_counts=counts)
            self.assertAlmostEqual(r.edge_net.sum(), r.metrics['net_eur'], delta=1e-5)

    def test_single_track_restrictions_and_three_state_infrastructure_search(self):
        from rail_gen.services import Service
        e = Evaluator(graph_fixture(), config())
        with self.assertRaises(ValueError):
            e.evaluate([2, 0, 0], track_counts=[1, 0, 0])
        with self.assertRaises(ValueError):
            e.evaluate([1, 0, 0], track_counts=[1, 0, 0], services=[Service((0, 1), 'HSR', 1)])
        current = e.evaluate([1, 0, 0])
        states = {x[0] for x, _ in neighbours(current, e).values()}
        self.assertTrue({0, 2}.issubset(states))
        self.assertTrue(all(0 <= state <= 2 for state in states))

    def test_invalid_layout_config(self):
        for override in [{'single': {'cost_multiplier': 0}},
                         {'single': {'max_frequency': {'CR': 0}}},
                         {'single': {'max_frequency': {'HSR': 1}}}]:
            with self.assertRaises(ValueError):
                load_config(overrides={'track_layouts': override})
