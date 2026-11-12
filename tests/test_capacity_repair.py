"""Capacity packages must buy real capacity and retain hard constraints."""
from copy import deepcopy
import unittest

import numpy as np

from test_railgen import config, graph_fixture
from rail_gen.config import load_config
from rail_gen.evaluator import Evaluator
from rail_gen.incremental import buildable, capacity_packages, integration_choice, neighbours, topology
from rail_gen.search import search
from rail_gen.prep import City
from rail_gen.report import geojson


class CapacityRepairTests(unittest.TestCase):
    def test_sparse_services_include_real_headway_wait_and_cost(self):
        for k, frequency in [(.001, 1 / 3), (.012, .5), (.015, 2 / 3)]:
            with self.subTest(frequency=frequency):
                c = config()
                c['demand']['k'] = k
                e = Evaluator(graph_fixture(), c)
                r = e.evaluate([1, 0, 0])
                self.assertEqual(r.frequency[0], frequency)
                a, b = e.graph.cities[:2]
                distance = e.distance[0, 1]
                expected_time = (a.access_hours + b.access_hours + .5 / frequency +
                    e.lengths[0, 1] / c['tracks']['CR']['speed_kmh'] +
                    c['operations']['dwell_hours'] +
                    distance * c['rolling_stock']['CR']['yield_eur_pkm'] / c['demand']['value_of_time_eur'])
                self.assertAlmostEqual(r.generalized_time[0], expected_time)
                hours = c['operations']['hours_per_day'] * c['operations']['days_per_year']
                self.assertAlmostEqual(r.metrics['train_km_per_year'],
                                       2 * frequency * e.lengths[0, 1] * hours)

    def test_paid_formations_repair_spill_without_increasing_frequency_caps(self):
        c = config()
        c['demand']['k'] = .2
        c['operations']['maximum_train_units'] = 1
        graph = graph_fixture()
        # Independent sections: this test isolates formation capacity.
        graph.edges[1].alignments['CR'].tiles = [2]
        graph.edges[1].alignments['HSR'].tiles = [2]
        graph = type(graph).build(graph.cities, graph.edges, graph.tile_costs, graph.metadata)
        c['services']['plan_trials'] = 1
        single = Evaluator(graph, c).evaluate([1, 1, 0], details=True)
        c['operations']['maximum_train_units'] = 2
        double = Evaluator(graph, c).evaluate([1, 1, 0], details=True)
        self.assertFalse(single.feasible)
        self.assertTrue(double.feasible)
        self.assertGreater(single.metrics['spill_fraction'], .15)
        self.assertEqual(double.metrics['spill_fraction'], 0)
        self.assertTrue(np.all(double.frequency <= c['tracks']['CR']['max_frequency']))
        self.assertTrue(np.all(double.frequency <= c['tracks']['CR']['max_frequency']))
        self.assertTrue(np.all(double.train_units <= 2))
        self.assertEqual(double.train_units[2], 0)
        self.assertGreater(double.metrics['fleet_capital_eur'], single.metrics['fleet_capital_eur'])
        self.assertGreater(double.metrics['train_opex_eur'], single.metrics['train_opex_eur'])
        self.assertEqual(double.metrics['track_capital_eur'], single.metrics['track_capital_eur'])
        train_km = 2 * double.frequency * Evaluator(graph, c).lengths[np.arange(3), double.genome] * c['operations']['hours_per_day'] * c['operations']['days_per_year']
        formation_cost = np.where(double.train_units > 0,
            1 + (double.train_units - 1) * c['operations']['additional_unit_cost_fraction'], 0)
        expected_opex = (train_km * formation_cost).sum() * c['rolling_stock']['CR']['opex_eur_trainkm']
        self.assertAlmostEqual(double.metrics['train_opex_eur'], expected_opex)
        lines = [f for f in geojson(graph, double)['features'] if f['geometry']['type'] == 'LineString']
        self.assertEqual([f['properties']['train_units'] for f in lines], double.train_units[:2].tolist())

    def test_two_connection_package_crosses_infeasible_intermediate_networks(self):
        c = config()
        c['demand']['k'] = .4
        c['operations']['maximum_train_units'] = 1
        # Isolate alternative-route repair when widening cannot add capacity.
        c['track_layouts']['four']['max_frequency'] = c['track_layouts']['double']['max_frequency'].copy()
        c['search'].update(capacity_repair_steps=3, capacity_repair_candidates=8)
        c['services'].update(max_through_routes=0, plan_trials=1)
        cities = [City('a', 'A', 2, 46, 1e6, .8, .1),
                  City('b', 'B', 4, 47, 1e6, .8, .1),
                  City('c', 'C', 6, 46, 1e6, .8, .1)]
        graph = graph_fixture(cities=cities)
        for edge in graph.edges:
            del edge.alignments['HSR']
        graph.edges[1].alignments['CR'].tiles = [2]
        graph = type(graph).build(graph.cities, graph.edges, graph.tile_costs, graph.metadata)
        e = Evaluator(graph, c)
        current = e.evaluate([0, 0, 1])
        self.assertTrue(current.feasible)
        self.assertEqual(topology(current.genome, e), (1, 0))
        ordinary = [e.evaluate(x, repair=True) for x, _ in neighbours(current, e).values()]
        self.assertIsNone(integration_choice(current, [r for r in ordinary if r.feasible], e))
        packages, diagnostics = capacity_packages(current, e)
        selected = integration_choice(current, packages, e)
        self.assertIsNotNone(selected)
        # Both connecting edges are required; demand may make the original
        # direct edge redundant when the package is repaired.
        np.testing.assert_array_equal(selected.genome[:2], [1, 1])
        self.assertEqual(topology(selected.genome, e), (0, 0))
        self.assertTrue(selected.feasible)
        self.assertLessEqual(selected.metrics['spill_fraction'], .15)
        self.assertGreater(selected.metrics['upfront_eur'], current.metrics['upfront_eur'])
        self.assertGreater(diagnostics['evaluations'], 0)
        # Neither money nor unavailable infrastructure can be invented by repair.
        constrained = deepcopy(c)
        constrained['constraints']['upfront_budget_eur'] = current.metrics['upfront_eur']
        packages, _ = capacity_packages(current, Evaluator(graph, constrained))
        self.assertEqual(packages, [])
        packages, diagnostics = capacity_packages(current, e, should_stop=lambda: True)
        self.assertEqual(packages, [])
        self.assertEqual(diagnostics['evaluations'], 0)

    def test_default_capacity_policy_buys_capacity_for_high_demand(self):
        c = load_config(profile=None)
        c['demand'].update(a=.9, beta=1.6)
        self.assertEqual(c['operations']['maximum_train_units'], 2)
        self.assertGreater(c['search']['capacity_repair_steps'], 0)
        c['demand']['k'] = .2
        c['services']['plan_trials'] = 1
        expanded = Evaluator(graph_fixture(), c).evaluate([1, 1, 0])
        c['operations']['maximum_train_units'] = 1
        limited = Evaluator(graph_fixture(), c).evaluate([1, 1, 0])
        self.assertTrue(expanded.feasible)
        self.assertFalse(limited.feasible)
        self.assertGreater(expanded.metrics['passengers'], limited.metrics['passengers'])
        self.assertGreater(expanded.metrics['train_opex_eur'], limited.metrics['train_opex_eur'])

    def test_capacity_settings_are_validated(self):
        for overrides in [
            {'operations': {'maximum_train_units': 0}},
            {'operations': {'maximum_train_units': 1.5}},
            {'search': {'capacity_repair_steps': -1}},
            {'search': {'capacity_repair_candidates': 0}},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(overrides=overrides)

    def test_spill_buys_tracks_and_keeps_unmet_demand_visible(self):
        c = config()
        c['demand']['k'] = 1e4
        c['services']['plan_trials'] = 1
        e = Evaluator(graph_fixture(), c)
        fixed = e.evaluate([2, 0, 0], track_counts=[2, 0, 0])
        expanded = e.evaluate([2, 0, 0], repair=True)
        self.assertEqual(expanded.track_counts[0], 4)
        self.assertGreater(expanded.metrics['track_capital_eur'], fixed.metrics['track_capital_eur'])
        self.assertGreater(expanded.metrics['passengers'], fixed.metrics['passengers'])
        self.assertGreater(expanded.metrics['potential_rail_trips'], expanded.metrics['passengers'])
        self.assertFalse(expanded.feasible)
        self.assertTrue(buildable(expanded))
        constrained = deepcopy(expanded)
        constrained.constraints[0] = 1
        self.assertFalse(buildable(constrained))
        replay = e.evaluate(expanded.genome, services=expanded.services,
                            track_counts=expanded.track_counts)
        self.assertAlmostEqual(replay.metrics['passengers'], expanded.metrics['passengers'])

    def test_extreme_demand_search_grows_instead_of_retaining_empty_network(self):
        c = config()
        c['demand'].update(k=10, a=1.2)
        c['search'].update(algorithm='incremental', generations=5, population=12, profit=True)
        updates = []
        search(graph_fixture(), c, progress=None, observer=updates.append)
        built = [u.result for u in updates if u.result.metrics['active_edges'] > 0]
        self.assertTrue(built)
        self.assertTrue(any(np.any(r.track_counts == 4) for r in built))
        self.assertTrue(all(buildable(r) for r in built))
        self.assertTrue(any(r.metrics['lost_trips'] > 0 for r in built))


if __name__ == '__main__':
    unittest.main()
