"""Annual railway operating costs, physical tracks, and shared train passages."""
from types import SimpleNamespace
import unittest

import numpy as np

from rail_gen.alignment import Alignment, CandidateGraph, Edge
from rail_gen.config import load_config
from rail_gen.evaluator import Evaluator
from rail_gen.maintenance import length_matrices, maintenance
from rail_gen.prep import City
from rail_gen.services import Service


class OperatingCostTests(unittest.TestCase):
    def fixture(self, shared=False):
        c = load_config(profile=None)
        cities = [City(str(i), str(i), 2 + i * 2, 46, 1e5, .8, .1) for i in range(3)]
        edges = []
        for i, (a, b) in enumerate([(0, 1), (1, 2)]):
            edges.append(Edge(a, b, {t: Alignment(200, rate * 200, [0 if shared else i],
                [[cities[a].lon, cities[a].lat], [cities[b].lon, cities[b].lat]], tile_lengths=[200])
                for t, rate in [('CR', 6), ('HSR', 20)]}))
        graph = CandidateGraph.build(cities, edges, [[6, 6], [20, 20]])
        return c, graph

    def test_worked_example_single_physical_track(self):
        c, graph = self.fixture()
        e = Evaluator(graph, c)
        r = e.evaluate([1, 0], track_counts=[1, 0], services=[Service((0, 1), 'CR', 10 / 16)])
        m = r.metrics
        self.assertEqual(m['train_km_per_year'], 1460000)
        self.assertEqual(m['train_opex_eur'], 28470000)
        self.assertEqual(m['track_fixed_maintenance_eur'], 21000000)
        self.assertEqual(m['track_variable_maintenance_eur'], 2263000)
        self.assertEqual(m['railway_opex_eur'], 51733000)
        self.assertEqual(m['physical_track_km'], 200)
        self.assertEqual(r.services[0]['daily_frequency_per_direction'], 10)
        self.assertAlmostEqual(r.edge_net.sum(), m['net_eur'], delta=1e-6)

    def test_hsr_stock_on_conventional_track_and_track_count(self):
        c, graph = self.fixture()
        for count in (2, 4):
            r = Evaluator(graph, c).evaluate([1, 0], track_counts=[count, 0],
                services=[Service((0, 1), 'HSR', 10 / 16, 2)])
            self.assertEqual(r.metrics['train_opex_eur'], 1460000 * 31)
            self.assertEqual(r.metrics['track_fixed_maintenance_eur'], 200 * count * 105000)
            self.assertEqual(r.metrics['track_variable_maintenance_eur'], 1460000 * 1.55)
            self.assertAlmostEqual(r.edge_net.sum(), r.metrics['net_eur'], delta=1e-6)

    def test_shared_tracks_charge_fixed_once_and_sum_train_movements(self):
        c, graph = self.fixture(shared=True)
        e = Evaluator(graph, c)
        one = e.evaluate([1, 1], services=[Service((0, 1), 'CR', 1)])
        two = e.evaluate([1, 1], services=[Service((0, 1), 'CR', 1), Service((1, 2), 'HSR', 1)])
        self.assertEqual(one.metrics['track_fixed_maintenance_eur'], 400 * 105000)
        self.assertEqual(one.metrics['track_fixed_maintenance_eur'], two.metrics['track_fixed_maintenance_eur'])
        self.assertEqual(two.metrics['track_variable_maintenance_eur'], 2 * one.metrics['track_variable_maintenance_eur'])
        self.assertAlmostEqual(two.edge_net.sum(), two.metrics['net_eur'], delta=1e-6)

    def test_unused_track_still_pays_fixed_cost_and_zero_variable(self):
        c, graph = self.fixture()
        r = Evaluator(graph, c).evaluate([2, 0], services=[])
        self.assertEqual(r.metrics['track_fixed_maintenance_eur'], 400 * 145000)
        self.assertEqual(r.metrics['track_variable_maintenance_eur'], 0)
        self.assertEqual(r.metrics['train_opex_eur'], 0)

    def test_crossing_asset_uses_km_not_asset_weight(self):
        c, graph = self.fixture(shared=True)
        graph.metadata['water_crossings'] = [dict(resource_id=0, length_km=25)]
        for edge in graph.edges:
            for a in edge.alignments.values():
                a.tile_lengths = [1]
        lengths = length_matrices(graph)
        fixed, variable, km = maintenance(lengths, np.array([2, 1]), np.array([2, 2]), c['tracks'],
            SimpleNamespace(edges=[(0, 1)]), [1], 365 * 16)
        self.assertEqual(km, 50)
        self.assertEqual(fixed.sum(), 50 * 145000)
        self.assertEqual(variable.sum(), 2 * 365 * 16 * 25 * 2.9)
