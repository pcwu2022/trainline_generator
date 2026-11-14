"""Construction pricing and length-based shared infrastructure regressions."""
import unittest
from unittest.mock import patch

import numpy as np
from rasterio.transform import from_origin

from rail_gen.alignment import Alignment, CandidateGraph, Edge, cost_surfaces, segment_cell_lengths
from rail_gen.config import load_config
from rail_gen.prep import City
from rail_gen.water import prepare_crossings
from tests import test_railgen_water


class TrackPricingTests(unittest.TestCase):
    def test_urban_threshold_and_logarithmic_rate_for_both_tracks(self):
        c = load_config(profile=None, overrides={'terrain_kind': 'flat', 'terrain_file': '',
            'boundaries_file': '', 'bbox': [2, 45, 2.1, 45.1], 'alignment': {'padding_km': 0}})
        class Nearby:
            def __init__(self, xy):
                pass
            def query(self, xy):
                return np.zeros(len(xy)), np.zeros(len(xy), dtype=int)
        with patch('rail_gen.alignment.cKDTree', Nearby):
            for density in (0, 5000, 10000, 100000):
                rates, _, _, _ = cost_surfaces(c, (np.array([[0, 0]]), None, np.array([density])))
                for i, base in enumerate((6, 20)):
                    expected = base * (1 + .35 * max(0, np.log10((100 + density) / 5100)))
                    np.testing.assert_allclose(rates[i], expected)

    def test_actual_lengths_include_diagonals_and_partial_cells(self):
        transform = from_origin(0, 3000, 3000, 3000)
        points = np.array([[.3, 2.7], [5.7, -2.7]])
        lengths = segment_cell_lengths(points, transform, 2)
        expected = np.linalg.norm(points[1] - points[0])
        self.assertAlmostEqual(sum(lengths.values()), expected)
        self.assertAlmostEqual(lengths[0], expected / 2)
        self.assertAlmostEqual(lengths[3], expected / 2)

    def test_highest_length_adjusted_charge_is_shared_once(self):
        cities = [City(str(i), str(i), 2 + i, 46, 1e5) for i in range(3)]
        edges = [Edge(0, 1, {'CR': Alignment(1, 6, [0], [], tile_lengths=[1])}),
                 Edge(1, 2, {'CR': Alignment(2, 12, [0], [], tile_lengths=[2])})]
        graph = CandidateGraph.build(cities, edges, [[6], [20]])
        self.assertAlmostEqual(graph.track_capital([1, 1]), 12)
        self.assertAlmostEqual(graph.track_capital([1, 1], track_counts=[4, 2]), 12)
        self.assertAlmostEqual(graph.track_capital([1, 1], track_counts=[2, 4]), 24)
        np.testing.assert_allclose(graph.selected_tiles([1, 1]).toarray(), 1)

    def test_crossing_type_prices_only_actual_water_length(self):
        c, _, surfaces, water, transform = test_railgen_water.CrossingTests().fixture()
        for kind, price in c['alignment']['water_costs_meur_km'].items():
            c['alignment']['water_crossings'][0]['water_type'] = kind
            assets = prepare_crossings(c, surfaces, water, transform)
            self.assertAlmostEqual(assets[0]['water_km'], 3, places=5)
            # Four km portal-centre route: 3 km water plus 1 km land.
            np.testing.assert_allclose(assets[0]['costs'], 3 * price + 6, atol=1e-4)
        surfaces[:, 2, 4] = np.inf
        self.assertTrue(np.isinf(prepare_crossings(c, surfaces, water, transform)[0]['costs']).all())
        with self.assertRaisesRegex(ValueError, 'water_type'):
            load_config(profile=None, overrides={'alignment': {'water_crossings': [
                {'id': 'bad', 'geometry': [[2, 46], [3, 46]], 'water_type': 'unknown'}]}})
