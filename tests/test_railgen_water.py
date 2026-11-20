"""Water assets exercise routing, accounting and cache persistence together."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rasterio.transform import from_origin

from rail_gen.alignment import route, save_graph, load_graph, cost_surfaces, fingerprint
from rail_gen.config import load_config
from rail_gen.prep import City
from rail_gen.projection import regional_projection, project_cities
from rail_gen.water import crossing_cost


class CrossingTests(unittest.TestCase):
    def fixture(self):
        projection = regional_projection(load_config(profile=None))
        transform = from_origin(-4000, 3000, 1000, 1000)
        def coordinate(r, c):
            return list(projection.to_wgs84.transform(transform.c + (c + .5) * 1000,
                                           transform.f - (r + .5) * 1000))
        cities = [City(str(i), str(i), *coordinate(r, c), 1e5, .8, .1)
                  for i, (r, c) in enumerate([(2, 1), (2, 7), (3, 7)])]
        c = load_config(profile=None, overrides={'alignment': {
            'cell_km': 1, 'sea_crossings_enabled': True,
            'water_crossings': [{'id': 'test', 'geometry': [coordinate(2, 2), coordinate(2, 6)]}]}})
        water = np.zeros((6, 9), dtype=bool)
        water[:, 3:6] = True
        surfaces = np.full((2, 6, 9), 6.)
        surfaces[:, water] = 120.
        project_cities(cities, c)
        return c, cities, surfaces, water, transform

    def test_station_relocation_capture_and_shared_endpoints(self):
        c, cities, surfaces, water, transform = self.fixture()
        projection = regional_projection(c)
        original = cities[0].xy.copy()
        cities[0].lon, cities[0].lat = projection.to_wgs84.transform(
            *(original * 1000 + np.array([300., 200.])))
        city_center = cities[0].xy.copy()
        samples = (np.array([original, cities[1].xy, cities[2].xy]),
                   np.array([1000., 2000., 3000.]), np.array([100., 100., 100.]))
        with patch('rail_gen.alignment.cost_surfaces', return_value=(surfaces, water, np.zeros_like(water), transform)):
            graph = route(cities, [(0, 1), (0, 2)], c, samples, progress=None)
        np.testing.assert_allclose(cities[0].xy, city_center)
        np.testing.assert_allclose(cities[0].station_xy, original)
        self.assertAlmostEqual(cities[0].access_hours, 0., places=7)
        self.assertAlmostEqual(cities[0].kappa, 1000 / cities[0].population * c['prep']['uptake'])
        for edge in graph.edges:
            for alignment in edge.alignments.values():
                for coordinate, index in [(alignment.geometry[0], edge.a),
                                          (alignment.geometry[-1], edge.b)]:
                    np.testing.assert_allclose(np.array(projection.to_metric.transform(*coordinate)) / 1000,
                                               cities[index].station_xy)
        with tempfile.TemporaryDirectory() as folder:
            save_graph(graph, folder)
            loaded = load_graph(folder)
            np.testing.assert_allclose(loaded.cities[0].station_xy, original)

    def test_linear_water_cost_and_subdivision(self):
        value = crossing_cost([10, 20], [120, 120], [True, True], 25)
        self.assertAlmostEqual(value, 30 * 120)
        self.assertAlmostEqual(value, crossing_cost([30], [120], [True], 25))
        self.assertAlmostEqual(crossing_cost([10, 1, 20], [120, 6, 120], [True, False, True], 25),
            10 * 120 + 6 + 20 * 120)

    def test_route_shared_asset_and_cache(self):
        c, cities, surfaces, water, transform = self.fixture()
        with patch('rail_gen.alignment.cost_surfaces', return_value=(surfaces, water, np.zeros_like(water), transform)):
            graph = route(cities, [(0, 1), (0, 2)], c, None, progress=None)
        for e in graph.edges:
            a = e.alignments['HSR']
            self.assertEqual(a.crossing_ids, ('test',))
            self.assertAlmostEqual(a.water_km, 3, places=5)
            self.assertGreater(a.capex_meur, 360)
        shared = set(graph.edges[0].alignments['HSR'].tiles) & set(graph.edges[1].alignments['HSR'].tiles)
        self.assertIn(6 * 9, shared)
        total = graph.track_capital([2, 2])
        self.assertLess(total, sum(e.alignments['HSR'].capex_meur for e in graph.edges))
        from rail_gen.evaluator import Evaluator
        evaluator = Evaluator(graph, c)
        infra = evaluator.infrastructure(np.array([2, 2]), np.array([2, 2]))
        from rail_gen.services import Service, ServiceNetwork
        network = ServiceNetwork(evaluator, np.array([2, 2]),
            [Service((0, 1), 'HSR', 6), Service((0, 2), 'HSR', 6)], np.array([2, 2]), infra)
        usage = network.resource_services.toarray()
        shared_rows = np.all(usage > 0, axis=1)
        self.assertFalse(shared_rows.any())
        np.testing.assert_allclose(network.fit_frequencies(np.array([12., 12.])), [12., 12.])
        np.testing.assert_allclose(network.fit_frequencies(np.array([24., 24.])), [12., 12.])
        # Services on the same edge still share its finite movement capacity.
        same_edge = ServiceNetwork(evaluator, np.array([2, 2]),
            [Service((0, 1), 'HSR', 6), Service((1, 0), 'HSR', 6)], np.array([2, 2]), infra)
        np.testing.assert_allclose(same_edge.fit_frequencies(np.array([12., 12.])), [6., 6.])
        # Crossing and geographic tile overlap continue to share construction cost.
        selected = graph.selected_tiles([2, 2]).toarray()
        self.assertTrue(np.any(np.all(selected > 0, axis=0)))
        self.assertAlmostEqual(infra.edge_track.sum(), total * 1e6)
        with tempfile.TemporaryDirectory() as folder:
            save_graph(graph, folder)
            loaded = load_graph(folder)
            self.assertAlmostEqual(loaded.track_capital([2, 2]), total)
            self.assertEqual(list(loaded.edges[0].alignments['HSR'].crossing_ids), ['test'])

    def test_exclusion_invalidates_link_and_disabling_blocks_sea(self):
        c, cities, surfaces, water, transform = self.fixture()
        for disabled in (False, True):
            modified = surfaces.copy()
            if disabled:
                c['alignment']['sea_crossings_enabled'] = False
                modified[:, water] = np.inf
            else:
                modified[:, 2, 4] = np.inf
            with patch('rail_gen.alignment.cost_surfaces', return_value=(modified, water, np.zeros_like(water), transform)):
                with self.assertRaisesRegex(ValueError, 'No feasible'):
                    route(cities, [(0, 1)], c, None, progress=None)

    def test_configuration_validation(self):
        for overrides in [{'water_growth_km': 0}, {'water_sample_km': 0},
                          {'water_crossings': [{'id': 'x', 'geometry': [[0, 0], [0, 0]]}]},
                          {'water_crossings': [{'id': 'x', 'geometry': [[0, 0], [200, 0]]}]}]:
            with self.assertRaises(ValueError):
                load_config(profile=None, overrides={'alignment': overrides})

    def test_missing_dem_at_sea_does_not_block_crossing(self):
        import rasterio
        c = load_config(profile=None, overrides={'terrain_kind': 'flat', 'terrain_file': '',
            'boundaries_file': '', 'countries': [], 'bbox': [2, 45, 6, 47],
            'alignment': {'cell_km': 10, 'sea_crossings_enabled': True}})
        samples = (np.array([City('x', 'x', 4, 46, 1, metric_crs=regional_projection(c).crs.to_string()).xy]), np.array([1]), np.array([0]))
        surfaces, _, _, transform = cost_surfaces(c, samples)
        shape = surfaces.shape[1:]
        with tempfile.TemporaryDirectory() as folder:
            def write(name, values):
                path = str(Path(folder) / name)
                with rasterio.open(path, 'w', driver='GTiff', width=shape[1], height=shape[0],
                                   count=1, dtype='float32', crs=regional_projection(c).crs, transform=transform,
                                   nodata=np.nan) as dst:
                    dst.write(values.astype('float32'), 1)
                return path
            classes = np.zeros(shape); classes[:, 4:7] = 2; classes[2, 5] = 3
            dem = np.zeros(shape); dem[:, 4:7] = np.nan; dem[2, 2] = np.nan
            c['land_class_file'] = write('classes.tif', classes)
            c['terrain_kind'] = 'dem'; c['terrain_file'] = write('dem.tif', dem)
            rates, water, _, _ = cost_surfaces(c, samples)
            self.assertTrue(np.isfinite(rates[:, 10, 5]).all())
            self.assertTrue(np.isinf(rates[:, 2, 5]).all())
            self.assertTrue(np.isinf(rates[:, 2, 2]).all())
            self.assertTrue(water[10, 5])

    def test_crossing_parameters_invalidate_cache(self):
        c, _, _, _, _ = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input'
            path.write_text('fixture')
            for key in c:
                if key.endswith('_file'):
                    c[key] = str(path)
            for key, value in [('sea_crossings_enabled', False), ('water_growth_km', 30),
                               ('water_sample_km', .1)]:
                original = fingerprint(c)
                c['alignment'][key] = value
                self.assertNotEqual(original, fingerprint(c))
            original = fingerprint(c)
            c['alignment']['water_crossings'][0]['geometry'][0][0] += .01
            self.assertNotEqual(original, fingerprint(c))
