"""Analytic checks for the cached Pareto generator, independent of legacy solvers."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rail_gen.alignment import Alignment, CACHE_VERSION, CandidateGraph, Edge, cached_graph, cost_surfaces, fingerprint, load_graph, save_graph
from rail_gen.calibrate import backtest, fit, predict_corridors, read_corridors
from rail_gen.config import load_config, merge, validate
from rail_gen.demand import fare, rail_share
from rail_gen.evaluator import Evaluator, capital_recovery_factor
from rail_gen.prep import City, candidate_pairs, distances, station_capture
from rail_gen.search import bridge_components, nondominated, representative_indices, search


def config():
    return load_config(profile=None, overrides={'demand': {'k': .05}, 'search': {'algorithm': 'nsga2', 'population': 12, 'generations': 4,
        'greedy_fraction': .25, 'representatives': 3, 'hypervolume_tolerance': 0}})


def graph_fixture(cities=None, edges=None):
    cities = cities or [City('a', 'A', 2, 46, 2e6, .8, .1),
                         City('b', 'B', 4, 46, 1e6, .8, .1),
                         City('c', 'C', 6, 46, 500000, .8, .1)]
    if edges is None:
        edges = []
        for a, b, tiles in [(0, 1, [0, 1]), (1, 2, [1, 2]), (0, 2, [3, 4])]:
            d = float(distances(cities)[a, b]) * 1.05
            geometry = [[cities[a].lon, cities[a].lat], [cities[b].lon, cities[b].lat]]
            edges.append(Edge(a, b, {'CR': Alignment(d, 1200, tiles, geometry),
                                    'HSR': Alignment(d, 4000, tiles, geometry)}))
    costs = np.array([[600] * 5, [2000] * 5], dtype=float)
    return CandidateGraph.build(cities, edges, costs, dict(cache_version=CACHE_VERSION))


class ConfigTests(unittest.TestCase):
    def test_nested_profiles_and_validation(self):
        c = load_config(profile='smoke', overrides={'search': {'seed': 11}})
        self.assertEqual(c['prep']['max_cities'], 12)
        self.assertEqual(c['search']['seed'], 11)
        with self.assertRaisesRegex(ValueError, 'Unknown parameter'):
            load_config(overrides={'demand': {'typo': 1}})
        for overrides in [{'alignment': {'cell_km': 0}}, {'search': {'workers': 0}},
                          {'search': {'population': 2}}, {'demand': {'k': float('nan')}},
                          {'operations': {'target_load_factor': 1.1}}]:
            with self.assertRaises(ValueError):
                load_config(overrides=overrides)


class PreparationTests(unittest.TestCase):
    def test_catchment_population_is_exclusive_and_capped(self):
        c = config()
        cities = graph_fixture().cities
        xy = np.array([city.xy for city in cities])
        people = np.array([3e6, 200000, 0])
        station_capture(cities, xy, people, c)
        self.assertAlmostEqual(cities[0].kappa, c['prep']['uptake'])
        self.assertAlmostEqual(cities[1].kappa, .2 * c['prep']['uptake'])
        self.assertEqual(cities[2].kappa, 0)
        self.assertEqual(cities[0].access_hours, 0)

    def test_via_pruning_keeps_replacement_links(self):
        c = config()
        cities = graph_fixture().cities
        pairs = candidate_pairs(cities, c)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 2), pairs)
        self.assertNotIn((0, 2), pairs)

    def test_forced_long_corridor_survives_distance_cap(self):
        c = config()
        c['prep']['max_edge_km'] = 1
        c['prep']['force_pairs'] = 1
        cities = graph_fixture().cities[:2]
        self.assertEqual(candidate_pairs(cities, c), [(0, 1)])


class DemandTests(unittest.TestCase):
    def test_smooth_share_and_od_fare(self):
        c = config()['demand']
        self.assertAlmostEqual(float(rail_share(3, c)), .5)
        self.assertGreater(float(rail_share(2, c)), float(rail_share(5, c)))
        self.assertGreater(float(rail_share(8, c)), 0)
        self.assertEqual(float(rail_share(np.inf, c)), 0)
        self.assertAlmostEqual(float(fare(200, .1, c)), 20)


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.g = graph_fixture()
        self.e = Evaluator(self.g, self.c)

    def test_empty_network_and_invalid_genomes(self):
        r = self.e.evaluate([0, 0, 0], details=True)
        self.assertTrue(r.feasible)
        self.assertTrue(np.isfinite(r.objectives).all())
        self.assertEqual(r.metrics['upfront_eur'], 0)
        self.assertEqual(r.metrics['passengers'], 0)
        self.assertEqual(r.metrics['population_served'], 0)
        self.assertEqual(r.metrics['fleet_CR'], 0)
        for x in ([1, 2], [1, 3, 0], [1, .5, 0], [1, np.nan, 0]):
            with self.assertRaises(ValueError):
                self.e.evaluate(x)

    def test_annual_one_way_demand_doubles_for_two_way_reporting(self):
        r = self.e.evaluate([1, 0, 0])
        a, b = self.g.cities[:2]
        distance = self.e.distance[0, 1]
        p = self.c['demand']
        price = self.c['tracks']['CR']['yield_eur_pkm'] * distance
        gt = a.access_hours + b.access_hours + .5 / r.frequency[0] + self.g.edges[0].alignments['CR'].length_km / self.c['tracks']['CR']['speed_kmh'] + self.c['operations']['dwell_hours'] + price / p['value_of_time_eur']
        expected = 2 * p['k'] * (a.population * a.kappa) ** p['a'] * (b.population * b.kappa) ** p['a'] / distance ** p['beta'] * rail_share(gt, p)
        self.assertEqual(r.metrics['spill_fraction'], 0)
        self.assertAlmostEqual(r.metrics['passengers'], expected)
        self.assertAlmostEqual(r.directional_load[0], expected / 2)
        self.assertAlmostEqual(r.metrics['revenue_eur'], expected * price)

    def test_shared_tiles_paid_once_hsr_overrides_cr(self):
        self.assertEqual(self.g.track_capital(np.array([1, 1, 0])), 1800)
        self.assertEqual(self.g.track_capital(np.array([1, 2, 0])), 4600)
        self.assertEqual(self.g.track_capital(np.array([2, 2, 0])), 6000)

    def test_full_financial_allocation_reconciles(self):
        r = self.e.evaluate([1, 2, 0], details=True)
        self.assertAlmostEqual(r.edge_revenue.sum(), r.metrics['revenue_eur'])
        self.assertAlmostEqual(r.edge_net.sum(), r.metrics['net_eur'], delta=.00001)
        self.assertAlmostEqual(r.metrics['annual_cost_eur'], r.metrics['annual_capital_eur'] + r.metrics['operating_cost_eur'])
        self.assertAlmostEqual(r.metrics['upfront_eur'], r.metrics['track_capital_eur'] + r.metrics['station_capital_eur'] + r.metrics['fleet_capital_eur'])
        self.assertEqual(capital_recovery_factor(0, 50), .02)
        self.assertAlmostEqual(capital_recovery_factor(.04, 50), .0465502, places=6)

    def test_fare_revenue_does_not_increase_with_detour(self):
        r = self.e.evaluate([1, 0, 0])
        original_revenue_per_trip = r.metrics['revenue_eur'] / r.metrics['passengers']
        g = deepcopy(self.g)
        g.edges[0].alignments['CR'].length_km *= 1.8
        detour = Evaluator(g, self.c).evaluate([1, 0, 0])
        self.assertAlmostEqual(original_revenue_per_trip, detour.metrics['revenue_eur'] / detour.metrics['passengers'])
        self.assertLess(detour.metrics['passengers'], r.metrics['passengers'])

    def test_spill_obeys_capacity_and_is_constraint(self):
        c = deepcopy(self.c)
        c['demand']['k'] = 10000
        for track in c['tracks'].values():
            track['max_frequency'] = 1
        e = Evaluator(self.g, c)
        r = e.evaluate([1, 1, 0])
        self.assertGreater(r.metrics['spill_fraction'], .15)
        self.assertFalse(r.feasible)
        o = c['operations']
        maximum = c['tracks']['CR']['seats'] * o['maximum_load_factor'] * o['hours_per_day'] * o['days_per_year'] / o['peak_factor']
        self.assertTrue(np.all(r.directional_load <= maximum + 1e-7))
        self.assertTrue(np.all(r.frequency <= 1))

    def test_budget_subsidy_and_zero_load_repair(self):
        self.c['constraints']['upfront_budget_eur'] = 1
        self.assertGreater(self.e.evaluate([1, 0, 0]).constraints[0], 0)
        self.c['constraints']['maximum_subsidy_eur'] = 1
        self.assertGreater(self.e.evaluate([1, 0, 0]).constraints[1], 0)
        self.c['demand']['k'] = 0
        result = Evaluator(self.g, self.c).evaluate([1, 1, 2], repair=True)
        np.testing.assert_array_equal(result.genome, [0, 0, 0])
        self.assertEqual(result.metrics['annual_cost_eur'], 0)

    def test_transfers_only_on_branched_turns(self):
        cities = [City(str(i), str(i), lon, lat, 500000, .8, .1)
                  for i, (lon, lat) in enumerate([(2, 46), (4, 46), (6, 46), (4, 48)])]
        edges = []
        for i, (a, b) in enumerate([(0, 1), (1, 2), (1, 3)]):
            edges.append(Edge(a, b, {'CR': Alignment(200, 600, [i], [[cities[a].lon, cities[a].lat], [cities[b].lon, cities[b].lat]])}))
        g = graph_fixture(cities, edges)
        e = Evaluator(g, self.c)
        _, _, turns, _ = e._paths(np.arange(3), np.array([1, 1, 1]))
        values = {(a, b): value for a, b, value in zip(e.origins, e.destinations, turns)}
        self.assertEqual(values[0, 2], 0)
        self.assertEqual(values[0, 3], 1)
        self.assertEqual(values[2, 3], 1)

    def test_time_objective_empty_network_is_not_fastest(self):
        c = deepcopy(self.c)
        c['search']['include_time_objective'] = True
        e = Evaluator(self.g, c)
        self.assertGreater(e.evaluate([0, 0, 0]).objectives[3], e.evaluate([1, 0, 0]).objectives[3])


class CacheTests(unittest.TestCase):
    def test_parquet_sparse_roundtrip(self):
        g = graph_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            save_graph(g, temporary)
            copy = load_graph(temporary)
            self.assertEqual(asdict(copy.cities[0]), asdict(g.cities[0]))
            self.assertEqual(copy.edges, g.edges)
            np.testing.assert_array_equal(copy.tile_costs, g.tile_costs)
            for a, b in zip(copy.matrices, g.matrices):
                np.testing.assert_array_equal(a.toarray(), b.toarray())
            np.testing.assert_allclose(Evaluator(copy, config()).evaluate([1, 2, 0]).objectives,
                                       Evaluator(g, config()).evaluate([1, 2, 0]).objectives)

    def test_input_content_and_routing_parameters_invalidate_cache(self):
        c = config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input'
            path.write_text('first')
            for field in ('cities_file', 'population_file', 'boundaries_file', 'terrain_file'):
                c[field] = str(path)
            original = fingerprint(c)
            path.write_text('second')
            self.assertNotEqual(original, fingerprint(c))
            original = fingerprint(c)
            c['alignment']['cell_km'] *= 2
            self.assertNotEqual(original, fingerprint(c))

    def test_small_real_raster_pipeline_and_cache_reuse(self):
        c = load_config(profile=None, overrides={'terrain_kind': 'flat', 'terrain_file': '',
            'boundaries_file': '', 'countries': [], 'bbox': [1, 45, 7, 47],
            'prep': {'max_cities': 3, 'force_pairs': 1}, 'alignment': {'cell_km': 10}})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cities = graph_fixture().cities
            city_file = directory / 'cities.json'
            city_file.write_text(json.dumps([asdict(city) for city in cities]))
            population_file = directory / 'population.json'
            population_file.write_text(json.dumps(dict(type='FeatureCollection', features=[
                dict(type='Feature', geometry=dict(type='Point', coordinates=[city.lon, city.lat]),
                     properties=dict(population=city.population, area_km2=10)) for city in cities])))
            c['cities_file'], c['population_file'] = str(city_file), str(population_file)
            graph = cached_graph(c, directory / 'cache', progress=None)
            self.assertGreater(len(graph.edges), 0)
            again = cached_graph(c, directory / 'cache', progress=None)
            self.assertEqual(graph.metadata['fingerprint'], again.metadata['fingerprint'])
            self.assertEqual(graph.edges, again.edges)


class CalibrationTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i, distance in enumerate((100, 200, 300, 400, 500, 600)):
            rows.append(dict(corridor=str(i), population_a=2e6 / (i + 1), population_b=1e6,
                kappa_a=.8, kappa_b=.8, distance_km=distance, line_length_km=distance * 1.1,
                speed_kmh=230, frequency=2, access_a_hours=.1, access_b_hours=.1,
                yield_eur_pkm=.1, split='train' if i < 3 else 'test', source='synthetic test fixture',
                anchor='true', **{'class': 'strong' if i == 0 else 'thin' if i == 5 else ''}))
        parameters = config()['demand'].copy()
        parameters['k'] = 2.75
        for r, observed in zip(rows, predict_corridors(rows, parameters)):
            r['observed_annual_passengers'] = float(observed)
        return rows

    def test_k_fit_uses_only_training_and_holdout_rank_is_independent(self):
        c = config()
        rows = self.rows()
        parameters = fit(rows, c)
        self.assertAlmostEqual(parameters['k'], 2.75, places=5)
        rows[-1]['observed_annual_passengers'] *= 100
        self.assertAlmostEqual(fit(rows, c)['k'], 2.75, places=5)
        report = backtest(rows, c, parameters)
        self.assertEqual(report['checks']['anchor_magnitude']['status'], 'fail')
        self.assertEqual(report['status'], 'failed')

    def test_missing_holdout_is_not_a_pass(self):
        c = config()
        rows = self.rows()[:3]
        report = backtest(rows, c, fit(rows, c))
        self.assertEqual(report['checks']['ranking']['status'], 'not_assessed')
        self.assertNotEqual(report['status'], 'passed')


class SearchTests(unittest.TestCase):
    def test_reproducible_feasible_pareto_front_and_report(self):
        from rail_gen.report import write_report
        c, g = config(), graph_fixture()
        first = search(g, c, progress=None)
        second = search(g, c, progress=None)
        np.testing.assert_array_equal(first.genomes, second.genomes)
        np.testing.assert_allclose(first.objectives, second.objectives)
        self.assertGreater(len(first.genomes), 1)
        self.assertEqual(len(nondominated(first.objectives)), len(first.genomes))
        for x, objectives in zip(first.genomes, first.objectives):
            evaluated = Evaluator(g, c).evaluate(x, repair=True)
            self.assertTrue(evaluated.feasible)
            np.testing.assert_allclose(evaluated.objectives, objectives)
            np.testing.assert_array_equal(x, evaluated.genome)
        with tempfile.TemporaryDirectory() as temporary:
            rows = write_report(g, c, first, temporary)
            self.assertEqual(len(rows), len(first.genomes))
            self.assertTrue((Path(temporary) / 'pareto.png').exists())
            self.assertEqual(json.loads((Path(temporary) / 'backtest.json').read_text())['status'], 'not_calibrated')
            for feature in json.loads(next((Path(temporary) / 'designs').glob('*.geojson')).read_text())['features']:
                self.assertIn(feature['geometry']['type'], ('Point', 'LineString'))

    def test_representatives_preserve_anchors(self):
        objectives = np.array([[0, 0, 0], [20, -100, -30], [100, -150, -50], [10, -40, -10], [30, -90, -40]])
        selected = representative_indices(objectives, 3, 25)
        self.assertIn(0, selected)
        self.assertIn(1, selected)
        self.assertEqual(len(selected), 3)

    def test_parallel_evaluation_and_nsga3(self):
        c = config()
        c['search']['workers'] = 2
        c['search']['algorithm'] = 'nsga3'
        result = search(graph_fixture(), c, progress=None)
        self.assertGreater(len(result.genomes), 0)
        self.assertTrue(np.isfinite(result.objectives).all())

class TerrainSurfaceTests(unittest.TestCase):
    def base(self):
        c = load_config(profile=None, overrides={'terrain_kind': 'flat', 'terrain_file': '',
            'boundaries_file': '', 'countries': [], 'bbox': [2, 45, 6, 47],
            'alignment': {'cell_km': 10}})
        samples = (np.array([City('x', 'x', 4, 46, 1).xy]), np.array([1]), np.array([0]))
        return c, samples

    def raster(self, path, array, transform):
        import rasterio
        array = np.asarray(array, dtype='float32')
        with rasterio.open(path, 'w', driver='GTiff', width=array.shape[1], height=array.shape[0],
                           count=1, dtype='float32', crs='EPSG:3035', transform=transform,
                           nodata=np.nan) as dataset:
            dataset.write(array, 1)

    def test_track_specific_grade_limits_and_missing_dem(self):
        c, samples = self.base()
        flat, _, _, transform = cost_surfaces(c, samples)
        shape = flat.shape[1:]
        elevation = np.broadcast_to(np.arange(shape[1]) * 10000 * .035, shape).copy()
        elevation[2:4, 2:4] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'dem.tif'
            self.raster(path, elevation, transform)
            c['terrain_kind'], c['terrain_file'] = 'dem', str(path)
            c['land_class_file'] = str(Path(temporary) / 'land.tif')
            self.raster(c['land_class_file'], np.zeros(shape), transform)
            rates, _, grade, _ = cost_surfaces(c, samples)
            self.assertTrue(np.isinf(rates[0]).all())
            self.assertTrue(np.isfinite(rates[1, 10, 10]))
            self.assertAlmostEqual(float(grade[10, 10]), .035, places=5)
            self.assertTrue(np.isinf(rates[:, 2, 2]).all())

    def test_sea_is_blocked_inland_water_is_priced(self):
        c, samples = self.base()
        flat, _, _, transform = cost_surfaces(c, samples)
        classes = np.zeros(flat.shape[1:])
        classes[2, 2], classes[3, 3] = 1, 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'classes.tif'
            self.raster(path, classes, transform)
            c['land_class_file'] = str(path)
            rates, water, _, _ = cost_surfaces(c, samples)
            self.assertTrue(water[2, 2])
            self.assertAlmostEqual(rates[0, 2, 2], c['tracks']['CR']['base_meur_km'] + c['alignment']['water_meur_km'])
            self.assertTrue(np.isinf(rates[:, 3, 3]).all())

    def test_blocked_corner_uses_feasible_cardinal_fallback(self):
        from unittest.mock import patch
        from rasterio.transform import from_origin
        from rail_gen.alignment import route
        from rail_gen.prep import TO_WGS84
        transform = from_origin(4000000, 3000000, 1000, 1000)
        cities = []
        for i, (row, col) in enumerate([(1, 1), (2, 2)]):
            lon, lat = TO_WGS84.transform(transform.c + (col + .5) * 1000,
                                          transform.f - (row + .5) * 1000)
            cities.append(City(str(i), str(i), lon, lat, 100000, .8, .1))
        c = config()
        c['alignment']['cell_km'] = 1
        costs = np.ones((2, 5, 5))
        costs[:, 1, 2] = np.inf
        with patch('rail_gen.alignment.cost_surfaces', return_value=(costs, np.zeros((5, 5)), np.zeros((5, 5)), transform)):
            g = route(cities, [(0, 1)], c, None, progress=None)
        self.assertEqual(len(g.edges), 1)
        self.assertEqual(set(g.edges[0].alignments['CR'].tiles), {6, 11, 12})


if __name__ == '__main__':
    unittest.main()
