"""Hybrid rolling stock, actual journeys, full build costs and shared resources."""
from copy import deepcopy
import json
import math
import tempfile
from pathlib import Path
import unittest

import numpy as np

from test_railgen import config, graph_fixture
from rail_gen.alignment import Alignment, Edge, CandidateGraph, fingerprint
from rail_gen.config import load_config
from rail_gen.evaluator import Evaluator
from rail_gen.services import Service, ServiceNetwork, generate_plans
from rail_gen.track_layout import migrate_legacy_genome, MODEL_VERSION
from rail_gen.report import geojson
from rail_gen.analysis import network_analysis
from rail_gen.search import search


def fixture(shared=False):
    cities = graph_fixture().cities
    edges = []
    for i, (a, b) in enumerate([(0, 1), (1, 2), (0, 2)]):
        geometry = [[cities[a].lon, cities[a].lat], [cities[b].lon, cities[b].lat]]
        tile = 0 if shared and i < 2 else i
        edges.append(Edge(a, b, {t: Alignment(100 if i < 2 else 200, rate, [tile], geometry)
                                  for t, rate in [('CR', 600), ('HSR', 2000)]}))
    return CandidateGraph.build(cities, edges, np.array([[600] * 3, [2000] * 3]))


class HybridServiceTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.c['demand']['k'] = .01
        self.g = fixture()
        self.e = Evaluator(self.g, self.c)

    def ac(self, e=None):
        e = e or self.e
        return int(np.flatnonzero((e.origins == 0) & (e.destinations == 2))[0])

    def test_hsr_crosses_mixed_infrastructure_without_transfer(self):
        r = self.e.evaluate([2, 1, 0], services=[Service((0, 1, 2), 'HSR', 2)])
        j = self.ac()
        self.assertEqual(r.services[0]['section_speeds_kmh'], [230, 140])
        self.assertEqual(r.transfers[j], 0)
        self.assertAlmostEqual(r.waiting_hours[j], .25)
        self.assertAlmostEqual(r.in_vehicle_hours[j], 100 / 230 + 100 / 140 + 2 * self.c['operations']['dwell_hours'])
        slower = self.e.evaluate([1, 1, 0], services=r.service_plan)
        self.assertGreater(slower.in_vehicle_hours[j], r.in_vehicle_hours[j])
        self.assertEqual(slower.transfers[j], 0)
        # A conventional train cannot exploit the faster track's speed limit.
        cr = self.e.evaluate([2, 1, 0], services=[Service((0, 1, 2), 'CR', 2)])
        self.assertEqual(cr.services[0]['section_speeds_kmh'], [140, 140])
        # No HSR alignment is required on the conventional extension.
        g = deepcopy(self.g)
        del g.edges[1].alignments['HSR']
        extension = Evaluator(g, self.c).evaluate([2, 1, 0], services=r.service_plan)
        np.testing.assert_array_equal(extension.trips, r.trips)
        upgraded = self.e.evaluate([2, 2, 0], services=r.service_plan)
        self.assertGreater(upgraded.trips[j], r.trips[j])
        self.assertGreater(upgraded.metrics['track_capital_eur'], r.metrics['track_capital_eur'])

    def test_hsr_cost_and_complete_cycle_include_conventional_extension(self):
        r = self.e.evaluate([2, 1, 0], services=[Service((0, 1, 2), 'HSR', 2, 2)], details=True)
        o, stock = self.c['operations'], self.c['rolling_stock']['HSR']
        s = r.services[0]
        cycle = 2 * (100 / 230 + 100 / 140 + 2 * o['dwell_hours']) * (1 + o['turnaround_overhead'])
        fleet = math.ceil(math.ceil(cycle * 2) * 2 * (1 + o['spare_ratio']))
        self.assertAlmostEqual(s['cycle_hours'], cycle)
        self.assertEqual(s['fleet_units'], fleet)
        self.assertEqual(s['active_train_units'], s['active_formations'] * 2)
        self.assertEqual(r.metrics['fleet_CR'], 0)
        self.assertEqual(r.metrics['fleet_HSR'], fleet)
        self.assertEqual(r.metrics['fleet_capital_eur'], fleet * stock['train_meur'] * 1e6)
        self.assertAlmostEqual(r.metrics['train_opex_eur'], 2 * 2 * 200 * o['hours_per_day'] * o['days_per_year'] * 1.6 * stock['opex_eur_trainkm'])
        np.testing.assert_array_equal(r.train_units[:2], [2, 2])
        self.assertEqual(r.segment_capacity[0], r.segment_capacity[1])
        self.assertAlmostEqual(r.edge_net.sum(), r.metrics['net_eur'], delta=1e-6)

    def test_coupled_operating_cost_shares_fixed_costs_but_buys_all_sets(self):
        for train_type, rate, purchase_meur in [('CR', 12, 10), ('HSR', 20, 30)]:
            for fraction in (0, .6, 1):
                with self.subTest(train_type=train_type, fraction=fraction):
                    c = deepcopy(self.c)
                    c['operations']['additional_unit_cost_fraction'] = fraction
                    e = Evaluator(self.g, c)
                    single = e.evaluate([2, 1, 0], services=[Service((0, 1, 2), train_type, 2, 1)])
                    coupled = e.evaluate([2, 1, 0], services=[Service((0, 1, 2), train_type, 2, 2)])
                    annual_km = 2 * 2 * 200 * 16 * 360
                    self.assertAlmostEqual(single.metrics['train_opex_eur'], annual_km * rate)
                    self.assertAlmostEqual(coupled.metrics['train_opex_eur'], annual_km * rate * (1 + fraction))
                    self.assertEqual(coupled.metrics['train_km_per_year'], annual_km)
                    self.assertEqual(coupled.metrics['train_unit_km_per_year'], 2 * annual_km)
                    self.assertEqual(coupled.metrics['fleet_capital_eur'],
                                     coupled.services[0]['fleet_units'] * purchase_meur * 1e6)
                    self.assertAlmostEqual(coupled.segment_capacity[0], 2 * single.segment_capacity[0])
                    extra_operations = annual_km * rate * fraction * 1.1
                    self.assertAlmostEqual(coupled.metrics['operating_cost_eur'] -
                                           single.metrics['operating_cost_eur'], extra_operations)
                    self.assertAlmostEqual(coupled.edge_net.sum(), coupled.metrics['net_eur'], delta=1e-6)

    def test_full_new_construction_and_station_costs_survive_no_services(self):
        trunk = self.e.evaluate([2, 0, 0], services=[])
        extended = self.e.evaluate([2, 1, 0], services=[])
        self.assertEqual(extended.metrics['track_capital_eur'] - trunk.metrics['track_capital_eur'], 600e6)
        self.assertGreater(extended.metrics['station_capital_eur'], trunk.metrics['station_capital_eur'])
        self.assertEqual(extended.metrics['passengers'], 0)
        self.assertGreater(extended.objectives[0], trunk.objectives[0])
        self.assertEqual(extended.metrics['fleet_capital_eur'], 0)

    def test_service_changes_charge_transfer_penalty_and_each_wait(self):
        through = self.e.evaluate([2, 1, 0], services=[Service((0, 1, 2), 'HSR', 2)])
        split = self.e.evaluate([2, 1, 0], services=[Service((0, 1), 'HSR', 2), Service((1, 2), 'HSR', 1)])
        j = self.ac()
        self.assertEqual(split.transfers[j], 1)
        self.assertAlmostEqual(split.waiting_hours[j], .25 + .5)
        self.assertAlmostEqual(split.generalized_time[j] - through.generalized_time[j], .5 + self.c['operations']['transfer_hours'])
        self.assertLess(split.trips[j], through.trips[j])

    def test_shared_infrastructure_and_movements_not_double_counted(self):
        one = self.e.evaluate([1, 0, 0], services=[Service((0, 1), 'CR', 1)])
        two = self.e.evaluate([1, 0, 0], services=[Service((0, 1), 'CR', 1), Service((0, 1), 'HSR', 1)])
        self.assertEqual(two.metrics['track_capital_eur'], one.metrics['track_capital_eur'])
        self.assertEqual(two.metrics['station_capital_eur'], one.metrics['station_capital_eur'])
        self.assertEqual(two.frequency[0], 2)
        self.assertEqual(two.directional_load[0], two.trips.sum() / 2)
        self.assertGreater(two.metrics['train_opex_eur'], one.metrics['train_opex_eur'])
        c = deepcopy(self.c)
        c['tracks']['CR']['max_frequency'] = 1
        e = Evaluator(self.g, c)
        overloaded = e.evaluate([1, 0, 0], services=[Service((0, 1), 'CR', 1), Service((0, 1), 'HSR', 1)])
        self.assertFalse(overloaded.feasible)
        self.assertEqual(overloaded.constraints[4], 1)
        # Different abstract edges sharing a physical tile share its movement cap.
        e = Evaluator(fixture(shared=True), c)
        split = e.evaluate([1, 1, 0], services=[Service((0, 1), 'CR', .75), Service((1, 2), 'CR', .75)])
        through = e.evaluate([1, 1, 0], services=[Service((0, 1, 2), 'CR', .75)])
        self.assertFalse(split.feasible)
        self.assertTrue(through.feasible)
        self.assertEqual(through.metrics['track_capital_eur'], 600e6)

    def test_passenger_capacity_is_per_service_section_and_formation(self):
        c = deepcopy(self.c)
        c['demand']['k'] = 10000
        e = Evaluator(self.g, c)
        r = e.evaluate([2, 1, 0], services=[Service((0, 1, 2), 'HSR', 1, 2)])
        self.assertFalse(r.feasible)
        self.assertGreater(r.metrics['spill_fraction'], .15)
        self.assertTrue(np.all(r.segment_load <= r.segment_capacity + 1e-7))
        self.assertTrue(np.all(r.directional_load <= r.edge_capacity + 1e-7))

    def test_empty_disconnected_invalid_and_financially_infeasible(self):
        empty = self.e.evaluate([0, 0, 0])
        self.assertEqual(empty.services, [])
        self.assertTrue(empty.feasible)
        self.assertTrue(np.isfinite(empty.objectives).all())
        disconnected = self.e.evaluate([1, 0, 0])
        self.assertEqual(disconnected.trips[self.ac()], 0)
        for plan in [[Service((0, 2), 'HSR', 1)], [Service((0, 1, 0), 'CR', 1)],
                     [Service((0, 1), 'CR', 0)], [Service((0, 1), 'CR', 1, 3)]]:
            with self.assertRaises(ValueError):
                self.e.evaluate([1, 0, 0], services=plan)
        c = deepcopy(self.c)
        c['constraints']['upfront_budget_eur'] = 1
        self.assertFalse(Evaluator(self.g, c).evaluate([2, 1, 0]).feasible)

    def test_disconnected_active_components_do_not_create_passenger_journeys(self):
        cities = deepcopy(self.g.cities)
        fourth = deepcopy(cities[2])
        fourth.id, fourth.name, fourth.lon = 'fourth', 'Fourth', 6
        cities.append(fourth)
        edges = [deepcopy(self.g.edges[0]), Edge(2, 3, deepcopy(self.g.edges[1].alignments))]
        g = CandidateGraph.build(cities, edges, self.g.tile_costs)
        c = deepcopy(self.c)
        c['constraints']['minimum_component_share'] = .8
        e = Evaluator(g, c)
        r = e.evaluate([2, 1])
        self.assertEqual(r.metrics['active_components'], 2)
        crossing = (e.origins < 2) & (e.destinations >= 2)
        self.assertTrue(crossing.any())
        self.assertTrue(np.all(r.trips[crossing] == 0))
        self.assertFalse(r.feasible)
        self.assertGreater(r.constraints[3], 0)

    def test_generation_is_bounded_reproducible_and_adapts_to_infrastructure(self):
        x = np.array([2, 1, 0])
        plans = generate_plans(self.e, x, np.array([2, 2, 0]))
        self.assertEqual(plans, generate_plans(self.e, x, np.array([2, 2, 0])))
        self.assertLessEqual(len(plans), self.c['services']['plan_trials'])
        self.assertTrue(any(s.route == (0, 1, 2) and s.train_type == 'HSR' for plan in plans for s in plan))
        partial = generate_plans(self.e, np.array([2, 0, 0]), np.array([2, 0, 0]))
        self.assertTrue(all(2 not in s.route for p in partial for s in p))
        a, b = self.e.evaluate(x), self.e.evaluate(x)
        self.assertEqual(a.services, b.services)
        np.testing.assert_array_equal(a.objectives, b.objectives)
        # A construction change re-evaluates the full passenger/service system.
        conventional = self.e.evaluate([1, 1, 0])
        self.assertNotEqual(a.metrics['track_capital_eur'], conventional.metrics['track_capital_eur'])
        self.assertNotEqual(a.objectives[0], conventional.objectives[0])
        # The economic selector may choose CR stock on either network; the
        # mixed through candidate remains available for full-network comparison.
        self.c['services']['plan_trials'] = 1
        hybrid = Evaluator(self.g, self.c).evaluate(x)
        self.assertEqual(hybrid.services[0]['train_type'], 'HSR')
        self.assertEqual(hybrid.services[0]['route'], [0, 1, 2])

    def test_serialization_and_analysis_preserve_explicit_plan(self):
        r = self.e.evaluate([2, 1, 0], services=[Service((0, 1, 2), 'HSR', 2, 2)])
        encoded = json.loads(json.dumps(geojson(self.g, r), allow_nan=False))
        restored = self.e.evaluate(r.genome, services=encoded['services'], track_counts=r.track_counts)
        self.assertEqual(restored.services, r.services)
        np.testing.assert_array_equal(restored.objectives, r.objectives)
        self.assertEqual(encoded['model_version'], MODEL_VERSION)
        self.assertEqual(encoded['features'][1]['properties']['infrastructure_type'], 'CR')
        self.assertEqual(encoded['features'][1]['properties']['service_types'], ['HSR'])
        analysis = network_analysis(self.g, self.c, r)
        self.assertEqual(analysis['services'], r.services)

    def test_migration_and_config_validation(self):
        legacy = {'tracks': {'CR': {'speed_kmh': 120, 'seats': 300, 'train_meur': 15}},
                  'rolling_stock': {'CR': {'seats': 320}}}
        c = load_config(profile=None, overrides=legacy)
        self.assertEqual(c['tracks']['CR']['speed_kmh'], 120)
        self.assertEqual(c['rolling_stock']['CR']['speed_kmh'], 120)
        self.assertEqual(c['rolling_stock']['CR']['seats'], 320)
        self.assertNotIn('seats', c['tracks']['CR'])
        self.assertEqual(c['operations']['additional_unit_cost_fraction'], .6)
        x, counts = migrate_legacy_genome([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(x, [0, 1, 2, 1, 1, 2])
        np.testing.assert_array_equal(counts, [0, 2, 2, 1, 4, 4])
        with self.assertRaisesRegex(ValueError, 'migrate_legacy_genome'):
            self.e.evaluate([3, 0, 0])
        for overrides in [{'rolling_stock': {'HSR': {'seats': 0}}},
                          *[{'operations': {'additional_unit_cost_fraction': value}}
                            for value in (-.1, 1.1, True, float('nan'))],
                          {'services': {'plan_trials': 4}}, {'services': {'max_route_edges': 0}},
                          {'services': {'max_through_routes': 1.5}}]:
            with self.assertRaises(ValueError):
                load_config(overrides=overrides)

    def test_geometry_cache_is_independent_of_rolling_stock_and_services(self):
        c = deepcopy(self.c)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'
            path.write_text('geometry input')
            for key in c:
                if key.endswith('_file'):
                    c[key] = str(path)
            original = fingerprint(c)
            c['rolling_stock']['HSR']['seats'] *= 2
            c['services']['max_through_routes'] += 1
            self.assertEqual(fingerprint(c), original)
            c['tracks']['CR']['base_meur_km'] *= 2
            self.assertNotEqual(fingerprint(c), original)

    def test_graph_with_no_candidate_edges(self):
        g = CandidateGraph.build(self.g.cities, [], np.zeros((2, 0)))
        r = Evaluator(g, self.c).evaluate([], repair=True)
        self.assertTrue(r.feasible)
        self.assertEqual(r.services, [])
        self.assertTrue(np.isfinite(r.objectives).all())
        self.assertEqual(r.metrics['upfront_eur'], 0)

    def test_all_search_paths_are_reproducible(self):
        for algorithm in ('incremental', 'nsga2', 'nsga3'):
            c = deepcopy(self.c)
            c['search'].update(algorithm=algorithm, generations=3, workers=2)
            a, b = search(self.g, c, progress=None), search(self.g, c, progress=None)
            np.testing.assert_array_equal(a.genomes, b.genomes)
            np.testing.assert_array_equal(a.objectives, b.objectives)
            self.assertGreater(len(a.genomes), 0)
            for x in a.genomes:
                self.assertTrue(Evaluator(self.g, c).evaluate(x, repair=True).feasible)
