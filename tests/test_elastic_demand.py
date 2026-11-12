"""Network-wide demand responses, fixed references and capacity accounting."""
from copy import deepcopy
import json
import unittest

import numpy as np

from test_railgen import config
from test_hybrid_services import fixture
from rail_gen.config import load_config
from rail_gen.calibrate import fit
from rail_gen.demand import elastic_demand, rail_share, reference_demand
from rail_gen.evaluator import Evaluator
from rail_gen.services import Service
from rail_gen.analysis import network_analysis


class ElasticDemandTests(unittest.TestCase):
    def test_reference_has_no_logistic_or_fare_distance_penalty(self):
        p = config()['demand']
        market = np.array([100., 100., 100.])
        distance = np.array([30., 400., 1200.])
        hours, trips = reference_demand(market, distance, .5, p)
        np.testing.assert_allclose(trips, 2 * market)
        np.testing.assert_allclose(hours, .5 + distance * p['reference_detour_ratio'] /
                                   p['reference_speed_kmh'] + p['reference_wait_hours'] +
                                   p['reference_transfer_hours'])
        changed = dict(p, t0_hours=0, sigma_hours=100,
                       reference_yield_eur_pkm=10, value_of_time_eur=1)
        other_hours, other_trips = reference_demand(market, distance, .5, changed)
        np.testing.assert_array_equal(other_hours, hours)
        np.testing.assert_array_equal(other_trips, trips)

    def test_cannot_fit_unused_logistic_parameter(self):
        with self.assertRaisesRegex(ValueError, 't0_hours only affects rail_share'):
            fit([], config(), ('t0_hours',))

    def setUp(self):
        self.c = config()
        self.c['demand']['k'] = .001
        self.g = fixture()
        self.e = Evaluator(self.g, self.c)

    def test_example_identity_deterioration_and_disconnection(self):
        q = np.array([1e6, 5e5, 100, 100, 100])
        old = np.array([4, 6, 2, 2, 2])
        new = np.array([3, 5, 2, 3, np.inf])
        total, shift, induced = elastic_demand(q, old, new, self.c['demand'])
        np.testing.assert_allclose(total, q * (new / old) ** -1.4)
        np.testing.assert_allclose(total, q + shift + induced)
        self.assertAlmostEqual(total[0], 1495940.193928266, places=5)
        self.assertEqual(total[2], q[2])
        self.assertEqual(shift[2], 0)
        self.assertEqual(induced[2], 0)
        self.assertLess(shift[3], 0)
        self.assertLess(induced[3], 0)
        self.assertEqual(total[4], 0)
        self.assertEqual(shift[4], -100)
        self.assertEqual(induced[4], 0)

    def test_connecting_line_and_upgrade_benefit_remote_market(self):
        partial = self.e.evaluate([1, 0, 0], services=[Service((0, 1), 'CR', 2)])
        connected = self.e.evaluate([1, 1, 0], services=[Service((0, 1, 2), 'HSR', 2)])
        faster = self.e.evaluate([1, 2, 0], services=[Service((0, 1, 2), 'HSR', 2)])
        j = np.flatnonzero((self.e.origins == 0) & (self.e.destinations == 2))[0]
        self.assertEqual(partial.trips[j], 0)
        self.assertGreater(connected.trips[j], 0)
        self.assertGreater(faster.trips[j], connected.trips[j])
        np.testing.assert_array_equal(partial.reference_trips, faster.reference_trips)
        # Evaluating intervening candidates never rebases demand.
        again = self.e.evaluate([1, 1, 0], services=connected.service_plan)
        np.testing.assert_array_equal(again.trips, connected.trips)

    def test_countries_do_not_change_demand(self):
        plan = [Service((0, 1, 2), 'HSR', 2)]
        domestic = self.e.evaluate([2, 2, 0], services=plan)
        g = deepcopy(self.g)
        for city, country in zip(g.cities, ('FR', 'DE', 'BE')):
            city.country = country
        international = Evaluator(g, self.c).evaluate([2, 2, 0], services=plan)
        np.testing.assert_array_equal(domestic.trips, international.trips)
        np.testing.assert_array_equal(domestic.reference_trips, international.reference_trips)

    def test_frequency_fare_and_legacy_mode(self):
        slow = self.e.evaluate([1, 1, 0], services=[Service((0, 1, 2), 'CR', 1)])
        frequent = self.e.evaluate([1, 1, 0], services=[Service((0, 1, 2), 'CR', 2)])
        self.assertTrue(np.all(frequent.trips > slow.trips))
        c = deepcopy(self.c)
        c['rolling_stock']['CR']['yield_eur_pkm'] *= 2
        expensive = Evaluator(self.g, c).evaluate([1, 1, 0], services=slow.service_plan)
        np.testing.assert_array_equal(expensive.trips, slow.trips)
        self.assertGreater(expensive.metrics['revenue_eur'], slow.metrics['revenue_eur'])
        c['demand']['model'] = 'rail_share'
        e = Evaluator(self.g, c)
        legacy = e.evaluate([1, 1, 0], services=slow.service_plan)
        np.testing.assert_allclose(legacy.trips, 2 * e.base * rail_share(legacy.generalized_time, c['demand']))

    def test_capacity_components_and_json_reconcile(self):
        self.c['demand']['k'] = 1e4
        e = Evaluator(self.g, self.c)
        r = e.evaluate([2, 1, 0], services=[Service((0, 1, 2), 'HSR', 1)])
        m = r.metrics
        self.assertGreater(m['lost_trips'], 0)
        np.testing.assert_allclose(r.reference_trips + r.shift_change + r.induced_change, r.potential_trips)
        self.assertAlmostEqual(m['reference_rail_trips'] + m['modal_shift_change'] + m['induced_demand_change'], m['potential_rail_trips'])
        self.assertAlmostEqual(m['carried_reference_trips'] + m['carried_modal_shift_change'] + m['carried_induced_demand_change'], m['passengers'])
        self.assertAlmostEqual(m['passengers'] + m['lost_trips'], m['potential_rail_trips'])
        self.assertTrue(np.all(r.segment_load <= r.segment_capacity + 1e-7))
        data = network_analysis(self.g, self.c, r, e, reevaluate=False)
        json.dumps(data, allow_nan=False)
        self.assertEqual(len(data['od_demand']), len(e.base))
        self.assertTrue(all(0 <= x['satisfaction'] <= 1 for x in data['corridors']))

    def test_config_validation_and_sensitivities(self):
        for value in (-1, -1.4, -2):
            self.assertEqual(load_config(profile=None, overrides={'demand': {'total_elasticity': value}})['demand']['total_elasticity'], value)
        for values in ({'model': 'unknown'}, {'total_elasticity': -.5},
                       {'shift_elasticity': 0}, {'total_elasticity': True},
                       {'reference_speed_kmh': 0}, {'reference_detour_ratio': .5},
                       {'reference_wait_hours': -1}, {'shift_elasticity': float('nan')}):
            with self.assertRaises(ValueError):
                load_config(profile=None, overrides={'demand': values})
