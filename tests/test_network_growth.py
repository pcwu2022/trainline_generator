import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_heuristic import problem
from train_optimizer.annealing import RoutePool, canonical, evaluate, network_coverage, solve_annealing
from train_optimizer.network_growth import (connected_designs, connectivity, growth_energy,
    integration_pressure, ready_to_consolidate, regional_proposal)
from train_optimizer.outputs import LiveState


def two_regions():
    p = problem()
    p.cities = [dict(id=i, name=str(i), population=100) for i in range(4)]
    p.locations = [dict(p.locations[0], id=i, city=i, coordinate=(i*.5, 51)) for i in range(4)]
    p.stations = [dict(p.stations[0], id=i, city=i, location=i, covered_population=100) for i in range(4)]
    base = p.edges[0]
    p.edges = [dict(base, id=i, a=a, b=b, km=40, track_capital=.4e9)
               for i, (a, b) in enumerate([(0, 1), (2, 3), (1, 2)])]
    p.demand = [dict(id=i, origin=a, destination=b, volume=100, km=km)
                for i, (a, b, km) in enumerate([(0, 1, 40), (2, 3, 40), (0, 3, 120)])]
    local = canonical([((0, 1), (0,), 1), ((2, 3), (1,), 1)])
    return p, local


class NetworkGrowthTests(unittest.TestCase):
    def test_exploration_can_seed_a_second_region_despite_final_connectivity(self):
        p, local = two_regions()
        pool = RoutePool(p)
        with patch('train_optimizer.annealing.Router') as router:
            proposal, action = regional_proposal(p, pool, local[:1], np.random.default_rng(42), [1], 'exploration')
            router.assert_not_called()
        self.assertEqual(action, 'seed_region')
        record = evaluate(p, pool, proposal, .1, lambda: 5, threading.Event())
        self.assertIsNotNone(record)
        self.assertEqual(connectivity(record[:2])['component_count'], 2)
        self.assertFalse(record[0].problem.config.require_connected_network)
        self.assertTrue(p.config.require_connected_network)
        self.assertIsNone(evaluate(p, pool, proposal, 1, lambda: 5, threading.Event(), final=True))

    def test_bridge_unlocks_onward_passengers_and_counts_completed_trips_once(self):
        p, local = two_regions()
        pool = RoutePool(p)
        before = evaluate(p, pool, local, .1, lambda: 5, threading.Event())
        # Endpoints 1–2 have no direct OD demand, but connecting the regions
        # unlocks the 0–3 market through three train rides.
        rng = Mock()
        rng.random.return_value = 0
        rng.choice.return_value = 2  # The short 1–2 bridge, with no endpoint OD.
        proposal, action = regional_proposal(p, pool, local, rng, [1], 'integration')
        self.assertEqual(action, 'bridge')
        after = evaluate(p, pool, proposal, 1, lambda: 5, threading.Event(), final=True)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(sum(before[1][i] for i in before[0].q.values()), 200)
        self.assertAlmostEqual(sum(after[1][i] for i in after[0].q.values()), 300)
        self.assertEqual(connectivity(after[:2])['component_count'], 1)
        self.assertTrue(ready_to_consolidate(after[:2], 1, p.config))

    def test_high_coverage_disconnected_network_cannot_skip_integration(self):
        p, local = two_regions()
        record = evaluate(p, RoutePool(p), local, .1, lambda: 5, threading.Event())[:2]
        self.assertEqual(network_coverage(record, p.config, 300), 1)
        self.assertFalse(ready_to_consolidate(record, 1, p.config))
        stats = connectivity(record)
        self.assertAlmostEqual(stats['largest_component_population_share'], .5)

    def test_existing_services_are_excluded_from_regional_additions(self):
        p, local = two_regions()
        connected = canonical(list(local)+[((1, 2), (2,), 1)])
        pool = RoutePool(p)
        proposal, action = regional_proposal(p, pool, connected, np.random.default_rng(0), [1], 'integration')
        self.assertEqual(proposal, connected)
        self.assertEqual(action, 'no_regional_candidate')

    def test_population_share_is_weighted_and_optional_components_remain_configurable(self):
        p, local = two_regions()
        for station, population in zip(p.stations, [900, 100, 1, 1]):
            station['covered_population'] = population
        record = evaluate(p, RoutePool(p), local, .1, lambda: 5, threading.Event())[:2]
        self.assertAlmostEqual(connectivity(record)['largest_component_population_share'], 1000/1002)
        self.assertFalse(ready_to_consolidate(record, 1, p.config))
        p.config.require_connected_network = False
        self.assertTrue(ready_to_consolidate(record, 1, p.config))

    def test_growth_reward_includes_coverage_even_with_no_minimum_market_requirement(self):
        p, local = two_regions()
        pool = RoutePool(p)
        smaller = evaluate(p, pool, local[:1], .1, lambda: 5, threading.Event())[:2]
        larger = evaluate(p, pool, local, .1, lambda: 5, threading.Event())[:2]
        self.assertEqual(p.config.minimum_city_served_fraction, 0)
        small_energy = growth_energy((0, 0, 0), smaller, .1, 0, p.config, .5, 300)
        large_energy = growth_energy((0, 0, 0), larger, .1, 0, p.config, 1, 300)
        self.assertLess(large_energy, small_energy)
        self.assertEqual(integration_pressure(.1, p.config), 0)
        self.assertEqual(integration_pressure(.75, p.config), 1)

    def test_solver_integrates_before_cost_cutting(self):
        p, local = two_regions()
        connected = canonical(list(local)+[((1, 2), (2,), 1)])
        p.config.annealing_iterations = 20
        state = LiveState()
        with patch('train_optimizer.annealing.propose', side_effect=[(local, 'regional_test'), (connected, 'bridge')]) as propose:
            status = solve_annealing(p, state, 5, threading.Event())
        self.assertEqual(status, 'Annealing feasible')
        self.assertEqual(propose.call_args_list[1].kwargs['stage'], 'integration')
        self.assertEqual(state.snapshot()[-1]['stage'], 'consolidation')
        self.assertTrue(state.snapshot()[1].problem.config.require_connected_network)

    def test_regional_seeding_prefers_geographic_novelty(self):
        p, local = two_regions()
        for i, lon in ((4, 6), (5, 6.1)):
            p.cities.append(dict(id=i, name=str(i), population=100))
            p.locations.append(dict(p.locations[0], id=i, city=i, coordinate=(lon, 51)))
            p.stations.append(dict(p.stations[0], id=i, city=i, location=i))
        p.locations[2]['coordinate'] = (.1, 51)
        p.locations[3]['coordinate'] = (.2, 51)
        p.edges.append(dict(p.edges[0], id=3, a=4, b=5))
        p.demand = [dict(id=i, origin=a, destination=b, volume=100, km=10)
                    for i, (a, b) in enumerate([(2, 3), (4, 5)])]
        pool = RoutePool(p)
        pool.demand_weights = Mock(return_value=np.ones(2))
        rng = Mock()
        rng.random.return_value = 0
        rng.choice.side_effect = [1, 4, 5]
        proposal, action = regional_proposal(p, pool, local[:1], rng, [1], 'exploration')
        probabilities = rng.choice.call_args_list[0].kwargs['p']
        self.assertGreater(probabilities[1], probabilities[0])
        self.assertEqual(action, 'seed_region')
        self.assertIn(4, {site for stops, ids, f in proposal for site in stops})

    def test_disconnected_exploration_only_saves_connected_feasible_components(self):
        p, local = two_regions()
        p.config.annealing_iterations = 2
        state = LiveState()
        with patch('train_optimizer.annealing.propose', return_value=(local, 'regional_test')):
            status = solve_annealing(p, state, 5, threading.Event())
        self.assertEqual(status, 'Annealing feasible')
        self.assertEqual(len(connected_designs(local)), 2)
        self.assertEqual(connectivity(state.snapshot()[1:3])['component_count'], 1)
        self.assertEqual(state.snapshot()[-1]['component_count'], 1)
        self.assertEqual(state.snapshot()[-1]['exploratory_component_count'], 2)
        self.assertEqual(state.snapshot()[-1]['exploratory_coverage_fraction'], 1)
        self.assertEqual(state.snapshot()[-1]['coverage_fraction'], .5)

    def test_new_hyperparameters_are_validated(self):
        from train_optimizer.settings import Config
        for values in [dict(annealing_integration_coverage=0), dict(annealing_connectivity_target=1.1),
                       dict(annealing_connectivity_penalty=-1), dict(annealing_network_effect_weight=-1),
                       dict(annealing_exploration_radius_km=0)]:
            with self.assertRaises(ValueError):
                Config.from_dict(values)
