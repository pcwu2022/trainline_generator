import threading
import unittest

from test_heuristic import problem
from train_optimizer.annealing import RoutePool, evaluate, network_coverage, solve_annealing
from train_optimizer.config import Config
from train_optimizer.consolidation import service_protection
from train_optimizer.needs import rail_need_bound
from train_optimizer.outputs import LiveState


class PassengerNeedsTests(unittest.TestCase):
    def test_default_metric_is_needs(self):
        self.assertEqual(Config().annealing_coverage_metric, 'needs')

    def test_full_station_coverage_does_not_mean_passenger_needs_are_satisfied(self):
        p = problem()
        p.config.annealing_coverage_metric = 'needs'
        p.config.train_seats = 25
        pool = RoutePool(p)
        model, vector, *_ = evaluate(p, pool, (((0, 1), (0,), 1),), 1, lambda: 5, threading.Event())
        self.assertEqual(len(model.x), 2)
        self.assertAlmostEqual(network_coverage((model, vector), p.config, pool.capture_upper_bound), .25)
        metrics = model.describe(vector)['metrics']
        self.assertEqual(metrics['satisfied_needs_per_hour'], 25)
        self.assertEqual(metrics['rail_eligible_needs_per_hour'], 100)
        self.assertAlmostEqual(metrics['needs_satisfaction_fraction'], .25)

    def test_fixed_denominator_keeps_unserved_markets_needs_visible(self):
        p = problem()
        p.config.annealing_coverage_metric = 'needs'
        p.cities.append(dict(id=2, name='Unserved'))
        p.locations.append(dict(p.locations[1], id=2, city=2, coordinate=(2, 51)))
        p.stations.append(dict(p.stations[1], id=2, city=2, location=2))
        p.demand.append(dict(id=1, origin=0, destination=2, volume=100, km=140))
        pool = RoutePool(p)
        record = evaluate(p, pool, (((0, 1), (0,), 1),), 1, lambda: 5, threading.Event())[:2]
        self.assertEqual(pool.needs_upper_bound, 200)
        self.assertEqual(record[0].needs_upper_bound, 200)
        self.assertAlmostEqual(network_coverage(record, p.config, pool.capture_upper_bound), .5)

    def test_eligibility_accounts_for_access_capture_and_best_case_journey_limits(self):
        p = problem()
        p.stations[0]['capture_fraction'] = .5
        p.stations[1]['capture_fraction'] = .4
        self.assertAlmostEqual(rail_need_bound(p), 20)
        p.stations[0]['access_hours'] = 10
        self.assertEqual(rail_need_bound(p), 0)
        # Another feasible site restores only its own capturable demand.
        p.locations.append(dict(p.locations[0], id=2))
        p.stations.append(dict(p.stations[0], id=2, location=2, access_hours=0, capture_fraction=.2))
        self.assertAlmostEqual(rail_need_bound(p), 8)
        p.locations[1]['coordinate'] = (120, 51)
        self.assertEqual(rail_need_bound(p), 0)

    def test_cost_cutting_must_preserve_need_satisfaction_target(self):
        p = problem()
        p.config.annealing_coverage_metric = 'needs'
        p.config.frequencies_per_hour = [1, 2]
        p.config.train_seats = 50
        p.config.annealing_retained_ridership = .4
        pool = RoutePool(p)
        before = evaluate(p, pool, (((0, 1), (0,), 2),), 1, lambda: 5, threading.Event(), final=True)
        floors = service_protection(*before[:2], p.config)
        floors.update(coverage_metric='needs', minimum_coverage=.75, coverage_denominator=pool.needs_upper_bound)
        self.assertIsNone(evaluate(p, pool, (((0, 1), (0,), 1),), 1, lambda: 5, threading.Event(), final=True, protection=floors))

    def test_no_eligible_needs_does_not_enter_an_endless_goal_loop(self):
        p = problem()
        p.config.annealing_coverage_metric = 'needs'
        p.config.annealing_budget_mode = 'goals'
        p.stations[0]['access_hours'] = 10
        state = LiveState()
        solve_annealing(p, state, 5, threading.Event())
        self.assertEqual(state.snapshot()[-1]['stopping_reason'], 'no_eligible_needs')
        self.assertFalse(state.snapshot()[-1]['coverage_target_met'])
