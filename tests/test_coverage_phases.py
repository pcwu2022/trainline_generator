import threading
import unittest
from unittest.mock import patch

from test_heuristic import problem
from test_consolidation import corridor_problem
from train_optimizer.annealing import RoutePool, evaluate, network_coverage, solve_annealing
from train_optimizer.consolidation import service_protection
from train_optimizer.outputs import LiveState


class RecordingReporter:
    failure = None

    def __init__(self, state):
        self.state = state
        self.requested = threading.Event()
        self.reports = []

    def capture_growth(self):
        pass

    def notify_iterations(self):
        self.reports.append(self.state.snapshot()[-1])


class CoveragePhaseTests(unittest.TestCase):
    def test_below_target_stays_out_of_consolidation_through_the_last_proposal(self):
        p = problem()
        p.cities.extend([dict(id=2, name='Unserved A'), dict(id=3, name='Unserved B')])
        p.config.annealing_iterations = 8
        state = LiveState()
        reporter = RecordingReporter(state)
        design = (((0, 1), (0,), 1),)
        with patch('train_optimizer.annealing.propose', return_value=(design, 'add')):
            solve_annealing(p, state, 5, threading.Event(), reporter=reporter)
        stats = state.snapshot()[-1]
        self.assertEqual(stats['stage'], 'integration')
        self.assertEqual(stats['stopping_reason'], 'proposal_limit')
        self.assertEqual(stats['coverage_fraction'], .5)
        self.assertFalse(stats['coverage_target_met'])
        self.assertEqual(stats['cost_candidates_tested'], 0)
        self.assertAlmostEqual(reporter.reports[-1]['financial_weight'], .7)
        self.assertGreater(reporter.reports[-1]['temperature'], p.config.annealing_end_temperature)

    def test_high_coverage_switches_immediately_without_waiting_for_iterations(self):
        p, design = corridor_problem()
        p.config.annealing_iterations = 1000
        state = LiveState()
        with patch('train_optimizer.annealing.propose', return_value=(design, 'add')) as grow:
            solve_annealing(p, state, 5, threading.Event())
        stats = state.snapshot()[-1]
        self.assertEqual(grow.call_count, 1)
        self.assertEqual(stats['stage'], 'consolidation')
        self.assertGreater(stats['cost_candidates_tested'], 0)
        self.assertGreater(stats['cost_improvements'], 0)
        self.assertTrue(stats['coverage_target_met'])

    def test_consolidation_cannot_reduce_coverage_below_the_transition_target(self):
        p, design = corridor_problem()
        p.config.minimum_city_served_fraction = 0
        p.cities.append(dict(id=3, name='Unserved'))
        pool = RoutePool(p)
        baseline = evaluate(p, pool, design, 1, lambda: 5, threading.Event(), final=True)
        protection = service_protection(*baseline[:2], p.config)
        protection.update(coverage_metric='markets', minimum_coverage=.75, coverage_denominator=4)
        removed = (((0, 2), (0,), 1),)
        self.assertIsNotNone(evaluate(p, pool, removed, 1, lambda: 5, threading.Event(), final=True))
        self.assertIsNone(evaluate(p, pool, removed, 1, lambda: 5, threading.Event(), final=True, protection=protection))

    def test_coverage_metrics_use_fixed_regional_denominators(self):
        p = problem()
        for city in p.cities:
            city['population'] = 1000
        p.cities.append(dict(id=2, name='Unserved', population=2000))
        for station in p.stations:
            station['covered_population'] = 500
        pool = RoutePool(p)
        record = evaluate(p, pool, (((0, 1), (0,), 1),), 1, lambda: 5, threading.Event())[:2]
        self.assertAlmostEqual(network_coverage(record, p.config, pool.capture_upper_bound), 2/3)
        p.config.annealing_coverage_metric = 'population'
        self.assertAlmostEqual(network_coverage(record, p.config, pool.capture_upper_bound), .25)
        p.config.annealing_coverage_metric = 'demand'
        self.assertAlmostEqual(network_coverage(record, p.config, pool.capture_upper_bound), 1)
        self.assertEqual(network_coverage(None, p.config, pool.capture_upper_bound), 0)
