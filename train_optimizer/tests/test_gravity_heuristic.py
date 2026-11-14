import threading
import unittest
from unittest.mock import patch

from test_heuristic import problem
from train_optimizer.annealing import RoutePool, evaluate, gravity_pairs, solve_annealing
from train_optimizer.outputs import LiveState


class GravityTests(unittest.TestCase):
    def test_ranking_combines_reverse_pairs_and_uses_gravity(self):
        p = problem()
        p.cities = [dict(id=i, population=n) for i, n in enumerate([100, 200, 1000])]
        p.demand = [dict(origin=0, destination=1, km=10, volume=999),
                    dict(origin=1, destination=0, km=10, volume=999),
                    dict(origin=0, destination=2, km=10, volume=1)]
        self.assertEqual(list(gravity_pairs(p)), [(0, 2), (0, 1)])

    def test_conventional_evaluation_disables_high_speed(self):
        p = problem()
        p.config.train_types = ['CR', 'HSR']
        result = evaluate(p, RoutePool(p), (((0, 1), (0,), 1),), 1,
                          lambda: 5, threading.Event(), conventional_only=True)
        self.assertIsNotNone(result)
        model, vector = result[:2]
        self.assertTrue(all(vector[i] < 1e-6 for i in model.hsr_tracks.values()))
        for (line, frequency), index in model.z.items():
            if model.lines[line]['train_type'] == 'HSR':
                self.assertLess(vector[index], 1e-6)

    def test_gravity_runs_first_then_transitions(self):
        p = problem()
        p.config.gravity_heuristic = True
        p.config.gravity_heuristic_demand_threshold = .01
        p.config.annealing_iterations = 5
        state = LiveState()
        calls = []
        original = evaluate
        def recording(*args, **kwargs):
            calls.append(kwargs.get('conventional_only', False))
            return original(*args, **kwargs)
        with patch('train_optimizer.annealing.evaluate', side_effect=recording):
            solve_annealing(p, state, 5, threading.Event())
        self.assertTrue(calls[0])
        self.assertNotEqual(state.snapshot()[-1]['stage'], 'gravity_heuristic')
        self.assertIsNotNone(state.snapshot()[2])

    def test_speed_up_preserves_regional_demand_sampling(self):
        import numpy as np
        from train_optimizer.network_growth import regional_proposal
        from train_optimizer.sampling import sample_indices
        from test_network_growth import two_regions
        p, design = two_regions()
        p.config.speed_up = True
        with patch('train_optimizer.network_growth.sample_indices', wraps=sample_indices) as sample:
            regional_proposal(p, RoutePool(p), design, np.random.default_rng(1), [1],
                              'exploration', allowed=['seed_region'])
        self.assertTrue(sample.called)
        self.assertTrue(all(call.args[1] == p.config.annealing_candidate_pair_limit
                            for call in sample.call_args_list))

    def test_speed_up_limits_only_exploratory_solves(self):
        from train_optimizer.model import NetworkModel
        p = problem()
        p.config.speed_up = True
        p.config.mip_relative_gap = .01
        pool = RoutePool(p)
        original = NetworkModel.solver
        calls = []
        def recording(model, config, limit):
            calls.append((config.mip_relative_gap, limit))
            return original(model, config, limit)
        with patch.object(NetworkModel, 'solver', recording):
            record = evaluate(p, pool, (((0, 1), (0,), 1),), 1,
                              lambda: 5, threading.Event())
            self.assertIsNotNone(record)
            evaluate(p, pool, (((0, 1), (0,), 1),), 1,
                     lambda: 5, threading.Event(), final=True, previous=record[:2])
        self.assertEqual(calls, [(.05, 2.), (.01, 5)])
        self.assertEqual(p.config.mip_relative_gap, .01)
        self.assertGreater(pool.timings['solving'], 0)

    def test_speed_up_timeout_is_unevaluated(self):
        import highspy
        from unittest.mock import MagicMock
        from train_optimizer.annealing import EvaluationTimeout
        p = problem()
        p.config.speed_up = True
        solver = MagicMock()
        solver.getSolution.return_value.value_valid = False
        solver.getModelStatus.return_value = highspy.HighsModelStatus.kTimeLimit
        with patch('train_optimizer.annealing.NetworkModel.solver', return_value=solver):
            with self.assertRaises(EvaluationTimeout):
                evaluate(p, RoutePool(p), (((0, 1), (0,), 1),), 1,
                         lambda: 5, threading.Event())

    def test_service_start_matches_patterns_after_line_ids_change(self):
        from types import SimpleNamespace
        from train_optimizer.annealing import service_start
        line = dict(stops=[0, 1], segment_km=[100], geometry=[(0, 0), (1, 1)], train_type='CR')
        old = SimpleNamespace(lines=[line], z={(0, 1): 0})
        new = SimpleNamespace(lines=[dict(line, train_type='HSR'), line],
                              z={(0, 1): 7, (1, 1): 8, (1, 2): 9})
        self.assertEqual(service_start(new, (old, [1.])), {8: 1.})

    def test_speed_up_respects_remaining_budget_and_looser_configured_gap(self):
        from train_optimizer.model import NetworkModel
        p = problem()
        p.config.speed_up = True
        p.config.mip_relative_gap = .1
        original = NetworkModel.solver
        calls = []
        def recording(model, config, limit):
            calls.append((config.mip_relative_gap, limit))
            return original(model, config, limit)
        with patch.object(NetworkModel, 'solver', recording):
            evaluate(p, RoutePool(p), (((0, 1), (0,), 1),), 1,
                     lambda: .5, threading.Event())
        self.assertEqual(calls, [(.1, .5)])
