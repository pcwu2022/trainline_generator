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

    def test_speed_up_bounds_regional_demand_sampling(self):
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
        self.assertTrue(all(call.args[1] <= 10 for call in sample.call_args_list))
