import contextlib
import io
import threading
import unittest
from unittest.mock import patch

from test_heuristic import problem
from train_optimizer.annealing import solve_annealing
from train_optimizer.settings import Config
from train_optimizer.outputs import LiveState
from train_optimizer.runner import main


class GoalStoppingTests(unittest.TestCase):
    def test_goal_mode_passes_time_and_proposal_budgets_until_interrupted(self):
        p = problem()
        p.config.annealing_budget_mode = 'goals'
        p.config.annealing_iterations = 2
        p.cities.extend([dict(id=2, name='Unserved A'), dict(id=3, name='Unserved B')])
        state, interrupted = LiveState(), threading.Event()
        clock, calls = [0.], [0]
        def propose(*args, **kwargs):
            calls[0] += 1
            clock[0] += 100
            if calls[0] == 4:
                interrupted.set()
            return (((0, 1), (0,), 1),), 'regional_test'
        with patch('train_optimizer.annealing.time.monotonic', side_effect=lambda: clock[0]), patch('train_optimizer.annealing.propose', side_effect=propose):
            solve_annealing(p, state, 5, interrupted)
        self.assertEqual(calls[0], 4)
        self.assertGreater(state.snapshot()[-1]['iterations'], p.config.annealing_iterations)
        self.assertEqual(state.snapshot()[-1]['stopping_reason'], 'interrupted')
        self.assertNotEqual(state.snapshot()[-1]['stage'], 'consolidation')

    def test_bounded_mode_still_enforces_the_total_time_limit(self):
        p = problem()
        state = LiveState()
        clock = [0.]
        def propose(*args, **kwargs):
            clock[0] = 100
            return (((0, 1), (0,), 1),), 'regional_test'
        with patch('train_optimizer.annealing.time.monotonic', side_effect=lambda: clock[0]), patch('train_optimizer.annealing.propose', side_effect=propose):
            solve_annealing(p, state, 5, threading.Event())
        self.assertEqual(state.snapshot()[-1]['stopping_reason'], 'time_limit')

    def test_goal_mode_stops_after_completed_consolidation(self):
        p = problem()
        p.config.annealing_budget_mode = 'goals'
        p.config.annealing_iterations = 2
        state = LiveState()
        with patch('train_optimizer.annealing.propose', return_value=((((0, 1), (0,), 1),), 'regional_test')):
            status = solve_annealing(p, state, 5, threading.Event())
        self.assertEqual(status, 'Annealing feasible')
        self.assertEqual(state.snapshot()[-1]['stopping_reason'], 'bounded_cost_neighborhood_exhausted')
        self.assertEqual(state.snapshot()[-1]['stage'], 'consolidation')

    def test_cli_time_limit_selects_bounded_mode_and_explicit_goals_wins(self):
        for options, expected in [([], 'goals'), (['--time-limit', '5'], 'bounded'),
                                  (['--time-limit', '5', '--budget-mode', 'goals'], 'goals')]:
            baseline = Config()
            with patch('train_optimizer.runner.load_config', return_value=baseline), patch('train_optimizer.runner.Config.from_dict', side_effect=ValueError('stop after config')) as config:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(options)
            self.assertEqual(config.call_args.args[0]['annealing_budget_mode'], expected)
        with self.assertRaises(ValueError):
            Config.from_dict(dict(annealing_budget_mode='unknown'))
