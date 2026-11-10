import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from test_heuristic import problem
from train_optimizer.annealing import RoutePool, evaluate, financial_cost, propose, solve_annealing
from train_optimizer.settings import Config
from train_optimizer.outputs import LiveState


class AnnealingTests(unittest.TestCase):
    def test_financial_objective_does_not_use_blas_matmul(self):
        class NoMatmul(np.ndarray):
            def __matmul__(self, other):
                raise AssertionError('Financial reduction must avoid BLAS matmul')

        p = problem()
        with patch('train_optimizer.annealing.financial_cost',
                   side_effect=lambda model: financial_cost(model).view(NoMatmul)):
            with np.errstate(all='raise'):
                result = evaluate(p, RoutePool(p), (((0, 1), (0,), 1),), .1,
                                  lambda: 5, threading.Event())
        self.assertIsNotNone(result)
        model, vector, service, money, *_ = result
        expected = sum(float(c) * float(v) for c, v in zip(financial_cost(model), vector))
        self.assertAlmostEqual(money, expected)
        self.assertAlmostEqual(service + money, model.objective(vector))

    def run_search(self, p, checkpoint=None, interrupted=None):
        p.config.annealing_iterations = 40
        state = LiveState()
        status = solve_annealing(p, state, 5, interrupted or threading.Event(), checkpoint)
        return state, status

    def test_empty_start_lazy_routes_and_full_validation(self):
        p = problem()
        p.config.minimum_city_served_fraction = 1
        route = {k: v for k, v in p.edges[0].items() if k not in ('id', 'a', 'b')}
        p.edges = []
        with patch('train_optimizer.annealing.Router') as router:
            router.return_value.routes.return_value = [route]
            state, status = self.run_search(p)
            self.assertEqual(router.return_value.routes.call_count, 1)
        self.assertEqual(status, 'Annealing feasible')
        _, model, vector, *_ = state.snapshot()
        self.assertTrue(model.validation(vector)['valid'])
        self.assertEqual(model.describe(vector)['metrics']['served_cities'], 2)
        self.assertEqual(state.snapshot()[-1]['iteration_unit'], 'annealing_proposal')

    def test_failed_routing_is_cached(self):
        p = problem()
        p.edges = []
        with patch('train_optimizer.annealing.Router') as router:
            router.return_value.routes.return_value = []
            pool = RoutePool(p)
            self.assertEqual(pool.get(0, 1), [])
            self.assertEqual(pool.get(1, 0), [])
            self.assertEqual(router.return_value.routes.call_count, 1)

    def test_exploration_relaxes_only_coverage(self):
        p = problem()
        p.config.minimum_city_served_fraction = 1
        p.config.minimum_served_fraction = 1
        # A third city cannot yet be served by this partial design.
        p.cities.append(dict(id=2, name='Third'))
        pool = RoutePool(p)
        design = (((0, 1), (0,), 1),)
        result = evaluate(p, pool, design, .1, lambda: 5, threading.Event())
        self.assertIsNotNone(result)
        self.assertGreater(result[4], 0)
        self.assertIsNone(evaluate(p, pool, design, 1, lambda: 5, threading.Event(), final=True))
        p.config.upfront_budget = 1
        self.assertIsNone(evaluate(p, pool, design, .1, lambda: 5, threading.Event()))

    def test_worse_empty_proposal_cannot_erase_incumbent(self):
        p = problem()
        p.cities.append(dict(id=2, name='Unserved'))
        p.config.annealing_start_temperature = 1e9
        p.config.annealing_end_temperature = 1e9
        design = (((0, 1), (0,), 1),)
        with patch('train_optimizer.annealing.propose', side_effect=[(design, 'add'), ((), 'remove')]*20):
            state, status = self.run_search(p)
        self.assertEqual(status, 'Annealing feasible')
        self.assertIsNotNone(state.snapshot()[2])
        self.assertGreater(state.snapshot()[-1]['accepted'], 1)

    def test_checkpoint_restores_network_and_rejects_other_assumptions(self):
        p = problem()
        state, _ = self.run_search(p)
        model, vector = state.snapshot()[1:3]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.npz'
            np.savez(path, vector=vector, fingerprint=model.fingerprint(), annealing=model.annealing_checkpoint)
            resumed, status = self.run_search(p, path)
            self.assertEqual(status, 'Annealing feasible')
            self.assertLessEqual(resumed.snapshot()[1].objective(resumed.snapshot()[2]), model.objective(vector)+1e-8)
            p.config.ticket_per_km += .01
            with self.assertRaisesRegex(ValueError, 'different assumptions'):
                self.run_search(p, path)

    def test_interrupt_does_not_route(self):
        event = threading.Event()
        event.set()
        with patch('train_optimizer.annealing.Router') as router:
            state, status = self.run_search(problem(), interrupted=event)
            router.assert_not_called()
        self.assertIsNone(state.snapshot()[2])
        self.assertIn('exhausted', status)

    def test_shared_capacity_and_subsidy_stay_hard(self):
        p = problem()
        p.config.maximum_annual_subsidy = 1
        state, _ = self.run_search(p)
        self.assertIsNone(state.snapshot()[2])
        p.config.maximum_annual_subsidy = 0
        p.config.frequencies_per_hour = [2]
        p.config.track_trains_per_hour_per_direction = 1
        state, _ = self.run_search(p)
        self.assertIsNone(state.snapshot()[2])

    def test_inactive_station_choices_and_frequencies_do_not_expand_lp(self):
        p = problem()
        p.config.frequencies_per_hour = [1, 2, 4]
        p.stations[0]['capture_fraction'] = .2
        p.locations.append(dict(p.locations[0], id=2))
        p.stations.append(dict(p.stations[0], id=2, location=2, capture_fraction=1))
        pool = RoutePool(p)
        model, vector, *_ = evaluate(p, pool, (((0, 1), (0,), 1),), .1,
                                     lambda: 5, threading.Event())
        self.assertEqual(len(model.problem.stations), 2)
        self.assertEqual(len(model.z), 1)
        self.assertEqual(model.capture_upper_bound, 100)
        self.assertLessEqual(sum(vector[i] for i in model.q.values()), 20+1e-8)

    def test_station_move_updates_every_incident_service(self):
        p = problem()
        p.cities.append(dict(id=2, name='Third'))
        p.locations.extend([dict(p.locations[1], id=2, city=2), dict(p.locations[0], id=3)])
        p.stations.extend([dict(p.stations[1], id=2, city=2, location=2),
                           dict(p.stations[0], id=3, location=3)])
        p.edges.extend([dict(p.edges[0], id=1, a=0, b=2),
                        dict(p.edges[0], id=2, a=3, b=1),
                        dict(p.edges[0], id=3, a=3, b=2)])
        rng = Mock()
        rng.choice.side_effect = ['move', 0, 3, 2, 3]
        design = (((0, 1), (0,), 1), ((0, 2), (1,), 1))
        moved, action = propose(p, RoutePool(p), design, rng, [1])
        self.assertEqual(action, 'move')
        self.assertNotIn(0, {site for stops, ids, f in moved for site in stops})
        self.assertEqual({edge for stops, ids, f in moved for edge in ids}, {2, 3})

    def test_two_services_share_construction_but_consume_combined_capacity(self):
        p = problem()
        p.cities.append(dict(id=2, name='Third'))
        p.locations.append(dict(p.locations[1], id=2, city=2))
        p.stations.append(dict(p.stations[1], id=2, city=2, location=2))
        p.edges.append(dict(p.edges[0], id=1, a=1, b=2))
        design = (((0, 1), (0,), 1), ((0, 1, 2), (0, 1), 1))
        p.config.track_trains_per_hour_per_direction = 1
        self.assertIsNone(evaluate(p, RoutePool(p), design, 1, lambda: 5, threading.Event()))
        p.config.track_trains_per_hour_per_direction = 2
        model, vector, *_ = evaluate(p, RoutePool(p), design, 1, lambda: 5, threading.Event())
        self.assertEqual(len(model.y), 2)
        self.assertEqual(sum(vector[i]*model.capital[i] for i in model.y.values()), 2e9)

    def test_connected_additions_do_not_route_disconnected_pairs(self):
        p = problem()
        for i in (2, 3):
            p.cities.append(dict(id=i, name=str(i)))
            p.locations.append(dict(p.locations[1], id=i, city=i))
            p.stations.append(dict(p.stations[1], id=i, city=i, location=i))
        p.edges.append(dict(p.edges[0], id=1, a=1, b=2))
        p.demand = [dict(id=i, origin=a, destination=b, volume=v, km=100)
                    for i, (a, b, v) in enumerate([(0, 1, 100), (1, 2, 100), (2, 3, 10000)])]
        rng = Mock()
        rng.choice.side_effect = ['add', 1, 1, 2, 1]
        pool = RoutePool(p)
        with patch('train_optimizer.annealing.Router') as router:
            added, action = propose(p, pool, (((0, 1), (0,), 1),), rng, [1])
            router.assert_not_called()
        probabilities = rng.choice.call_args_list[1].kwargs['p']
        self.assertEqual(probabilities[2], 0)
        self.assertEqual(len(added), 2)

    def test_invalid_settings(self):
        for data in [dict(annealing_iterations=1), dict(annealing_end_temperature=0),
                     dict(annealing_initial_financial_weight=0), dict(annealing_coverage_penalty=0), dict(annealing_growth_probability=1.1)]:
            with self.assertRaises(ValueError):
                Config.from_dict(data)
