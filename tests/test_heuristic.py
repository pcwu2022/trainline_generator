import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from test_soft_profit import example
from train_optimizer.geography import Problem
from train_optimizer.heuristic import heuristic_problem, solve_heuristic
from train_optimizer.model import NetworkModel
from train_optimizer.outputs import LiveState


def problem():
    p = example()
    p.config.annealing_budget_mode = 'bounded'
    p.config.annealing_coverage_metric = 'markets'
    return Problem(p.config, [], p.cities, p.locations, p.stations, p.edges, p.demand, [])


class HeuristicTests(unittest.TestCase):
    def run_search(self, p, checkpoint=None):
        model = NetworkModel(heuristic_problem(p))
        state = LiveState()
        state.update(problem=model.problem, model=model)
        status = solve_heuristic(model, state, 5, threading.Event(), checkpoint)
        return model, state.snapshot()[2], status

    def test_feasible_design_passes_full_validation_and_has_no_optimality_claim(self):
        p = problem()
        p.config.minimum_city_served_fraction = 1
        p.config.minimum_served_fraction = .1
        model, vector, status = self.run_search(p)
        self.assertEqual(status, 'Heuristic feasible')
        self.assertTrue(model.validation(vector)['valid'])
        result = model.describe(vector)
        self.assertEqual(result['metrics']['served_cities'], 2)
        self.assertGreaterEqual(result['metrics']['served_fraction'], .1)
        second_model, second_vector, _ = self.run_search(p)
        self.assertAlmostEqual(model.objective(vector), second_model.objective(second_vector))

    def test_subsidy_ceiling_is_not_relaxed(self):
        p = problem()
        p.config.maximum_annual_subsidy = 1
        _, vector, status = self.run_search(p)
        self.assertIsNone(vector)
        self.assertEqual(status, 'Heuristic search exhausted without a validated network')

    def test_best_station_capture_is_chosen_without_mutating_input(self):
        p = problem()
        original = p.stations[0]
        p.stations.append(dict(original, id=2, capture_fraction=.1))
        reduced = heuristic_problem(p)
        self.assertEqual(len(reduced.stations), 2)
        self.assertEqual(reduced.stations[0]['capture_fraction'], 1)
        self.assertEqual(len(p.stations), 3)
        self.assertEqual(p.config.maximum_line_stops, 20)
        self.assertEqual(reduced.config.maximum_line_stops, 3)

    def test_checkpoint_preserves_feasible_incumbent(self):
        p = problem()
        model, vector, _ = self.run_search(p)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.npz'
            np.savez(path, vector=vector, fingerprint=model.fingerprint())
            resumed, result, status = self.run_search(p, path)
            self.assertEqual(status, 'Heuristic feasible')
            self.assertTrue(resumed.validation(result)['valid'])

    def test_interruption_returns_without_claiming_infeasibility(self):
        p = problem()
        model = NetworkModel(heuristic_problem(p))
        state = LiveState()
        interrupted = threading.Event()
        interrupted.set()
        status = solve_heuristic(model, state, 5, interrupted)
        self.assertEqual(status, 'Heuristic search exhausted without a validated network')
        self.assertIsNone(state.snapshot()[2])
        self.assertEqual(state.snapshot()[-1]['iteration_unit'], 'heuristic_design_evaluation')

    def test_no_frequency_above_track_limit_is_used(self):
        p = problem()
        p.config.frequencies_per_hour = [2]
        p.config.track_trains_per_hour_per_direction = 1
        _, vector, status = self.run_search(p)
        self.assertIsNone(vector)
        self.assertEqual(status, 'Heuristic search exhausted without a validated network')

    def test_through_train_replaces_shuttles_and_reduces_journey_time(self):
        p = problem()
        p.config.maximum_extra_lines = 12
        p.config.maximum_shared_track_detour = 2
        p.config.minimum_city_served_fraction = 1
        p.config.minimum_served_fraction = 1
        p.locations.append(dict(id=2, city=2, xy=(200, 0), coordinate=(2, 51), center_offset_km=0))
        p.cities.append(dict(id=2, name='Destination'))
        p.stations.append(dict(p.stations[0], id=2, city=2, location=2))
        p.edges.append(dict(p.edges[0], id=1, a=1, b=2, geometry=[(1, 51), (2, 51)]))
        p.demand = [dict(id=0, origin=0, destination=2, volume=100, km=200)]
        p.config.time_weight = 2
        model, vector, status = self.run_search(p)
        self.assertEqual(status, 'Heuristic feasible')
        result = model.describe(vector)
        self.assertTrue(any(len(line['stops']) == 3 for line in result['lines']))
        self.assertLess(result['metrics']['mean_journey_hours'], 200/120 + 1 + 5/60)
        self.assertTrue(model.validation(vector)['valid'])

    def test_small_city_bypass_improves_time_and_profit(self):
        p = problem()
        p.config.maximum_extra_lines = 0
        p.config.minimum_served_fraction = 1
        p.config.station_capital_cost = 1e9
        p.locations.append(dict(id=2, city=2, xy=(200, 0), coordinate=(2, 51), center_offset_km=0))
        p.cities.append(dict(id=2, name='Destination'))
        p.stations.append(dict(p.stations[0], id=2, city=2, location=2))
        p.edges.extend([
            dict(p.edges[0], id=1, a=1, b=2, geometry=[(1, 51), (2, 51)]),
            dict(p.edges[0], id=2, a=0, b=2, km=100, track_capital=2.1e9,
                 geometry=[(0, 51), (2, 51)])])
        p.demand = [dict(id=0, origin=0, destination=2, volume=100, km=100)]
        model = NetworkModel(heuristic_problem(p))
        baseline = np.zeros(len(model.names))
        for index in model.x.values():
            baseline[index] = 1
        for edge in (0, 1):
            baseline[model.y[edge]] = 1
            line = next(line for line in model.lines if line['edges'] == [edge])
            baseline[model.z[line['id'], 1]] = 1
        baseline[model.names.index('root_0')] = 1
        indices = np.flatnonzero(model.integer).astype(np.int32)
        solver = model.solver(p.config, 5)
        solver.changeColsBounds(len(indices), indices, baseline[indices], baseline[indices])
        solver.run()
        baseline = np.asarray(solver.getSolution().col_value)
        self.assertTrue(model.validation(baseline)['valid'])
        old = model.describe(baseline)
        new_model, vector, _ = self.run_search(p)
        new = new_model.describe(vector)
        self.assertEqual(new['metrics']['served_cities'], 2)
        self.assertLess(new['metrics']['mean_journey_hours'], old['metrics']['mean_journey_hours'])
        self.assertGreater(new['metrics']['horizon_profit'], old['metrics']['horizon_profit'])
        p.config.maximum_journey_detour_ratio = 1.1
        capped = NetworkModel(heuristic_problem(p))
        self.assertFalse(capped.validation(baseline)['valid'])

    def test_demand_corridors_connect_hubs_beyond_nearest_towns(self):
        from dataclasses import replace
        from train_optimizer.config import Config
        from train_optimizer.geography import candidate_market_pairs
        cities = [dict(id=i, center=(lon, 50), population=population)
                  for i, (lon, population) in enumerate([(0, 1000000), (.1, 1000),
                                                        (3, 1000000), (3.1, 1000)])]
        config = Config(solver_method="heuristic", neighbors_per_city=1, maximum_edge_km=500,
                        heuristic_demand_connections=1)
        self.assertNotIn((0, 2), candidate_market_pairs(cities, replace(config, solver_method='milp')))
        self.assertIn((0, 2), candidate_market_pairs(cities, config))

    def test_invalid_solver_configuration(self):
        from train_optimizer.config import Config
        for data in [dict(solver_method='unknown'), dict(heuristic_iterations=0),
                     dict(maximum_journey_detour_ratio=.5)]:
            with self.assertRaises(ValueError):
                Config.from_dict(data)
