import unittest

from test_soft_profit import example
from train_optimizer.settings import Config
from train_optimizer.model import NetworkModel


class RegionalNetworkTests(unittest.TestCase):
    def test_overlapping_catchments_with_service_and_coverage(self):
        problem = example()
        problem.config.minimum_city_served_fraction = 1
        problem.config.minimum_selected_origin_served_fraction = 0.1
        problem.config.minimum_served_fraction = 0.1
        problem.config.minimum_service_frequency = 2
        problem.config.frequencies_per_hour = [1, 2]
        # Previously forbidden: centers 100 km apart with 75 km catchments.
        for station in problem.stations:
            station['radius_km'] = 75
        model = NetworkModel(problem)
        self.assertEqual(set(f for _, f in model.z), {2})
        solver = model.solver(problem.config, 10)
        solver.run()
        self.assertEqual(solver.modelStatusToString(solver.getModelStatus()), 'Optimal')
        result = model.describe(solver.getSolution().col_value)
        self.assertEqual(result['metrics']['served_cities'], 2)
        self.assertGreaterEqual(result['metrics']['needs_satisfaction_fraction'], 0.1 - 1e-8)
        self.assertLessEqual(result['metrics']['served_trips_per_hour'], 100)

    def test_subsidy_ceiling_is_hard_even_with_profit_slack(self):
        problem = example()
        problem.config.maximum_annual_subsidy = 1
        model = NetworkModel(problem)
        solver = model.solver(problem.config, 10)
        solver.run()
        self.assertEqual(solver.modelStatusToString(solver.getModelStatus()), 'Infeasible')

    def test_invalid_regional_options(self):
        for options in [dict(minimum_city_served_fraction=1.1),
                        dict(minimum_selected_origin_served_fraction=1.1),
                        dict(minimum_service_frequency=0),
                        dict(minimum_service_frequency=3, frequencies_per_hour=[1, 2])]:
            with self.assertRaises(ValueError):
                Config.from_dict(options)

    def test_disconnected_coverage_explains_candidate_limit(self):
        problem = example()
        problem.edges = []
        problem.config.minimum_city_served_fraction = 1
        with self.assertRaisesRegex(ValueError, 'largest candidate track component contains only 1'):
            NetworkModel(problem)
