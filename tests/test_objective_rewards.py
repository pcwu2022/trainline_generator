import unittest

from test_soft_profit import example
from train_optimizer.model import NetworkModel


class ObjectiveRewardTests(unittest.TestCase):
    def test_rewards_population_ridership_and_shorter_waits(self):
        problem = example(weight=0)
        c = problem.config
        c.population_weight = 1
        c.frequencies_per_hour = [1, 2]
        c.cost_weight = 0
        for city in problem.cities:
            city['population'] = 500
        for station in problem.stations:
            station['covered_population'] = 100
        # Same access, location, capture and costs: prefer covering more people.
        problem.stations.append(dict(problem.stations[0], id=2, covered_population=200))
        model = NetworkModel(problem)
        solver = model.solver(c, 10)
        solver.run()
        report = model.describe(solver.getSolution().col_value)
        self.assertEqual({s['id'] for s in report['stations']}, {1, 2})
        self.assertAlmostEqual(report['metrics']['covered_population_fraction'], 0.3)
        self.assertAlmostEqual(report['metrics']['served_trips_per_hour'], 100)
        # Equal financial costs in the objective: prefer shorter waits at 2/h.
        self.assertEqual(report['lines'][0]['frequency_per_hour_per_direction'], 2)
