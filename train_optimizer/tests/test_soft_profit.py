"""Solve a small loss-making network to check the soft financial target."""
from types import SimpleNamespace
import unittest

from train_optimizer.settings import Config
from train_optimizer.model import NetworkModel


def example(weight=1):
    c = Config.from_dict(dict(train_types=['CR'], ticket_per_km=0,
                              maximum_line_stops=20, maximum_journey_detour_ratio=0,
                              annealing_initial_financial_weight=.1, annealing_proposal_cost_exponent=1, profit_shortfall_weight=weight,
                              frequencies_per_hour=[1], maximum_extra_lines=0))
    locations = [dict(id=i, city=i, xy=(i*100, 0), coordinate=(i, 51), center_offset_km=0)
                 for i in range(2)]
    return SimpleNamespace(config=c, cities=[dict(id=i, name=str(i)) for i in range(2)],
        locations=locations,
        stations=[dict(id=i, city=i, location=i, radius_km=1, capture_fraction=1, access_hours=0)
                  for i in range(2)],
        edges=[dict(id=0, a=0, b=1, km=100, track_capital=1e9, geometry=[(0, 51), (1, 51)])],
        demand=[dict(id=0, origin=0, destination=1, volume=100, km=100)])


class SoftProfitTests(unittest.TestCase):
    def test_loss_making_network_is_feasible_and_shortfall_is_penalized(self):
        model = NetworkModel(example())
        solver = model.solver(model.problem.config, 10)
        solver.run()
        self.assertEqual(solver.modelStatusToString(solver.getModelStatus()), 'Optimal')
        vector = solver.getSolution().col_value
        report = model.describe(vector)
        self.assertTrue(report['feasible'])
        self.assertFalse(report['validation']['financial_ok'])
        metrics = report['metrics']
        self.assertLess(metrics['horizon_profit'], 0)
        self.assertFalse(metrics['profit_target_met'])
        self.assertAlmostEqual(metrics['profit_shortfall'], metrics['profit_shortfall_slack'], places=3)
        self.assertAlmostEqual(metrics['profit_shortfall_penalty'],
                               metrics['profit_shortfall']/model.appraisal_annuity/1e8)
        vector[model.profit_shortfall] = 0
        self.assertFalse(model.validation(vector)['valid'])

    def test_weight_controls_penalty_and_rejects_negative_values(self):
        model = NetworkModel(example(3))
        self.assertAlmostEqual(model.cost[model.profit_shortfall], 3/1e5)
        with self.assertRaises(ValueError):
            Config.from_dict(dict(profit_shortfall_weight=-1))
