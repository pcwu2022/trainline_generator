import unittest

from test_soft_profit import example
from train_optimizer.cost_log import cost_breakdown_log
from train_optimizer.model import NetworkModel


class CostLogTests(unittest.TestCase):
    def test_costs_reconcile_with_solved_network(self):
        model = NetworkModel(example())
        solver = model.solver(model.problem.config, 10)
        solver.run()
        result = model.describe(solver.getSolution().col_value)
        from dataclasses import asdict
        result.update(utc_time='2026-11-09', phase='optimal',
                      assumptions=asdict(model.problem.config), solver={'status': 'Optimal'})
        text = cost_breakdown_log(result)
        m = result['metrics']
        c = model.problem.config
        annual = (sum(t['annual_maintenance'] for t in result['tracks'])
                  + len(result['stations']) * c.station_annual_cost
                  + sum(l['annual_train_km'] * c.train_cost_per_km
                        + l['annual_departures'] * c.train_cost_per_departure for l in result['lines']))
        self.assertAlmostEqual(annual, m['annual_operating_cost'])
        self.assertIn(f"Total annual operating cost: EUR {annual:,.2f}", text)
        self.assertIn(f"Profit: EUR {m['horizon_profit']:,.2f}", text)
        self.assertIn('Profit target met: False', text)
        self.assertIn('dimensionless, not a cash cost', text)
        self.assertIn('Line 0 (0 -> 1', text)

    def test_no_incumbent_has_no_invented_costs(self):
        text = cost_breakdown_log(dict(utc_time='2026-11-09', phase='infeasible', feasible=False))
        self.assertIn('No feasible incumbent; costs are unavailable.', text)
        self.assertNotIn('Total initial capital', text)
