import unittest
from train_optimizer.config import Config
from train_optimizer.finance import asset_cashflows, asset_pv_factor, lifecycle_report
from test_soft_profit import example
from train_optimizer.model import NetworkModel


class LifecycleTests(unittest.TestCase):
    def test_repeated_renewal_and_terminal_boundary(self):
        self.assertEqual(asset_cashflows(100, 10, 30), ({10: 100, 20: 100}, 0))
        self.assertEqual(asset_cashflows(100, 25, 30), ({25: 100}, 80))
        self.assertEqual(asset_cashflows(100, 40, 30), ({}, 25))
        self.assertAlmostEqual(asset_pv_factor(25, 30, 0), 1.2)

    def test_discounted_cashflows_match_formula(self):
        c = Config()
        capitals = {'track': 400, 'station': 500, 'fleet': 250}
        for rate in [0, .02, .04, .06]:
            result = lifecycle_report(capitals, 200, 40, c, rate)
            self.assertAlmostEqual(result['npv'], sum(row['net']/(1+rate)**row['year']
                                                    for row in result['cashflows']))
            self.assertAlmostEqual(result['pv_cost'], 40*result['annuity_factor'] +
                sum(value*asset_pv_factor(c.asset_lives_years[kind], 30, rate)
                    for kind, value in capitals.items()))
            self.assertEqual(result['assets']['fleet']['renewals_by_year'], {25: 250})

    def test_solver_target_matches_report_and_sensitivity(self):
        model = NetworkModel(example())
        solver = model.solver(model.problem.config, 10)
        solver.run()
        metrics = model.describe(solver.getSolution().col_value)['metrics']
        self.assertEqual(metrics['profit_target_basis'], 'lifecycle_present_value')
        self.assertEqual([x['discount_rate'] for x in metrics['discount_rate_sensitivity']], [.02,.04,.06])
        self.assertAlmostEqual(metrics['horizon_cost'], metrics['capital_cost'] +
                               10*metrics['annual_operating_cost'])
        self.assertAlmostEqual(metrics['profit_shortfall'], metrics['profit_shortfall_slack'], places=3)
        self.assertAlmostEqual(metrics['profit_shortfall_penalty'],
                               metrics['profit_shortfall']/model.appraisal_annuity/1e8)

    def test_validation(self):
        for settings in [dict(appraisal_years=0), dict(discount_rate=-.1),
                         dict(discount_rates=[float('nan')]), dict(discount_rates=[]),
                         dict(asset_lives_years={'track': 0, 'station': 50, 'fleet': 25})]:
            with self.assertRaises(ValueError):
                Config.from_dict(settings)
