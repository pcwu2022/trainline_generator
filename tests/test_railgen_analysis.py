"""Financial allocations reconcile and endpoint satisfaction excludes transfer traffic."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_railgen import config, graph_fixture
from rail_gen.analysis import network_analysis, write_analysis, demand_denominator
from rail_gen.evaluator import Evaluator


class NetworkAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config, self.graph = config(), graph_fixture()
        self.evaluator = Evaluator(self.graph, self.config)

    def test_financial_allocations_reconcile_to_network_totals(self):
        # Deliberately supply a result without detailed financial allocations.
        result = self.evaluator.evaluate([1, 2, 0])
        data = network_analysis(self.graph, self.config, result, self.evaluator)
        m, rows = data['metrics'], data['corridors']
        for key, total in [('revenue_eur', m['revenue_eur']),
                           ('annual_cost_eur', m['annual_cost_eur']),
                           ('annual_net_eur', m['net_eur'])]:
            self.assertAlmostEqual(sum(r[key] for r in rows) / 1e6, total / 1e6, places=8)
        for row in rows:
            self.assertAlmostEqual(row['revenue_cost_ratio'], row['revenue_eur'] / row['annual_cost_eur'])

    def test_station_endpoints_do_not_count_transfer_legs(self):
        e = self.evaluator
        result = e.evaluate([1, 1, 0])
        data = network_analysis(self.graph, self.config, result, e)
        rows = data['stations']
        self.assertAlmostEqual(sum(r['carried_trips'] for r in rows), 2 * result.metrics['passengers'])
        self.assertAlmostEqual(sum(r['eligible_trips'] for r in rows), 2 * result.metrics['eligible_need_trips'])
        b_market = (e.origins == 1) | (e.destinations == 1)
        self.assertAlmostEqual(rows[1]['carried_trips'], result.trips[b_market].sum())
        # A-to-C trips pass through B and belong to both corridor loads, not B's endpoints.
        self.assertGreater(sum(r['carried_trips'] for r in data['corridors']), result.metrics['passengers'])

    def test_station_market_keeps_unserved_destinations(self):
        e = self.evaluator
        result = e.evaluate([1, 0, 0])
        data = network_analysis(self.graph, self.config, result, e)
        self.assertFalse(data['stations'][2]['served'])
        self.assertGreater(data['stations'][2]['eligible_trips'], 0)
        self.assertEqual(data['stations'][2]['satisfaction'], 0)
        a_market = (e.origins == 0) | (e.destinations == 0)
        self.assertAlmostEqual(data['stations'][0]['eligible_trips'], demand_denominator(e, result)[a_market].sum())

    def test_empty_network_dashboard_is_valid_and_json_has_no_nan(self):
        result = self.evaluator.evaluate([0, 0, 0])
        with tempfile.TemporaryDirectory() as temporary:
            image, path = Path(temporary) / 'analysis.png', Path(temporary) / 'analysis.json'
            data = write_analysis(self.graph, self.config, result, image, path, evaluator=self.evaluator)
            self.assertTrue(image.exists())
            self.assertEqual(data['corridors'], [])
            self.assertEqual(json.loads(path.read_text())['genome'], [0, 0, 0])
            self.assertNotIn('NaN', path.read_text())
