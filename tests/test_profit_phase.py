"""Profit transition, local saturation and exact boundary artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_railgen import graph_fixture
from test_railgen_live import incremental_config
from rail_gen.incremental import profit_choice
from rail_gen.evaluator import Evaluator
from rail_gen.live import LiveReporter
from rail_gen.search import search
from rail_gen.__main__ import parser


class ProfitTests(unittest.TestCase):
    def test_cli(self):
        self.assertTrue(parser().parse_args(['run', '--profit']).profit)

    def test_switch_and_profit_never_decreases(self):
        c = incremental_config()
        c['search'].update(profit=True, generations=30)
        updates = []
        run = search(graph_fixture(), c, progress=None, observer=updates.append)
        boundary = next(u for u in updates if u.phase == 'before_profit')
        self.assertEqual(boundary.iteration, 24)
        profit = [u for u in updates if u.phase == 'profit']
        self.assertTrue(profit)
        self.assertEqual(profit[0].iteration, 25)
        previous = boundary
        for update in profit:
            self.assertGreaterEqual(update.result.metrics['net_eur'], previous.result.metrics['net_eur'])
            previous = update
        self.assertIn(run.status, ('completed', 'profit_saturated'))
        self.assertLessEqual(updates[-1].iteration, 30)

    def test_exhaustion_before_final_twenty_percent_does_not_prevent_profit(self):
        c = incremental_config()
        c['search'].update(profit=True, generations=10)
        with patch('rail_gen.incremental.neighbours', return_value={}), \
                patch('rail_gen.incremental.demand_connections', return_value={}):
            run = search(graph_fixture(), c, progress=None)
        self.assertEqual(run.status, 'profit_saturated')
        self.assertEqual(run.history[-1]['iteration'], 9)
        self.assertEqual(run.history[-2]['phase'], 'before_profit')

    def test_profit_can_sacrifice_service(self):
        from copy import deepcopy
        e = Evaluator(graph_fixture(), incremental_config())
        current = e.evaluate([1, 1, 0])
        candidate = deepcopy(current)
        candidate.metrics['net_eur'] += 10 * current.metrics['annual_cost_eur']
        candidate.metrics['passengers'] = 0
        candidate.metrics['population_served'] = 0
        self.assertIs(profit_choice(current, [candidate], current.metrics), candidate)
        candidate.metrics['net_eur'] = current.metrics['net_eur'] - 1
        self.assertIsNone(profit_choice(current, [candidate], current.metrics))

    def test_boundary_snapshot_ignores_cadence(self):
        c, g = incremental_config(), graph_fixture()
        c['search'].update(profit=True, generations=6)
        c['reporting']['snapshot_every'] = 100
        with tempfile.TemporaryDirectory() as temporary:
            reporter = LiveReporter(g, c, temporary)
            updates = []
            def observe(update):
                updates.append(update)
                reporter(update)
            run = search(g, c, progress=None, observer=observe)
            reporter.finish(run)
            directory = Path(temporary)
            saved = json.loads((directory / 'before_profit.json').read_text())
            boundary = next(u for u in updates if u.phase == 'before_profit')
            self.assertEqual(saved['iteration'], 5)
            np.testing.assert_array_equal(saved['genome'], boundary.result.genome)
            self.assertEqual(saved['metrics'], boundary.result.metrics)
            self.assertTrue((directory / 'before_profit.png').is_file())
            self.assertTrue((directory / 'before_profit.geojson').is_file())
