"""Adaptive weights on the original large-neighborhood incremental search."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from rail_gen.config import load_config
from rail_gen.incremental import objective_weights, weighted_objective, growth_choice
from rail_gen.search import search


class WeightedIncrementalTests(unittest.TestCase):
    def test_weight_boundaries_and_fixed_scales(self):
        for q, name, weights in [(0, 'expansion', [.05,.7,.25]),
                                 (.3, 'development', [.2,.55,.25]),
                                 (.6, 'consolidation', [.7,.2,.1])]:
            phase, actual = objective_weights(q)
            self.assertEqual(phase, name)
            np.testing.assert_allclose(actual, weights)
        result = SimpleNamespace(objectives=np.array([1e8,-2e6,-3e6,100]))
        self.assertAlmostEqual(weighted_objective(result, np.array([.05,.7,.25]),
                                                 np.array([1e8,2e6,3e6])), -.9)

    def test_growth_ranking_changes_with_weights(self):
        def result(cost, passengers, population):
            return SimpleNamespace(objectives=np.array([cost,-passengers,-population]),
                metrics=dict(annual_cost_eur=cost, passengers=passengers, population_served=population))
        current = result(0,0,0)
        a, b = result(1,10,1), result(1,1,10)
        scales = np.ones(3)
        passengers = lambda r: weighted_objective(r, np.array([.05,.7,.25]), scales)
        population = lambda r: weighted_objective(r, np.array([.05,.25,.7]), scales)
        self.assertIs(growth_choice(current, [a,b], passengers), a)
        self.assertIs(growth_choice(current, [a,b], population), b)

    def test_150_proposals_and_no_ten_iteration_early_stop(self):
        c = load_config(profile=None)
        c['search'].update(generations=12, growth_fraction=1, capacity_repair_steps=0)
        m = 11
        current = SimpleNamespace(genome=np.zeros(m,dtype=int), objectives=np.zeros(3),
            feasible=True, constraints=np.zeros(5), services=[], track_counts=np.zeros(m,dtype=int),
            metrics=dict(need_satisfaction_ratio=0., spill_fraction=0.))
        evaluator = SimpleNamespace(m=m, reference_trips=np.ones(1), population=np.ones(m),
                                   config=c, evaluate=lambda *args, **kwargs: current)
        proposals = {}
        for i in range(1, 2001):
            x = np.array([(i >> j) & 1 for j in range(m)])
            proposals[tuple(x)] = (x, 1.)
        with patch('rail_gen.incremental.Evaluator', return_value=evaluator), \
             patch('rail_gen.incremental.neighbours', return_value=proposals), \
             patch('rail_gen.incremental.demand_connections', return_value={}), \
             patch('rail_gen.incremental.topology', return_value=(1,0)):
            run = search(None, c, progress=None)
        self.assertEqual(run.history[-1]['iteration'], 12)
        self.assertEqual(run.evaluations, 1+12*150)
        self.assertFalse(any(r['accepted'] for r in run.history))
        self.assertTrue(all(r['phase'] in ('growth', 'integration') for r in run.history))
        self.assertTrue(all(r['objective_weights'] == [.05,.7,.25] for r in run.history))
