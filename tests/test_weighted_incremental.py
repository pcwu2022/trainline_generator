"""Adaptive weights on the original large-neighborhood incremental search."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from rail_gen.config import load_config
from rail_gen.incremental import objective_weights, weighted_objective, growth_choice, neighbours
from rail_gen.search import search


class WeightedIncrementalTests(unittest.TestCase):
    def test_station_sampling_bias_follows_satisfaction(self):
        c = load_config()
        edges = [SimpleNamespace(a=a, b=b, alignments={'CR': SimpleNamespace(capex_meur=1)})
                 for a, b in [(0, 1), (1, 2), (2, 3)]]
        e = SimpleNamespace(n=4, graph=SimpleNamespace(edges=edges), config=c,
                            a=np.array([0, 1, 2]), b=np.array([1, 2, 3]),
                            gravity_scores=np.ones(3), allowed=np.ones((3, 3), dtype=bool))
        current = SimpleNamespace(genome=np.array([1, 0, 0]), metrics={})
        for satisfaction, expected in [(0, [2, 1.5, 1]), (.2999, [2, 1.5, 1]),
                                       (.3, [5, 3, 1]), (.5999, [5, 3, 1]),
                                       (.6, [9, 5, 1])]:
            current.metrics['need_satisfaction_ratio'] = satisfaction
            with patch('rail_gen.incremental.bridge_components', side_effect=lambda x, *_: x):
                proposals = neighbours(current, e)
            for edge, weight in enumerate(expected):
                proposal = current.genome.copy()
                proposal[edge] = 2
                self.assertEqual(proposals[tuple(proposal)][1], weight)
        c['search']['served_station_bias']['consolidation'] = 0
        with patch('rail_gen.incremental.bridge_components', side_effect=lambda x, *_: x):
            self.assertTrue(all(weight == 1 for _, weight in neighbours(current, e).values()))

    def test_station_sampling_bias_validation(self):
        for invalid in [-1, float('nan'), True, 'high']:
            with self.assertRaises(ValueError):
                load_config(overrides={'search': {'served_station_bias': {'expansion': invalid}}})

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
