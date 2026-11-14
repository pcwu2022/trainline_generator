"""Annealing acceptance, coordinated proposals, and seeded search integration."""
from types import SimpleNamespace
import unittest
import numpy as np
from rail_gen.incremental import annealing_accept, exploratory_batch
from rail_gen.config import load_config
from rail_gen.search import search
from test_railgen import graph_fixture
from test_railgen_live import incremental_config


def result(genome, energy, hard=0):
    return SimpleNamespace(genome=np.array(genome), objectives=np.array([energy, 0, 0]),
                           constraints=np.array([hard, 0, 0, 0, 0]))


class AnnealingTests(unittest.TestCase):
    def test_acceptance_temperature_and_hard_limits(self):
        parent = result([0], 0)
        self.assertTrue(annealing_accept(parent, result([1], -1), np.ones(3), .01, np.random.default_rng(1)))
        self.assertTrue(annealing_accept(parent, result([1], 1), np.ones(3), 100, np.random.default_rng(1)))
        self.assertFalse(annealing_accept(parent, result([1], 1), np.ones(3), .001, np.random.default_rng(1)))
        self.assertFalse(annealing_accept(parent, result([1], -1, hard=1), np.ones(3), 100, np.random.default_rng(1)))

    def test_batches_include_raw_edits_and_skip_conflicts(self):
        parent = result([0, 0], 0)
        proposals = [np.array([1, 0]), np.array([2, 0]), np.array([0, 1])]
        outputs = [exploratory_batch(parent, proposals, np.random.default_rng(seed), 3)
                   for seed in range(30)]
        self.assertTrue(any(np.count_nonzero(x) == 2 for x in outputs))
        self.assertTrue(all(x[0] in (0, 1, 2) and x[1] in (0, 1) for x in outputs))

    def test_seeded_run_cools_and_preserves_archive_feasibility(self):
        c = incremental_config()
        c['search']['annealing'] = True
        first = search(graph_fixture(), c, progress=None)
        second = search(graph_fixture(), c, progress=None)
        np.testing.assert_array_equal(first.current_genome, second.current_genome)
        records = [r['annealing'] for r in first.history if 'annealing' in r]
        self.assertTrue(records)
        self.assertAlmostEqual(records[0]['temperature'], .1)
        self.assertAlmostEqual(records[-1]['temperature'], .001)
        self.assertTrue(all(a['temperature'] >= b['temperature'] for a, b in zip(records, records[1:])))
        from rail_gen.evaluator import Evaluator
        evaluator = Evaluator(graph_fixture(), c)
        self.assertTrue(all(evaluator.evaluate(x).feasible for x in first.genomes))

    def test_invalid_configuration(self):
        for override in [{'annealing_final_temperature': 0},
                         {'annealing_initial_temperature': .0001},
                         {'annealing': True, 'algorithm': 'nsga2'}]:
            with self.assertRaises(ValueError):
                load_config(overrides={'search': override})
