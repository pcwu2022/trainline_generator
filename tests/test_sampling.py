import itertools
import unittest

import numpy as np

from train_optimizer.sampling import (PHASE_ACTION_WEIGHTS, choose_action, pop_cost_candidate,
                                      sample_actions, sample_indices, sample_pairs)
from train_optimizer.settings import Config
from test_network_growth import two_regions
from train_optimizer.annealing import RoutePool
from train_optimizer.network_growth import regional_proposal


class SamplingTests(unittest.TestCase):
    def test_pair_indices_cover_triangle_and_bound_large_inputs(self):
        for size in range(12):
            items = list(range(size))
            self.assertEqual(list(sample_pairs(items, 0, None)), list(itertools.combinations(items, 2)))
        pairs = list(sample_pairs(range(100000), 256, np.random.default_rng(1)))
        self.assertEqual(len(set(pairs)), 256)
        self.assertTrue(all(0 <= a < b < 100000 for a, b in pairs))

    def test_seed_reproduces_changing_action_subsets(self):
        def walk():
            rng = np.random.default_rng(42)
            return [sample_actions(['add', 'extend', 'move'], .5, rng) for _ in range(20)]
        self.assertEqual(walk(), walk())
        self.assertTrue(all(len(x) == 2 for x in walk()))
        self.assertGreater(len({tuple(x) for x in walk()}), 1)
        self.assertEqual(len(sample_indices(10**9, 256, np.random.default_rng(1))), 256)

    def test_regional_generation_respects_action_mask(self):
        p, design = two_regions()
        proposal, action = regional_proposal(p, RoutePool(p), design,
            np.random.default_rng(3), [1], 'integration', allowed=['seed_region'])
        self.assertEqual(proposal, design)
        self.assertEqual(action, 'no_regional_candidate')

    def test_phase_priorities_change_sampled_moves(self):
        def draws(phase):
            rng = np.random.default_rng(31)
            weights = PHASE_ACTION_WEIGHTS[phase]
            result = []
            for _ in range(2000):
                allowed = sample_actions(weights, .67, rng, weights)
                result.append(choose_action(allowed, phase, rng))
            return result
        early, middle = draws('exploration'), draws('integration')
        self.assertGreater(early.count('seed_region'), early.count('bridge')*2)
        self.assertGreater(middle.count('bridge'), middle.count('seed_region')*2)
        self.assertEqual(early, draws('exploration'))

    def test_cost_sampling_retains_all_untested_candidates(self):
        original = [(100-i, family, (i,)) for i, family in enumerate(
                    ['alignment', 'reuse', 'hub', 'frequency', 'remove', 'move',
                     'junction_station', 'through']*3)]
        pending, seen = list(original), []
        rng = np.random.default_rng(12)
        while pending:
            seen.append(pop_cost_candidate(pending, .25, rng))
        self.assertCountEqual(seen, original)
        self.assertNotEqual(seen, original)

    def test_invalid_sampling_configuration(self):
        for value in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                Config.from_dict(dict(annealing_action_fraction=value))
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                Config.from_dict(dict(annealing_candidate_pair_limit=value))
