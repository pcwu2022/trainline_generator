"""Combined edits must earn acceptance through a fresh network evaluation."""
from types import SimpleNamespace
import unittest
import numpy as np

from rail_gen.config import load_config
from rail_gen.incremental import combined_choice


def result(genome, value, hard=0):
    return SimpleNamespace(genome=np.array(genome), value=value,
                           constraints=np.array([hard, 0, 0]))


def choose(parent, candidates):
    eligible = [c for c in candidates if c.value > parent.value]
    return max(eligible, key=lambda c: c.value) if eligible else None


class CombinedChoiceTests(unittest.TestCase):
    def setUp(self):
        self.current = result([0, 0, 0], 0)
        self.a = result([1, 0, 0], 3)
        self.b = result([0, 1, 0], 2)
        self.c = result([0, 0, 1], 1)

    def test_combines_disjoint_edits_and_skips_conflicting_alternative(self):
        seen = []
        def score(genome):
            seen.append(genome.tolist())
            return result(genome, 6)
        selected, evaluated = combined_choice(self.current,
            [self.a, result([2, 0, 0], 2.5), self.b, self.c], choose, score, 10)
        self.assertEqual(seen, [[1, 1, 1]])
        self.assertIs(selected, evaluated[0])

    def test_rejects_hard_constraint_failure_and_retries_smaller_batch(self):
        candidates = [result(np.eye(4, dtype=int)[i], 4-i) for i in range(4)]
        seen = []
        def score(genome):
            seen.append(genome.tolist())
            return result(genome, 8, hard=int(genome.sum() > 2))
        selected, evaluated = combined_choice(result([0]*4, 0), candidates,
                                              choose, score, 4)
        self.assertEqual(len(evaluated), 2)
        self.assertEqual(selected.genome.tolist(), [1, 1, 0, 0])

    def test_combination_must_beat_best_single_even_if_it_improves_parent(self):
        selected, evaluated = combined_choice(self.current, [self.a, self.b],
            choose, lambda x: result(x, 2), 10)
        self.assertIs(selected, self.a)
        self.assertEqual(len(evaluated), 1)

    def test_disabled_limit_no_useful_candidates_and_stop_skip_evaluation(self):
        def score(_):
            self.fail('Unexpected combined evaluation')
        for limit, stop in [(1, False), (10, True)]:
            selected, evaluated = combined_choice(self.current, [self.a, self.b],
                choose, score, limit, lambda: stop)
            self.assertIs(selected, self.a)
            self.assertEqual(evaluated, [])
        self.assertEqual(combined_choice(self.current, [], choose, score, 10), (None, []))

    def test_config_requires_positive_integer_limit(self):
        self.assertEqual(load_config()['search']['batch_accept_limit'], 10)
        for invalid in [0, 1.5, True]:
            with self.assertRaises(ValueError):
                load_config(overrides={'search': {'batch_accept_limit': invalid}})
