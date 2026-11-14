"""Regression checks for reusable infrastructure and frequency-dependent routing."""
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from test_railgen import config
from test_hybrid_services import fixture
from rail_gen.analysis import network_analysis
from rail_gen.evaluator import Evaluator
from rail_gen.services import Service, ServiceNetwork, generate_plans


class EvaluationReuseTests(TestCase):
    def setUp(self):
        self.config = config()
        self.graph = fixture(shared=True)
        self.e = Evaluator(self.graph, self.config)

    def assert_assignment_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in actual.items():
            other = expected[name]
            if sparse.issparse(value):
                np.testing.assert_array_equal(value.toarray(), other.toarray(), err_msg=name)
            else:
                np.testing.assert_array_equal(value, other, err_msg=name)

    def test_frequency_changes_update_routes_waits_and_loads(self):
        # A fast through service competes with feeders: changing frequencies
        # must update both waits and which service the passengers choose.
        x, counts = np.array([2, 1, 0]), np.array([2, 2, 0])
        services = [Service((0, 1, 2), 'HSR', 1),
                    Service((0, 1), 'CR', 1), Service((1, 2), 'CR', 1)]
        network = ServiceNetwork(self.e, x, services, counts)
        first = network.assign(np.array([.3333333333333333, 6., 6.]))
        second = network.assign(np.array([6., .3333333333333333, .3333333333333333]))
        self.assertFalse(np.array_equal(first['wait'], second['wait']))
        self.assertNotEqual((first['incidence'] != second['incidence']).nnz, 0)
        for frequencies in [np.array([6., 1., 2.]), np.array([.3333333333333333, 6., 6.])]:
            fresh = ServiceNetwork(self.e, x, services, counts).assign(frequencies)
            self.assert_assignment_equal(network.assign(frequencies), fresh)
        # Returned data from earlier assignments remains independent.
        self.assert_assignment_equal(first, network.assign(np.array([.3333333333333333, 6., 6.])))

    def test_rejected_services_leave_valid_network_resources(self):
        self.config['operations']['minimum_frequency'] = 4
        self.config['track_layouts']['double']['max_frequency']['HSR'] = 6
        x, counts = np.array([2, 1, 2]), np.full(3, 2)
        networks = generate_plans(self.e, x, counts, return_networks=True)
        self.assertEqual([n.services for n in networks], generate_plans(self.e, x, counts))
        self.assertTrue(any(len(n.services) < 3 for n in networks))
        for network in networks:
            rebuilt = ServiceNetwork(self.e, x, network.services, counts)
            np.testing.assert_array_equal(network.resource_services.toarray(), rebuilt.resource_services.toarray())
            frequencies = np.full(len(network.services), 4.)
            self.assert_assignment_equal(network.assign(frequencies), rebuilt.assign(frequencies))
            self.assertTrue(np.all(network.resource_services @ frequencies <= network.resource_caps + 1e-9))

    def test_plan_routing_uses_sparse_dijkstra_with_source_subset(self):
        # Source city 1 occupies result row 0. Its only path to city 2 goes
        # through city 0, including a valid zero-weight infrastructure edge.
        self.e.origins = np.array([1])
        self.e.destinations = np.array([2])
        self.e.sources = np.array([1])
        self.e.source_row = np.array([0])
        self.e.ranked_od = [0]
        self.e.lengths[0, 1] = 0
        with patch('rail_gen.services.shortest_path', wraps=shortest_path) as routing:
            plans = generate_plans(self.e, np.array([1, 0, 1]), np.array([2, 0, 2]))
        self.assertTrue(any(s.route == (1, 0, 2) for plan in plans for s in plan))
        self.assertTrue(sparse.isspmatrix_csr(routing.call_args.args[0]))
        self.assertEqual(routing.call_args.kwargs['method'], 'D')
        np.testing.assert_array_equal(routing.call_args.kwargs['indices'], [1])

        disconnected = generate_plans(self.e, np.array([0, 0, 1]), np.array([0, 0, 2]))
        self.assertTrue(all(s.route == (0, 2) for plan in disconnected for s in plan))

    def test_direct_segment_lookup_preserves_assignment(self):
        network = ServiceNetwork(self.e, np.array([2, 1, 2]),
            [Service((2, 1, 0), 'HSR', 1), Service((0, 1), 'CR', 1),
             Service((1, 2), 'CR', 1), Service((0, 2), 'HSR', 1)],
            np.full(3, 2))
        network._prepare_assignment()

        def binary_lookup(parents, nodes):
            keys = parents.astype(np.int64) * network._size + nodes
            return network._arc_segments[np.searchsorted(network._arc_keys, keys)]

        # Check every graph arc, including reverse riding, boarding, and the
        # explicit zero-cost alighting arcs that sparse nonzero() would omit.
        adjacency = network._adjacency
        parents = np.repeat(np.arange(network._size), np.diff(adjacency.indptr))
        np.testing.assert_array_equal(
            network._segments_for_arcs(parents, adjacency.indices),
            binary_lookup(parents, adjacency.indices))

        for frequencies in ([6., 1., 2., 1.], [1., 6., 6., 2.]):
            for blocked in (None, np.zeros(5, dtype=bool),
                            np.array([True, False, False, False, True]),
                            np.ones(5, dtype=bool)):
                actual = network.assign(frequencies, blocked)
                with patch.object(network, '_segments_for_arcs', side_effect=binary_lookup):
                    expected = network.assign(frequencies, blocked)
                self.assert_assignment_equal(actual, expected)
                for name in ('incidence', 'edge_incidence'):
                    for array in ('data', 'indices', 'indptr'):
                        np.testing.assert_array_equal(
                            getattr(actual[name], array), getattr(expected[name], array))

    def test_candidate_resources_do_not_leak_between_threads_or_track_counts(self):
        cases = [(np.array(x), np.array(counts)) for x, counts in [
            ([1, 1, 0], [1, 1, 0]), ([1, 1, 0], [4, 4, 0]),
            ([2, 1, 2], [2, 4, 2]), ([0, 0, 0], [0, 0, 0])]]
        def evaluate(case):
            x, counts = case
            return self.e.evaluate(x, repair=True, track_counts=counts)
        expected = [evaluate(case) for case in cases]
        with ThreadPoolExecutor(4) as workers:
            actual = list(workers.map(evaluate, cases * 3))
        for index, result in enumerate(actual):
            other = expected[index % len(cases)]
            self.assertEqual(result.metrics, other.metrics)
            self.assertEqual(result.service_plan, other.service_plan)
            np.testing.assert_array_equal(result.objectives, other.objectives)
            np.testing.assert_array_equal(result.trips, other.trips)
        self.assertGreater(expected[1].metrics['track_capital_eur'], expected[0].metrics['track_capital_eur'])

    def test_analysis_can_reuse_complete_result_without_changing_output(self):
        result = self.e.evaluate([2, 1, 0])
        expected = network_analysis(self.graph, self.config, result, self.e)
        with patch.object(self.e, 'evaluate', side_effect=AssertionError('Unexpected reevaluation')):
            actual = network_analysis(self.graph, self.config, result, self.e, reevaluate=False)
        self.assertEqual(actual, expected)

    def test_empty_demand_supports_repeated_assignment(self):
        self.config['demand']['k'] = 0
        e = Evaluator(self.graph, self.config)
        network = ServiceNetwork(e, np.array([1, 0, 0]), [Service((0, 1), 'CR', 1)], np.array([2, 0, 0]))
        self.assert_assignment_equal(network.assign([1.]), network.assign([2.]))
        self.assertEqual(len(e.evaluate([1, 0, 0], repair=True).trips), 0)
