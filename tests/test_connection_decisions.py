"""Demand-led construction packages and capacity-aware service preservation."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from test_railgen import config
from test_hybrid_services import fixture
from rail_gen.alignment import CandidateGraph
from rail_gen.evaluator import Evaluator
from rail_gen.incremental import demand_connections
from rail_gen.services import Service, ServiceNetwork


class ConnectionDecisionTests(TestCase):
    def setUp(self):
        self.c = config()
        self.g = fixture()
        # Isolate an endpoint market; the middle node is a transfer station.
        self.g.cities[1].kappa = 0
        self.e = Evaluator(self.g, self.c)

    def test_full_connection_is_proposed_without_a_direct_candidate(self):
        graph = CandidateGraph.build(self.g.cities, self.g.edges[:2], self.g.tile_costs)
        e = Evaluator(graph, self.c)
        current = e.evaluate([0, 0])
        proposals = demand_connections(current, e)
        self.assertIn((1, 1), proposals)
        self.assertIn((2, 2), proposals)
        # City names have no effect on proposal selection.
        for city in graph.cities:
            city.name = 'Renamed'
        self.assertEqual(set(proposals), set(demand_connections(current, e)))

    def test_addition_keeps_incumbent_plan_when_regeneration_is_worse(self):
        current = self.e.evaluate([2, 2, 0], services=[Service((0, 1, 2), 'HSR', 3, 2)],
                                  track_counts=[4, 4, 0])
        with patch('rail_gen.evaluator.generate_plans', return_value=[
                ServiceNetwork(self.e, np.array([2, 2, 2]), (), np.array([4, 4, 2]))]):
            result = self.e.evaluate([2, 2, 2], incumbent=current, decision='growth')
        self.assertGreaterEqual(result.metrics['passengers'], current.metrics['passengers'] - 1e-6)
        np.testing.assert_array_equal(result.track_counts, [4, 4, 2])
        self.assertEqual(result.service_plan, current.service_plan)
        self.assertGreater(result.metrics['track_capital_eur'], current.metrics['track_capital_eur'])

    def test_residual_market_uses_slower_service_with_its_own_elasticity(self):
        network = ServiceNetwork(self.e, np.array([2, 2, 0]), [
            Service((0, 1, 2), 'HSR', 3), Service((0, 1, 2), 'CR', 2)], np.array([2, 2, 0]))
        first = network.assign(np.array([3., 2.]))
        initial = first['trips'][0]
        capacities = np.array([initial * .1, initial * .1, initial, initial])
        assignment, carried = network.assign_capacity(np.array([3., 2.]), capacities, first)
        loads = np.asarray(assignment['incidence'] @ (carried / 2)).ravel()
        self.assertGreater(loads[2], 0)
        self.assertTrue(np.all(loads <= capacities + 1e-7))
        self.assertGreater(carried.sum(), initial * .2)
        self.assertLess(carried.sum(), initial)  # Slower journeys attract fewer trips.
        np.testing.assert_allclose(assignment['trips'], self.e.reference_trips +
                                   assignment['shift_change'] + assignment['induced_change'])
        self.assertGreater(assignment['journey_time'][0], first['journey_time'][0])
        # Temporary blocked arcs never leak into later assignments.
        np.testing.assert_allclose(network.assign(np.array([3., 2.]))['trips'], first['trips'])

    def test_split_path_revenue_and_segment_loads_reconcile(self):
        self.c['demand']['k'] = 10
        e = Evaluator(self.g, self.c)
        result = e.evaluate([2, 2, 2], services=[
            Service((0, 2), 'HSR', 1), Service((0, 1, 2), 'CR', 6)], details=True)
        self.assertTrue(np.all(result.segment_load <= result.segment_capacity + 1e-7))
        self.assertGreater(result.segment_load[1], 0)
        self.assertAlmostEqual(result.edge_revenue.sum(), result.metrics['revenue_eur'], delta=1e-5)
        self.assertAlmostEqual(result.edge_net.sum(), result.metrics['net_eur'], delta=1e-5)

    def test_expansion_can_add_service_without_reducing_incumbent_frequencies(self):
        current = self.e.evaluate([2, 0, 0], services=[Service((0, 1), 'HSR', 2)],
                                  track_counts=[4, 0, 0])
        result = self.e.evaluate([2, 2, 0], incumbent=current, decision='growth')
        self.assertGreater(result.metrics['passengers'], current.metrics['passengers'])
        self.assertGreaterEqual(result.track_counts[0], 4)

    def test_growth_accepts_complete_path_when_individual_edges_have_no_market(self):
        from rail_gen.search import search
        graph = CandidateGraph.build(self.g.cities, self.g.edges[:2], self.g.tile_costs)
        self.c['search'].update(algorithm='incremental', generations=1, population=4)
        run = search(graph, self.c, progress=None)
        self.assertTrue(np.all(run.history[-1]['genome']))
        self.assertGreater(run.history[-1]['metrics']['passengers'], 0)

    def test_empty_market_produces_no_packages(self):
        for city in self.g.cities:
            city.kappa = 0
        e = Evaluator(self.g, self.c)
        self.assertEqual(demand_connections(e.evaluate([0, 0, 0]), e), {})
