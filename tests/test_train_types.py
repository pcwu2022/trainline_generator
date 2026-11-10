import threading
import unittest

from test_heuristic import problem
from train_optimizer.annealing import RoutePool, evaluate
from train_optimizer.settings import Config
from train_optimizer.heuristic import solve_heuristic
from train_optimizer.model import NetworkModel
from train_optimizer.outputs import LiveState


class TrainTypeTests(unittest.TestCase):
    def make_problem(self):
        p = problem()
        p.config.train_types = ['CR', 'HSR']
        p.config.cost_weight = 0
        p.config.profit_shortfall_weight = 0
        p.config.ticket_per_km = .2
        return p

    def solve(self, p):
        model = NetworkModel(p)
        solver = model.solver(p.config, 5)
        solver.run()
        vector = solver.getSolution().col_value
        self.assertTrue(model.validation(vector)['valid'])
        return model, vector, model.describe(vector)

    def test_hsr_speed_costs_fares_and_shared_upgrade(self):
        p = self.make_problem()
        model, vector, result = self.solve(p)
        line = result['lines'][0]
        self.assertEqual(line['train_type'], 'HSR')
        self.assertEqual(line['speed_kmh'], 300)
        self.assertAlmostEqual(line['runtime_hours'], 100/300)
        self.assertAlmostEqual(result['tracks'][0]['capital_cost'], 1e9*3.75)
        self.assertAlmostEqual(result['tracks'][0]['annual_maintenance'], 1e9*3.75*.02)
        self.assertAlmostEqual(line['annual_operating_cost'], 2*p.config.annual_hours*100*15*1.65)
        self.assertAlmostEqual(result['metrics']['annual_revenue'], 100*100*.2*2*p.config.annual_hours)
        self.assertEqual(len(result['lines']), 1)

    def test_high_infrastructure_cost_can_favor_cr(self):
        p = self.make_problem()
        p.config.cost_weight = 10
        _, _, result = self.solve(p)
        self.assertEqual(result['lines'][0]['train_type'], 'CR')
        self.assertAlmostEqual(result['tracks'][0]['capital_cost'], 1e9)

    def test_hsr_makes_short_journey_limit_feasible(self):
        p = self.make_problem()
        p.config.maximum_journey_hours = 1
        _, _, result = self.solve(p)
        self.assertEqual(result['lines'][0]['train_type'], 'HSR')
        self.assertAlmostEqual(result['metrics']['served_trips_per_hour'], 100)

    def test_annealing_selects_technology(self):
        p = self.make_problem()
        record = evaluate(p, RoutePool(p), (((0, 1), (0,), 1),), 1,
                          lambda: 5, threading.Event())
        self.assertIsNotNone(record)
        model, vector = record[:2]
        self.assertEqual(model.describe(vector)['lines'][0]['train_type'], 'HSR')

    def test_heuristic_selects_technology(self):
        p = self.make_problem()
        model = NetworkModel(p)
        state = LiveState()
        state.update(problem=p, model=model)
        solve_heuristic(model, state, 5, threading.Event())
        vector = state.snapshot()[2]
        self.assertTrue(model.validation(vector)['valid'])
        self.assertEqual(model.describe(vector)['lines'][0]['train_type'], 'HSR')

    def test_mixed_transfer_uses_each_trains_yield(self):
        p = self.make_problem()
        p.cities.append(dict(id=2, name='Destination'))
        p.locations.append(dict(p.locations[1], id=2, city=2, coordinate=(2, 51)))
        p.stations.append(dict(p.stations[1], id=2, city=2, location=2))
        p.edges = [dict(p.edges[0], km=50), dict(p.edges[0], id=1, a=1, b=2, km=50)]
        p.demand = [dict(id=0, origin=0, destination=2, volume=100, km=100)]
        lines = [dict(id=i, a=a, b=b, stops=[a, b], edges=[i], segment_km=[50],
                      km=50, geometry=[], train_type=t, frequencies=[1])
                 for i, (a, b, t) in enumerate([(0, 1, 'CR'), (1, 2, 'HSR')])]
        model = NetworkModel(p, lines=lines)
        solver = model.solver(p.config, 5)
        solver.run()
        result = model.describe(solver.getSolution().col_value)
        self.assertTrue(result['feasible'])
        self.assertAlmostEqual(result['metrics']['served_trips_per_hour'], 100)
        self.assertAlmostEqual(result['metrics']['annual_revenue'], 100*(50*.2+50*.4)*p.config.annual_hours)
        self.assertEqual([t['train_type'] for t in result['tracks']], ['CR', 'HSR'])

    def test_validation_and_canonical_default(self):
        self.assertEqual(Config.from_dict({}).train_types, ['CR', 'HSR'])
        for types in ([], ['CR', 'CR'], ['unknown'], {}, [['CR']]):
            with self.assertRaises(ValueError):
                Config.from_dict({'train_types': types})
        for name in ('hsr_capex_ratio', 'hsr_opex_ratio', 'hsr_speed_ratio', 'hsr_fare_ratio'):
            for value in (0, -1, float('inf'), float('nan'), True):
                with self.assertRaises(ValueError):
                    Config.from_dict({name: value})
