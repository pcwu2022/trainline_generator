import unittest

import h3
from pyproj import Geod

from test_soft_profit import example
from train_optimizer.config import Config
from train_optimizer.geography import Router
from train_optimizer.model import NetworkModel


class RoutingAlternativeTests(unittest.TestCase):
    def router(self):
        config = Config(terrain_search_limit=2500, routing_sample_km=2)
        router = Router(config, [dict(h3=h3.latlng_to_cell(0, 0, 5), density=0)])
        router.allowed = lambda cell: abs(h3.cell_to_latlng(cell)[0]) < 1 and abs(h3.cell_to_latlng(cell)[1]) < 1
        return router

    def test_uniform_terrain_is_direct_not_hex_zigzag(self):
        router = self.router()
        a, b = (-.4, -.1), (.4, .2)
        routes = router.routes(a, b)
        direct = routes[0]
        expected = Geod(ellps='WGS84').inv(*a, *b)[2]/1000
        self.assertEqual(direct['alignment_kind'], 'direct')
        self.assertAlmostEqual(direct['km'], expected, places=6)
        self.assertEqual(len(routes), 1)

    def test_costly_terrain_keeps_direct_and_cheaper_detour(self):
        router = self.router()
        def cost(cell):
            lat, lon = h3.cell_to_latlng(cell)
            return 200e6 if abs(lon) < .15 and abs(lat-.1) < .25 else 10e6
        router.cost = cost
        routes = router.routes((-.6, 0), (.6, .2))
        self.assertEqual(len(routes), 2)
        direct, detour = routes
        self.assertGreater(detour['km'], direct['km'])
        self.assertLess(detour['track_capital'], direct['track_capital'])
        self.assertEqual(detour['alignment_kind'], 'terrain_any_angle')

    def test_search_limit_preserves_valid_direct_candidate(self):
        router = self.router()
        router.config.terrain_search_limit = 1
        routes = router.routes((-.6, 0), (.6, .2))
        self.assertEqual([r['alignment_kind'] for r in routes], ['direct'])

    def test_direct_is_rejected_when_sampled_blocked_terrain_crossed(self):
        router = self.router()
        original = router.allowed
        router.allowed = lambda cell: original(cell) and not (abs(h3.cell_to_latlng(cell)[1]) < .15 and abs(h3.cell_to_latlng(cell)[0]-.1) < .25)
        routes = router.routes((-.6, 0), (.6, .2))
        self.assertTrue(routes)
        self.assertNotIn('direct', [route['alignment_kind'] for route in routes])

    def test_solver_can_choose_speed_or_construction_saving(self):
        for cost_weight, expected_edge in [(0, 0), (100, 1)]:
            with self.subTest(cost_weight=cost_weight):
                p = example(weight=0)
                p.config.maximum_shared_track_lines = 0
                p.config.cost_weight = cost_weight
                p.config.minimum_served_fraction = 1
                p.config.station_capital_cost = 0
                p.config.station_annual_cost = 0
                p.config.train_capital_cost = 0
                p.config.train_cost_per_km = 0
                p.config.maintenance_fraction = 0
                direct = dict(p.edges[0], km=10, track_capital=1e9, alignment_kind='direct')
                detour = dict(direct, id=1, km=30, track_capital=1e6, alignment_kind='terrain_any_angle')
                p.edges = [direct, detour]
                model = NetworkModel(p)
                self.assertEqual({tuple(l['edges']) for l in model.lines}, {(0,), (1,)})
                solver = model.solver(p.config, 10)
                solver.run()
                report = model.describe(solver.getSolution().col_value)
                self.assertEqual([t['id'] for t in report['tracks']], [expected_edge])
                self.assertEqual(len(report['lines']), 1)
