import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

import h3
import numpy as np

from test_heuristic import problem
from train_optimizer.annealing import RoutePool, evaluate
from train_optimizer.config import Config
from train_optimizer.geography import Router


class WaterCrossingTests(unittest.TestCase):
    def test_explicit_water_cells_are_priced_and_reported_without_double_counting(self):
        cell = h3.latlng_to_cell(51, 4, 6)
        lat, lon = h3.cell_to_latlng(cell)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'water.json'
            path.write_text(json.dumps(dict(h3_resolution=6, cells={cell: dict(terrain_class='water', multiplier=1.5)})))
            config = Config(terrain_file=str(path))
            router = Router(config, [dict(h3=cell, density=0)])
            self.assertEqual(router.cost(cell), 180e6)
            water = router.alignment([(lon-.001, lat), (lon+.001, lat)])
            self.assertAlmostEqual(water['water_km'], water['km'])
            self.assertAlmostEqual(water['water_capital'], water['track_capital'])
            self.assertAlmostEqual(water['track_capital'], water['km']*180e6*math.exp(water['km']/25))
            router.terrain = {}
            land = router.alignment([(lon-.001, lat), (lon+.001, lat)])
            self.assertEqual(land['water_km'], 0)
            self.assertEqual(land['water_capital'], 0)
            self.assertAlmostEqual(water['track_capital']/land['track_capital'], 18*math.exp(water['km']/25))

    def test_raster_alignment_prices_the_midpoint_instead_of_the_h3_center(self):
        cell = h3.latlng_to_cell(51, 4, 6)
        lat, lon = h3.cell_to_latlng(cell)
        router = Router(Config(), [dict(h3=cell, density=0)])
        raster = Mock()
        raster.terrain_class.return_value = 'water'
        raster.cost_per_km.return_value = 200e6
        router.terrain_map = raster
        # A center-based lookup would silently price this synthetic water as land.
        router.cost = Mock(return_value=10e6)
        result = router.alignment([(lon-.001, lat), (lon+.001, lat)])
        router.cost.assert_not_called()
        self.assertAlmostEqual(result['track_capital'], result['km']*200e6*math.exp(result['km']/25))
        raster.cost_per_km.side_effect = ValueError('Outside actual raster coverage')
        self.assertIsNone(router.alignment([(lon-.001, lat), (lon+.001, lat)]))

    def test_cost_aware_proposals_reduce_the_weight_of_an_equally_busy_water_link(self):
        p = problem()
        p.cities.append(dict(id=2, name='Other'))
        p.locations.append(dict(p.locations[1], id=2, city=2, coordinate=(-1, 51)))
        p.demand = [dict(id=0, origin=0, destination=1, volume=100, km=70),
                    dict(id=1, origin=0, destination=2, volume=100, km=70)]
        p.config.terrain_file = 'synthetic.tiff'
        p.cells = [dict(h3=h3.latlng_to_cell(51, 0, 6), density=0)]
        pool = RoutePool(p)
        pool.router = Mock()
        pool.router.terrain_map.cost_per_km.side_effect = lambda lon, lat: 200e6 if lon > 0 else 10e6
        weights = pool.demand_weights()
        self.assertAlmostEqual(weights[1]/weights[0], 20, places=4)
        calls = pool.router.terrain_map.cost_per_km.call_count
        np.testing.assert_equal(weights, pool.demand_weights())
        self.assertEqual(pool.router.terrain_map.cost_per_km.call_count, calls)
        pool.router.routes.assert_not_called()
        p.config.annealing_proposal_cost_exponent = 0
        raw = RoutePool(p).demand_weights()
        np.testing.assert_equal(raw, [100, 100])

    def test_shared_water_track_is_counted_once_in_network_metrics(self):
        p = problem()
        p.cities.append(dict(id=2, name='Third'))
        p.locations.append(dict(p.locations[1], id=2, city=2, coordinate=(2, 51)))
        p.stations.append(dict(p.stations[1], id=2, city=2, location=2))
        p.edges[0].update(water_km=5, water_capital=200e6)
        p.edges.append(dict(p.edges[0], id=1, a=1, b=2, water_km=0, water_capital=0))
        design = (((0, 1), (0,), 1), ((0, 1, 2), (0, 1), 1))
        model, vector, *_ = evaluate(p, RoutePool(p), design, 1, lambda: 5, threading.Event(), final=True)
        metrics = model.describe(vector)['metrics']
        self.assertEqual(metrics['total_water_track_km'], 5)
        self.assertEqual(metrics['water_track_capital'], 200e6)

    def test_crossing_growth_and_vertex_invariance(self):
        cell = h3.latlng_to_cell(51, 4, 6)
        router = Router(Config(), [dict(h3=cell, density=0)])
        router.terrain_map = Mock()
        router.terrain_map.terrain_class.return_value = 'water'
        router.terrain_map.cost_per_km.return_value = 120e6
        short = router.alignment([(4, 51), (4.01, 51)])
        long = router.alignment([(4, 51), (5, 51)])
        split = router.alignment([long['geometry'][0], long['geometry'][len(long['geometry'])//2], long['geometry'][-1]])
        self.assertAlmostEqual(long['track_capital'], split['track_capital'], delta=1)
        self.assertGreater(long['track_capital']/long['km'], short['track_capital']/short['km']*10)

    def test_land_resets_separate_crossings(self):
        cell = h3.latlng_to_cell(51, 4, 6)
        router = Router(Config(), [dict(h3=cell, density=0)])
        raster = router.terrain_map = Mock()
        raster.terrain_class.side_effect = lambda lon, lat: 'normal' if 4.02 <= lon < 4.04 else 'water'
        raster.cost_per_km.side_effect = lambda lon, lat, density: 10e6 if 4.02 <= lon < 4.04 else 120e6
        a, b, c, d = [(lon, 51) for lon in (4, 4.02, 4.04, 4.06)]
        whole = router.alignment([a, b, c, d])
        pieces = [router.alignment([x, y]) for x, y in ((a, b), (b, c), (c, d))]
        self.assertAlmostEqual(whole['track_capital'], sum(p['track_capital'] for p in pieces), delta=1)
        self.assertAlmostEqual(whole['water_capital'], sum(p['water_capital'] for p in pieces), delta=1)

    def test_new_cost_parameters_are_validated(self):
        for values in [dict(water_cost_growth_km=0), dict(water_routing_sample_km=0),
                       dict(terrain_cost_ranges_million={"bridge": [20, 50]}),
                       dict(water_track_cost_per_km=0), dict(water_track_cost_per_km=-1),
                       dict(annealing_proposal_cost_exponent=-1), dict(annealing_proposal_cost_exponent=5)]:
            with self.assertRaises(ValueError):
                Config.from_dict(values)
