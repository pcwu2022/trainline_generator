"""Analytic gradient, continuous costs and missing-data regression checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin

from rail_gen.alignment import cost_surfaces
from rail_gen.config import load_config
from train_optimizer.settings import Config
from train_optimizer.slope import elevation_gradient, slope_cost_million, slope_multiplier
from train_optimizer.terrain import TerrainCostMap


class SlopeTests(unittest.TestCase):
    def test_curve_is_continuous_monotone_and_matches_cost_anchors(self):
        config = Config()
        grade = np.array([0, .01, .02, .05, .10, .20])
        np.testing.assert_allclose(slope_multiplier(grade), [1, 1.5, 2.5, 4, 7, 12])
        np.testing.assert_allclose(slope_cost_million(grade, 10, config.terrain_cost_ranges_million),
                                   [10, 15, 25, 40, 70, 120])
        around_boundary = slope_multiplier(np.array([.02 - 1e-8, .02, .02 + 1e-8]))
        self.assertLess(np.ptp(around_boundary), 1e-5)
        self.assertTrue(np.all(np.diff(slope_multiplier(np.linspace(0, .4, 1000))) >= 0))

    def test_planar_slope_is_rise_over_run_and_nodata_stays_unknown(self):
        elevation = np.tile(np.arange(8) * 60., (8, 1))
        np.testing.assert_allclose(elevation_gradient(elevation, 3000), .02)
        elevation[3, 3] = np.nan
        grade = elevation_gradient(elevation, 3000)
        self.assertTrue(np.isnan(grade[3, 3]))
        self.assertTrue(np.isnan(grade[3, 2]))
        self.assertAlmostEqual(grade[6, 6], .02)

    def test_relief_water_is_preserved_and_land_gets_dem_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c = Config.from_dict(dict(bbox=[7, 45, 9, 47]))
            to_metric = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)
            x, y = to_metric.transform(8, 46)
            transform = from_origin(x-60000, y+60000, 3000, 3000)
            path = root / 'dem.tif'
            with rasterio.open(path, 'w', driver='GTiff', count=1, width=40, height=40,
                               dtype='float32', crs='EPSG:3035', transform=transform) as dst:
                dst.write(np.tile(np.arange(40) * 90., (40, 1)).astype('float32'), 1)
            c.terrain_dem_file = str(path)
            terrain = TerrainCostMap(config=c)
            self.assertEqual(terrain.terrain_class(8, 46), 'moderate_mountains')
            self.assertAlmostEqual(terrain(8, 46), 30e6, delta=1e5)
            # Water classification bypasses missing DEM coverage.
            self.assertEqual(terrain(3, 54), c.water_track_cost_per_km)
            with self.assertRaisesRegex(ValueError, 'slope coverage'):
                terrain(4.9, 52.37)

    def test_fast_surface_and_map_use_the_same_continuous_cost(self):
        c = load_config(profile=None, overrides=dict(terrain_kind='flat', terrain_file='',
            boundaries_file='', countries=[], bbox=[7, 45, 9, 47]))
        samples = (np.array([[0., 0.]]), None, np.array([0.]))
        flat, _, _, transform = cost_surfaces(c, samples)
        height, width = flat.shape[1:]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'dem.tif'
            with rasterio.open(path, 'w', driver='GTiff', count=1, width=width, height=height,
                               dtype='float32', crs='EPSG:3035', transform=transform) as dst:
                dst.write(np.tile(np.arange(width) * 45., (height, 1)).astype('float32'), 1)
            c.update(terrain_kind='dem', terrain_file=str(path))
            rates, _, grade, _ = cost_surfaces(c, samples, terrain_only=True)
            np.testing.assert_allclose(grade, .015, atol=1e-6)
            np.testing.assert_allclose(rates, c['tracks']['CR']['base_meur_km'] * 2., rtol=1e-5)
            land = Path(temporary) / 'land.tif'
            with rasterio.open(land, 'w', driver='GTiff', count=1, width=width, height=height,
                               dtype='uint8', crs='EPSG:3035', transform=transform) as dst:
                dst.write(np.zeros((height, width), dtype='uint8'), 1)
            c['land_class_file'] = str(land)
            routes, _, _, _ = cost_surfaces(c, samples)
            np.testing.assert_allclose(routes[0], rates)
            np.testing.assert_allclose(routes[1] / routes[0],
                                       c['tracks']['HSR']['base_meur_km'] / c['tracks']['CR']['base_meur_km'])
