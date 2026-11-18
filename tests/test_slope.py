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
from rail_gen.slope import elevation_gradient, slope_multiplier


class SlopeTests(unittest.TestCase):
    def test_curve_is_continuous_monotone_and_matches_cost_anchors(self):
        grade = np.array([0, .01, .02, .05, .10, .20])
        np.testing.assert_allclose(slope_multiplier(grade), [1, 1.2, 1.5, 2, 3, 5])
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
            np.testing.assert_allclose(rates, c['tracks']['CR']['base_meur_km'] * 1.35, rtol=1e-5)
            land = Path(temporary) / 'land.tif'
            with rasterio.open(land, 'w', driver='GTiff', count=1, width=width, height=height,
                               dtype='uint8', crs='EPSG:3035', transform=transform) as dst:
                dst.write(np.zeros((height, width), dtype='uint8'), 1)
            c['land_class_file'] = str(land)
            routes, _, _, _ = cost_surfaces(c, samples)
            np.testing.assert_allclose(routes[0], rates)
            np.testing.assert_allclose(routes[1] / routes[0],
                                       c['tracks']['HSR']['base_meur_km'] / c['tracks']['CR']['base_meur_km'])
