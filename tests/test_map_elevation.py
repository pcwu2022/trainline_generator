"""Missing population cells get DEM means without modifying routing inputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import h3
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

from rail_gen.map_elevation import elevation_hexagons


class ElevationMapTests(unittest.TestCase):
    def test_missing_hexagons_use_mean_valid_elevations(self):
        occupied = h3.latlng_to_cell(24, 121, 6)
        boundary = Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h3.cell_to_parent(occupied, 4))])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = root/'population.json'
            population.write_text(json.dumps({'features': [{'properties': {'h3Index': occupied}}]}))
            land = root/'land.json'
            land.write_text(json.dumps({'features': [{'properties': {'iso_a2': 'TW'}, 'geometry': mapping(boundary)}]}))
            dem = root/'dem.tif'
            with rasterio.open(dem, 'w', driver='GTiff', width=200, height=200,
                               count=1, dtype='float32', crs='EPSG:4326',
                               transform=from_origin(120, 25, .01, .01), nodata=-9999) as dst:
                values = np.full((200, 200), 750, dtype='float32')
                values[::2, ::2] = -9999
                dst.write(values, 1)
            config = dict(terrain_file=str(dem), terrain_kind='etopo', population_file=str(population),
                          boundaries_file=str(land), countries=['TW'], bbox=[120, 23, 122, 25])
            original = copy.deepcopy(config)
            polygons, means = elevation_hexagons(config, lambda xy: xy)
            self.assertTrue(polygons)
            np.testing.assert_allclose(means, 750)
            from shapely.geometry import Point
            center = Point(121, 24)
            self.assertTrue(all(not Polygon(p).covers(center) for p in polygons))
            self.assertEqual(config, original)
