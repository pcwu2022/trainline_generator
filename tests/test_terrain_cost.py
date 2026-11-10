import tempfile
import unittest
from pathlib import Path

import h3
import numpy as np
import rasterio
from rasterio.transform import from_origin

from train_optimizer.config import Config
from train_optimizer.geography import Router
from train_optimizer.terrain import TerrainCostMap


class TerrainCostTests(unittest.TestCase):
    def test_density_threshold_without_raster(self):
        config = Config(urban_density_threshold=5000)
        cell = h3.latlng_to_cell(51, 4, 6)
        router = Router(config, [{'h3': cell, 'density': 5000}])
        self.assertEqual(router.cost(cell), 10_000_000)
        router.density[cell] = 6000
        self.assertEqual(router.cost(cell), 15_000_000)
        config.urban_density_threshold = 6000
        self.assertEqual(router.cost(cell), 10_000_000)
        with self.assertRaises(ValueError):
            Config.from_dict({'urban_density_threshold': -1})

    def test_classes_costs_and_router(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'relief.tiff'
            rgb = np.array([[[100, 115, 65], [80, 135, 180], [150, 125, 90]]], dtype='uint8')
            with rasterio.open(path, 'w', driver='GTiff', width=3, height=1,
                               count=3, dtype='uint8', crs='EPSG:4326',
                               transform=from_origin(0, 1, 1, 1)) as dst:
                dst.write(rgb.transpose(2, 0, 1))
            config = Config(terrain_file=str(path))
            terrain = TerrainCostMap(path, config)
            self.assertEqual([terrain.terrain_class(x, .5) for x in [.5, 1.5, 2.5]],
                             ['normal', 'bridge', 'moderate_mountains'])
            self.assertEqual([terrain(x, .5) for x in [.5, 1.5, 2.5]],
                             [10_000_000, 35_000_000, 32_500_000])
            self.assertEqual(terrain(.5, .5, density=100), 10_000_000)
            self.assertEqual(terrain(.5, .5, density=5000), 10_000_000)
            self.assertEqual(terrain(.5, .5, density=6000), 15_000_000)
            self.assertEqual(terrain(.5, .5, density=7000), 20_000_000)
            cell = h3.latlng_to_cell(.5, 1.5, 6)
            router = Router(config, [{'h3': cell, 'density': 100}])
            self.assertTrue(router.allowed(cell))
            self.assertEqual(router.cost(cell), 35_000_000)
            outside = h3.latlng_to_cell(30, 30, 6)
            self.assertFalse(router.allowed(outside))
            for lon, lat in [(3, .5), (-1, .5), (float('nan'), .5), (0, 91)]:
                with self.assertRaises(ValueError):
                    terrain(lon, lat)
            with self.assertRaises(ValueError):
                terrain(.5, .5, density=-1)
            config.terrain_cost_position = 1
            self.assertEqual(terrain(1.5, .5), 50_000_000)

    def test_supplied_map_land_water_and_mountains(self):
        terrain = TerrainCostMap()
        self.assertEqual(terrain(4.9, 52.37), 10_000_000)
        self.assertEqual(terrain(3, 54), 35_000_000)
        self.assertEqual(terrain(8, 46.5), 32_500_000)


if __name__ == '__main__':
    unittest.main()
