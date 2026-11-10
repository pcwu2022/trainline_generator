import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import types
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

from fetch_terrain import build_mosaic, fetch_terrain


class TerrainTests(unittest.TestCase):
    def test_download_mosaic_and_metric_average(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            # Adjacent tiles: known 2x2 averages, including a nodata pixel.
            for index, values in enumerate(([[10, 10], [10, -9999]], [[30, 30], [30, 30]])):
                path = root / f'DEM_{index}.tif'
                with rasterio.open(path, 'w', driver='GTiff', width=2, height=2,
                                   count=1, dtype='float32', crs='EPSG:3035',
                                   transform=from_origin(4000000 + index*3000, 3000000, 1500, 1500),
                                   nodata=-9999) as dst:
                    dst.write(np.array(values, dtype='float32'), 1)
                paths.append(path)
            mosaic = root / 'test.vrt'
            build_mosaic(paths, mosaic)
            with rasterio.open(mosaic) as src:
                self.assertEqual(src.shape, (2, 4))
                self.assertTrue(src.read(1, masked=True).mask[1, 1])
            class Gateway:
                def search_all(self, **kwargs):
                    self.criteria = kwargs
                    return [1, 2]
                def download_all(self, products, **kwargs):
                    return [str(p) for p in paths]
            with patch.dict(sys.modules, {'eodag': types.SimpleNamespace(EODataAccessGateway=Gateway)}):
                outputs = fetch_terrain([-10, 35, 25, 65], root/'downloads', root/'outputs', [3])
            with rasterio.open(outputs[0]) as src:
                self.assertEqual(src.crs.to_epsg(), 3035)
                self.assertEqual(src.res, (3000, 3000))
                np.testing.assert_allclose(src.read(1), [[10, 30]])
            with self.assertRaises(FileExistsError):
                fetch_terrain([-10, 35, 25, 65], root/'downloads', root/'outputs', [3])

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            fetch_terrain([25, 35, -10, 65], 'unused', 'unused')


if __name__ == '__main__':
    unittest.main()
