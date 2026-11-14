"""Mountain shading excludes expensive water, hills and urban surcharges."""
import json
from pathlib import Path
import tempfile
import unittest

import h3
import numpy as np
from matplotlib.colors import to_rgba

from train_optimizer.map_style import satisfaction_scale
from train_optimizer.map_terrain import legacy_terrain
from train_optimizer.settings import Config


class MountainBackgroundTests(unittest.TestCase):
    def test_full_satisfaction_is_dark_green(self):
        cmap, norm = satisfaction_scale()
        np.testing.assert_allclose(cmap(norm(1.0)), to_rgba('#124625'))

    def test_only_mountains_over_threshold_are_shaded_by_baseline_cost(self):
        tiles = sorted(h3.grid_disk(h3.latlng_to_cell(46, 8, 5), 1))
        kinds = ['water', 'hilly', 'moderate_mountains', 'difficult_mountains', 'very_difficult']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'terrain.json'
            path.write_text(json.dumps(dict(h3_resolution=5, cells={
                tile: dict(terrain_class=kind) for tile, kind in zip(tiles, kinds)})))
            config = Config.from_dict(dict(bbox=[7, 45, 9, 47], terrain_file=str(path)))
            polygons, costs, threshold, maximum = legacy_terrain(
                config, [dict(h3=tiles[0], density=100000)])
            self.assertEqual(len(polygons), 3)
            expected = [sum(config.terrain_cost_ranges_million[kind]) / 2 for kind in kinds[2:]]
            np.testing.assert_allclose(sorted(costs), sorted(expected))
            self.assertTrue(all(threshold < cost <= maximum for cost in costs))
