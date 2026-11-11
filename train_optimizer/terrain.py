"""Approximate rail costs from RGB relief and an optional elevation-slope overlay.

Relief-only input is a visual classifier; the DEM overlay adds regional ground slopes.
Coordinates are WGS84 (longitude, latitude); costs are currency units per km.
"""
from functools import lru_cache
import math

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import rowcol
from .slope import ElevationSlopeMap, slope_class, slope_cost_million


class TerrainCostMap:
    def __init__(self, path='data/exportImage.tiff', config=None):
        if config is None:
            from .settings import Config
            config = Config()
        self.config = config
        config.validate()
        with rasterio.open(path) as src:
            if src.count != 3 or src.crs is None or any(d != 'uint8' for d in src.dtypes):
                raise ValueError('Expected a georeferenced three-band uint8 RGB relief map')
            self.rgb = src.read()
            self.valid = np.all(src.read_masks() != 0, axis=0)
            self.transform = src.transform
            self.to_raster = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        self.height, self.width = self.rgb.shape[1:]
        self.slope_map = (ElevationSlopeMap(config.terrain_dem_file, config.bbox)
                          if config.terrain_dem_file else None)

    @lru_cache(maxsize=100000)
    def terrain_class(self, longitude, latitude):
        """Classify the map pixel. Unknown coverage raises rather than assuming flat land.

        Blue-dominant pixels represent mapped water (including large lakes).
        A configured DEM overrides land difficulty using metric ground slope.
        Without it, brown pixels represent moderate mountains and other valid
        pixels use normal cost. RGB thresholds are heuristic for exportImage.tiff.
        """
        if not (math.isfinite(longitude) and math.isfinite(latitude)
                and -180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError('Expected finite WGS84 longitude, latitude')
        x, y = self.to_raster.transform(longitude, latitude)
        row, col = rowcol(self.transform, x, y)
        if not (0 <= row < self.height and 0 <= col < self.width) or not self.valid[row, col]:
            raise ValueError(f'No terrain coverage at ({longitude}, {latitude})')
        r, g, b = (int(v) for v in self.rgb[:, row, col])
        if b > r * 1.12 and b > g * 1.04:
            return 'water'
        if self.slope_map is not None:
            return slope_class(self.slope_map.sample(longitude, latitude))
        if r > g * 1.05 and r > b * 1.15:
            return 'moderate_mountains'
        return 'normal'

    def cost_per_km(self, longitude, latitude, density=0):
        """Return local baseline/km with urban surcharge; Router adds crossing-length growth."""
        if not math.isfinite(density) or density < 0:
            raise ValueError('Population density must be finite and nonnegative')
        terrain = self.terrain_class(longitude, latitude)
        c = self.config
        baseline = c.base_track_cost_per_km
        if terrain == 'water':
            baseline = max(baseline, c.water_track_cost_per_km)
        elif self.slope_map is not None:
            baseline = float(slope_cost_million(
                self.slope_map.sample(longitude, latitude), baseline / 1e6,
                c.terrain_cost_ranges_million)) * 1e6
        elif terrain != 'normal':
            low, high = c.terrain_cost_ranges_million[terrain]
            baseline = max(baseline, (low + c.terrain_cost_position * (high-low)) * 1_000_000)
        return baseline + c.urban_cost_coefficient * max(0, density - c.urban_density_threshold)

    __call__ = cost_per_km
