"""Continuous, explicitly provisional ground-slope construction difficulty.

Slope is rise/run, not degrees or the railway's longitudinal gradient.
Anchors: 0%, 1%, 2%, 5%, 10%, 20%; interpolation avoids class-boundary jumps.
"""
import math

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.warp import reproject, Resampling, transform_bounds

SLOPE_ANCHORS = np.array([0, .01, .02, .05, .10, .20])
COST_MULTIPLIERS = np.array([1, 1.5, 2.5, 4, 7, 12])


def slope_multiplier(grade):
    return np.interp(grade, SLOPE_ANCHORS, COST_MULTIPLIERS, left=1, right=12)


def slope_cost_million(grade, baseline, ranges):
    anchors = [baseline, ranges['hilly'][0], ranges['moderate_mountains'][0],
               ranges['moderate_mountains'][1], ranges['difficult_mountains'][1],
               ranges['very_difficult'][1]]
    # Unusual scenario ranges must never make steeper terrain cheaper.
    anchors = np.maximum.accumulate(anchors)
    return np.interp(grade, SLOPE_ANCHORS, anchors)


def slope_class(grade):
    return ('normal' if grade < .01 else 'hilly' if grade < .02 else
            'moderate_mountains' if grade < .05 else
            'difficult_mountains' if grade < .10 else 'very_difficult')


def elevation_gradient(elevation, step):
    """Propagate missing elevation through gradients; never invent flat cells."""
    if min(elevation.shape) < 2:
        raise ValueError('DEM must contain at least two rows and columns')
    gy, gx = np.gradient(elevation, step)
    grade = np.hypot(gx, gy)
    grade[~np.isfinite(elevation)] = np.nan
    return grade


class ElevationSlopeMap:
    """Read a bounded metric DEM window; slope sampling never reopens a raster."""
    def __init__(self, path, bbox, cell_km=3):
        step = cell_km * 1000
        left, bottom, right, top = transform_bounds('EPSG:4326', 'EPSG:3035', *bbox, densify_pts=41)
        # Candidate catchments/corridors can extend beyond the selected bbox.
        padding = 100000
        left, bottom, right, top = left-padding, bottom-padding, right+padding, top+padding
        width, height = math.ceil((right-left)/step), math.ceil((top-bottom)/step)
        self.transform = from_origin(left, top, step, step)
        elevation = np.full((height, width), np.nan, dtype='float32')
        with rasterio.open(path) as src:
            if src.count != 1 or src.crs is None:
                raise ValueError('Slope terrain requires a georeferenced single-band elevation DEM in metres')
            reproject(rasterio.band(src, 1), elevation, src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=self.transform,
                      dst_crs='EPSG:3035', dst_nodata=np.nan, resampling=Resampling.average)
        self.grade = elevation_gradient(elevation, step)
        self.to_metric = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)

    def sample(self, lon, lat):
        x, y = self.to_metric.transform(lon, lat)
        row, col = rowcol(self.transform, x, y)
        if (not 0 <= row < self.grade.shape[0] or not 0 <= col < self.grade.shape[1]
                or not np.isfinite(self.grade[row, col])):
            raise ValueError(f'No elevation/slope coverage at ({lon}, {lat})')
        return float(self.grade[row, col])
