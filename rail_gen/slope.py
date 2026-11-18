"""Continuous ground-slope construction difficulty."""
import numpy as np

SLOPE_ANCHORS = np.array([0, .01, .02, .05, .10, .20])
COST_MULTIPLIERS = np.array([1, 1.2, 1.5, 2, 3, 5])

def slope_multiplier(grade):
    return np.interp(grade, SLOPE_ANCHORS, COST_MULTIPLIERS, left=1, right=5)


def elevation_gradient(elevation, step):
    """Propagate missing elevation through gradients; never invent flat cells."""
    if min(elevation.shape) < 2:
        raise ValueError('DEM must contain at least two rows and columns')
    gy, gx = np.gradient(elevation, step)
    grade = np.hypot(gx, gy)
    grade[~np.isfinite(elevation)] = np.nan
    return grade

