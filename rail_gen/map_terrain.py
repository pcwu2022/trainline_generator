"""Fixed mountain-only map backgrounds, priced without urban/water surcharges."""

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


def fast_terrain(config):
    if config.get('terrain_kind', 'flat') == 'flat' or not config.get('terrain_file'):
        return None
    from rail_gen.alignment import cost_surfaces
    from rail_gen.prep import TO_WGS84
    # A zero-density sample isolates terrain costs from population surcharges.
    rate, water, grade, transform = cost_surfaces(
        config, (np.array([[0., 0.]]), None, np.array([0.])), terrain_only=True)
    baseline = config['tracks']['CR']['base_meur_km']
    slope_based = config['terrain_kind'] == 'dem' or bool(config.get('terrain_dem_file'))
    threshold = baseline * (2.5 if slope_based else 1.5)
    mask = ~water & np.isfinite(rate) & (rate > threshold)
    if slope_based:
        mask &= grade >= .02
    polygons, costs = [], []
    for row, col in zip(*np.nonzero(mask)):
        corners = [transform * (col + x, row + y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
        lon, lat = TO_WGS84.transform(*np.asarray(corners).T)
        polygons.append(np.column_stack((lon, lat)))
        costs.append(rate[row, col])
    return polygons, costs, threshold, baseline * 12


def add_terrain(ax, figure, terrain, project):
    if terrain is None or not terrain[0]:
        return
    polygons, costs, threshold, maximum = terrain
    cmap = LinearSegmentedColormap.from_list('mountain_cost', ['#c69c72', '#4b2412'])
    collection = PolyCollection([project(p) for p in polygons], array=np.asarray(costs),
        cmap=cmap, norm=Normalize(threshold, max(maximum, max(costs))),
        edgecolors='none', zorder=1.5)
    ax.add_collection(collection)
    axis = figure.add_axes([.85, .81, .025, .10])
    figure.colorbar(collection, cax=axis, label='Conventional terrain (M€/km)')
