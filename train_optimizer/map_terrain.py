"""Fixed mountain-only map backgrounds, priced without urban/water surcharges."""
from functools import lru_cache
import json

import h3
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


@lru_cache(maxsize=4)
def _legacy_tiles(serialized, resolution):
    from .geography import Router
    from .settings import Config
    config = Config.from_dict(json.loads(serialized))
    west, south, east, north = config.bbox
    seed = h3.latlng_to_cell((south + north) / 2, (west + east) / 2, resolution)
    router = Router(config, [dict(h3=seed, density=0)])
    threshold = config.terrain_cost_ranges_million['moderate_mountains'][0] * 1e6
    maximum = max(high for low, high in config.terrain_cost_ranges_million.values()) * 1e6
    tiles = h3.polygon_to_cells(h3.LatLngPoly(
        [(south, west), (south, east), (north, east), (north, west)]), resolution)
    polygons, costs = [], []
    for tile in sorted(tiles):
        if not router.allowed(tile):
            continue
        lat, lon = h3.cell_to_latlng(tile)
        kind = (router.terrain_map.terrain_class(lon, lat) if router.terrain_map
                else router.terrain.get(tile, {}).get('terrain_class', 'normal'))
        if kind not in {'moderate_mountains', 'difficult_mountains', 'very_difficult'}:
            continue
        cost = router.cost(tile)
        if cost > threshold:
            polygons.append([(lon, lat) for lat, lon in h3.cell_to_boundary(tile)])
            costs.append(cost / 1e6)
    return polygons, costs, threshold / 1e6, maximum / 1e6


def legacy_terrain(config, cells):
    if not getattr(config, 'terrain_file', '') or not cells:
        return None
    from dataclasses import asdict
    return _legacy_tiles(json.dumps(asdict(config), sort_keys=True),
                         h3.get_resolution(cells[0]['h3']))


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
