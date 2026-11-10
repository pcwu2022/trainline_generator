"""Population-only market seeds and optional geographic labels."""
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree


def population_seeds(cells, config):
    """Rank eligible cell centers by population within a metric neighborhood.

    Greedily retain geographically separated centers. H3 IDs break ties so
    feature order and city-label records cannot affect the network candidates.
    """
    from .geography import haversine
    candidates = sorted((c for c in cells if c['density'] >= config.density_threshold
                         and c['population'] > 0), key=lambda c: c['h3'])
    if not candidates:
        return []
    w, s, e, n = config.bbox
    projection = Transformer.from_crs('EPSG:4326',
        f'+proj=aeqd +lat_0={(s+n)/2} +lon_0={(w+e)/2} +datum=WGS84 +units=m', always_xy=True)
    xy = np.array([projection.transform(c['lon'], c['lat']) for c in candidates]) / 1000
    tree = cKDTree(xy)
    population = np.array([c['population'] for c in candidates])
    scores = [float(population[tree.query_ball_point(point, config.market_radius_km)].sum())
              for point in xy]
    seeds = []
    for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]['h3'])):
        if scores[i] < config.minimum_city_population:
            continue
        cell = candidates[i]
        point = (cell['lon'], cell['lat'])
        if all(haversine(point, seed['coordinate']) >= config.market_radius_km for seed in seeds):
            seeds.append(dict(name=f'Market {len(seeds)+1}', country='', coordinate=point,
                              seed_population=scores[i], seed_h3=cell['h3']))
            if len(seeds) >= config.max_cities:
                break
    return seeds


def label_markets(cities, path, maximum_distance_km=30):
    """Attach nearest-city labels after candidate positions are finalized."""
    from .geography import haversine
    if not path or not Path(path).exists():
        return
    records = json.loads(Path(path).read_text())
    if not records:
        return
    for city in cities:
        nearest = min(records, key=lambda r: (haversine(city['center'], (float(r['lon']), float(r['lat']))), r['name']))
        distance = haversine(city['center'], (float(nearest['lon']), float(nearest['lat'])))
        if distance <= maximum_distance_km:
            city.update(name=nearest['name'], country=nearest.get('country', ''),
                        label_distance_km=distance)
    # Separate markets can legitimately be near the same named city.
    names = [c['name'] for c in cities]
    for city in cities:
        if names.count(city['name']) > 1:
            city['name'] += f" (market {city['id']+1})"


def market_membership(sample_xy, seed_xy, density, config):
    """Nearest-seed ownership, capped at the unified market radius (metric km)."""
    distance, owner = cKDTree(seed_xy).query(sample_xy)
    eligible = (np.asarray(density) >= config.density_threshold) & (distance <= config.market_radius_km)
    return owner, eligible
