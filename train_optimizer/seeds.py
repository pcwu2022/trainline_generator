"""Population-discovered market seeds with marginal coverage."""
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from rail_model import center_radius


def population_seeds(cells, config):
    """Greedy marginal population coverage with independent seed separation.

    All candidates come from population cells; city records only supply labels.
    """
    from .geography import haversine
    eligible = sorted((c for c in cells if c['density'] >= config.density_threshold
                       and c['population'] > 0), key=lambda c: c['h3'])
    if not eligible:
        return []
    w, s, e, n = config.bbox
    projection = Transformer.from_crs('EPSG:4326',
        f'+proj=aeqd +lat_0={(s+n)/2} +lon_0={(w+e)/2} +datum=WGS84 +units=m', always_xy=True)
    xy = np.array([projection.transform(c['lon'], c['lat']) for c in eligible]) / 1000
    tree = cKDTree(xy)
    population = np.array([c['population'] for c in eligible])
    candidates = [dict(coordinate=(c['lon'], c['lat']), name='', country='') for c in eligible]
    neighborhoods = []
    positions = []
    weights = []
    for candidate in candidates:
        point = np.asarray(projection.transform(*candidate['coordinate'])) / 1000
        positions.append(point)
        near = np.asarray(tree.query_ball_point(point, config.market_radius_km), dtype=int)
        neighborhoods.append(near)
        # Favor compact urban concentrations over circles spanning several towns.
        distance = np.linalg.norm(xy[near] - point, axis=1)
        weights.append(np.exp(-0.5 * (distance / (config.market_radius_km / 3))**2))
    totals = np.array([population[near].sum() for near in neighborhoods])
    scores = np.array([np.dot(population[near], weight)
                       for near, weight in zip(neighborhoods, weights)])
    available = totals >= config.minimum_city_population
    represented = np.zeros(len(eligible), dtype=bool)
    seeds = []
    spacing = config.minimum_seed_spacing_km or config.market_radius_km
    while len(seeds) < config.max_cities and available.any():
        # Re-score after every choice so overlapping metropolitan neighborhoods
        # do not repeatedly claim the same population as a new benefit.
        gains = [(float(np.dot(population[near] * (~represented[near]), weights[i]))
                  if available[i] and population[near[~represented[near]]].sum() >= config.minimum_city_population
                  else -1) for i, near in enumerate(neighborhoods)]
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        point = positions[i]
        near = neighborhoods[i]
        nearest = i  # Candidates and eligible cells share the same order.
        radius = center_radius(eligible[nearest]['density'], config.center_coefficient)
        seed = dict(candidates[i])
        seed.update(name=seed['name'] or f'Market {len(seeds)+1}',
                    seed_population=float(totals[i]), seed_score=float(scores[i]),
                    seed_marginal_population=gains[i],
                    seed_h3=eligible[nearest]['h3'], radius_km=radius)
        seeds.append(seed)
        represented[near] = True
        available[i] = False
        for j, candidate in enumerate(candidates):
            if available[j] and haversine(seed['coordinate'], candidate['coordinate']) < spacing:
                available[j] = False
    return seeds


def label_markets(cities, path, maximum_distance_km=30, countries=None):
    """Attach nearest-city labels after candidate positions are finalized."""
    from .geography import haversine
    if not path or not Path(path).exists():
        return
    records = json.loads(Path(path).read_text())
    if countries:
        records = [r for r in records if r.get('country') in countries]
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


def market_membership(sample_xy, seed_xy, density, config, radii=None):
    """Nearest-seed ownership, capped at each owner's radius (metric km)."""
    distance, owner = cKDTree(seed_xy).query(sample_xy)
    limits = config.market_radius_km if radii is None else np.asarray(radii)[owner]
    eligible = (np.asarray(density) >= config.density_threshold) & (distance <= limits + 1e-9)
    return owner, eligible
