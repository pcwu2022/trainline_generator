"""Metro nodes, exclusive station catchments and a small static candidate graph."""
from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import Delaunay, QhullError, cKDTree
from rail_model import center_radius

TO_METRIC = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)
TO_WGS84 = Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)


@dataclass
class City:
    id: str
    name: str
    lon: float
    lat: float
    population: float
    kappa: float = 0.0
    access_hours: float = 0.0
    country: str = ''
    radius_km: float = 0.0

    @property
    def xy(self):
        return np.array(TO_METRIC.transform(self.lon, self.lat)) / 1000


def distances(cities):
    """Great-circle distance matrix in km, independent of projected routing grid."""
    coordinates = np.radians([[c.lon, c.lat] for c in cities])
    lon, lat = coordinates.T
    h = np.sin((lat[:, None] - lat) / 2)**2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin((lon[:, None] - lon) / 2)**2
    return 12742.0176 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def read_cities(config, samples=None):
    path = Path(config['cities_file'])
    if path.suffix.lower() == '.csv':
        with path.open(newline='') as stream:
            records = list(csv.DictReader(stream))
    else:
        records = json.loads(path.read_text())
    nodes = []
    w, s, e, n = config['bbox']
    for index, r in enumerate(records):
        lon, lat = float(r['lon']), float(r['lat'])
        population = float(r.get('population', r.get('pop', 0)))
        if not all(np.isfinite([lon, lat, population])) or population < 0:
            raise ValueError(f'Invalid city record {index}')
        if not (w <= lon <= e and s <= lat <= n):
            continue
        if config['countries'] and r.get('country') not in config['countries']:
            continue
        if population < config['prep']['minimum_population']:
            continue
        nodes.append(City(str(r.get('id', index)), r['name'], lon, lat, population,
                          country=r.get('country', '')))
    nodes.sort(key=lambda c: (-c.population, c.id))
    if len(nodes) < 2 or len({c.id for c in nodes}) != len(nodes):
        raise ValueError('Need at least two cities with distinct IDs')
    if len({(c.lon, c.lat) for c in nodes}) != len(nodes):
        raise ValueError('City centres must have distinct coordinates; merge duplicate records')
    if samples is None:
        xy, _, density = population_samples(config)
    else:
        xy, _, density = samples
    set_capture_radii(nodes, xy, density, config)
    selected = []
    for city in nodes:
        # Keep the larger city when its metropolitan catchment contains a
        # smaller center. Only retained cities suppress later candidates.
        if any(np.linalg.norm(city.xy - other.xy) <= other.radius_km for other in selected):
            continue
        selected.append(city)
        if len(selected) == config['prep']['max_cities']:
            break
    if len(selected) < 2:
        raise ValueError('Need at least two cities outside retained metropolitan catchments')
    return selected


def set_capture_radii(cities, xy, density, config):
    """Use the same local density for city screening, capture and map circles."""
    _, nearest = cKDTree(xy).query([city.xy for city in cities])
    for city, index in zip(cities, nearest):
        city.radius_km = center_radius(float(density[index]), config['prep']['center_coefficient'])


def neighborhood_densities(features):
    """Smooth H3 density using 40% center and 10% per adjacent hexagon.

    Missing neighbors contribute zero; use the full input before bbox filtering.
    Non-H3 inputs retain their declared population/area density.
    """
    import h3
    raw = {}
    values = []
    for feature in features:
        p = feature['properties']
        count = float(p['population'])
        index = p.get('h3Index')
        area = h3.cell_area(index, unit='km^2') if index else float(p['area_km2'])
        if not np.isfinite(count) or count < 0:
            raise ValueError('Population cells must contain finite nonnegative population')
        if not np.isfinite(area) or area <= 0:
            raise ValueError('Population cell area must be positive')
        value = count / area
        values.append(value)
        if index:
            if index in raw:
                raise ValueError('Duplicate H3 population cells')
            raw[index] = value
    return np.asarray([
        .4 * value + .1 * sum(raw.get(neighbor, 0.0)
                             for neighbor in h3.grid_disk(index, 1) if neighbor != index)
        if (index := feature['properties'].get('h3Index')) else value
        for feature, value in zip(features, values)])


def population_samples(config):
    """Existing point/H3 GeoJSON, or polygon cells represented by their centroids."""
    import h3
    from shapely.geometry import shape
    features = json.loads(Path(config['population_file']).read_text())['features']
    smoothed_density = neighborhood_densities(features)
    xy, people, density = [], [], []
    w, s, e, n = config['bbox']
    for feature, local_density in zip(features, smoothed_density):
        geom, p = feature['geometry'], feature['properties']
        if geom['type'] == 'Point':
            lon, lat = geom['coordinates'][:2]
        else:
            point = shape(geom).centroid
            lon, lat = point.x, point.y
        if not (w <= lon <= e and s <= lat <= n):
            continue
        count = float(p['population'])
        if not np.isfinite(count) or count < 0:
            raise ValueError('Population cells must contain finite nonnegative population')
        area = h3.cell_area(p['h3Index'], unit='km^2') if 'h3Index' in p else float(p['area_km2'])
        if not np.isfinite(area) or area <= 0:
            raise ValueError('Population cell area must be positive')
        xy.append(np.array(TO_METRIC.transform(lon, lat)) / 1000)
        people.append(count)
        density.append(local_density)
    if not xy:
        raise ValueError('Population input has no cells within the bounding box')
    return np.asarray(xy), np.asarray(people), np.asarray(density)


def station_capture(cities, xy, people, density, config):
    """Assign samples exclusively; let the raster catchment exceed city-proper P."""
    p = config['prep']
    set_capture_radii(cities, xy, density, config)
    distance, owner = cKDTree([c.xy for c in cities]).query(xy)
    for i, city in enumerate(cities):
        included = (owner == i) & (distance <= city.radius_km)
        total = float(people[included].sum())
        # kappa is a multiplier relative to the city-file population, not a
        # probability: metropolitan catchments can exceed city-proper totals.
        city.kappa = total / city.population * p['uptake']
        city.access_hours = float(np.average(distance[included], weights=people[included])) / p['access_speed_kmh'] if total else 0.0
    return cities


def candidate_pairs(cities, config):
    p = config['prep']
    d = distances(cities)
    n = len(cities)
    pairs = set()
    for a in range(n):
        for b in np.argsort(d[a])[1:p['neighbors'] + 1]:
            if d[a, b] <= p['max_edge_km']:
                pairs.add(tuple(sorted((a, int(b)))))
    xy = np.array([c.xy for c in cities])
    if n >= 3:
        try:
            for triangle in Delaunay(xy).simplices:
                for a, b in zip(triangle, np.roll(triangle, 1)):
                    if d[a, b] <= p['max_edge_km']:
                        pairs.add(tuple(sorted((int(a), int(b)))))
        except QhullError:
            pass  # Collinear cities still have nearest-neighbour and forced edges.
    from .demand import gravity_matrix
    gravity = gravity_matrix(cities, config)
    ranked = sorted(((a, b) for a in range(n) for b in range(a + 1, n)),
                    key=lambda ab: (-gravity[ab], ab))
    pairs.update(ranked[:p['force_pairs']])
    retained = set(pairs)
    for a, b in sorted(pairs):
        direction = xy[b] - xy[a]
        position = (xy - xy[a]) @ direction / np.dot(direction, direction)
        near = np.linalg.norm(xy - (xy[a] + position[:, None] * direction), axis=1)
        via = np.flatnonzero((position > 0) & (position < 1) & (near <= p['via_tolerance_km']) &
                            (d[a] + d[b] <= p['via_detour_ratio'] * d[a, b]))
        if len(via):
            middle = int(via[np.argmin(near[via])])
            retained.discard((a, b))
            # Always retain the replacement links, including for forced long pairs.
            retained.add(tuple(sorted((a, middle))))
            retained.add(tuple(sorted((middle, b))))
    return sorted(retained)


def prepare(config):
    xy, people, density = population_samples(config)
    cities = read_cities(config, (xy, people, density))
    station_capture(cities, xy, people, density, config)
    return cities, candidate_pairs(cities, config), (xy, people, density)
