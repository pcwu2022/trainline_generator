"""Data preparation, catchment approximation and optional coarse terrain routing."""
from dataclasses import dataclass
import heapq
import itertools
import json
import math
from pathlib import Path

import h3
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from rail_model import uptake


@dataclass
class Problem:
    config: object
    cells: list
    cities: list
    locations: list
    stations: list
    edges: list
    demand: list
    warnings: list


def haversine(a, b):
    """Great-circle km for two (lon, lat) coordinates."""
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    x = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(x)))


def access_statistics(points, people, station, config):
    """Mean access distance relative to the same residents' best possible site."""
    people = np.asarray(people)
    points = np.asarray(points)
    def mean_distance(point):
        return float(np.average(np.linalg.norm(points - point, axis=1), weights=people))
    center = minimize(mean_distance, np.average(points, axis=0, weights=people),
                      method='Powell', options={'xtol': config.access_optimizer_tolerance_km}).x
    ideal = mean_distance(center)
    mean = mean_distance(station)
    # Degenerate colocated populations have a well-defined centered score of 1.
    score = max(1.0, mean / max(ideal, config.access_normalization_floor_km))
    return center, mean, score


class Router:
    def __init__(self, config, cells):
        self.config = config
        self.density = {c['h3']: c['density'] for c in cells}
        self.resolution = h3.get_resolution(cells[0]['h3'])
        self.terrain = {}
        self.land = None
        self.terrain_map = None
        if config.terrain_file and Path(config.terrain_file).suffix.lower() in {'.tif', '.tiff'}:
            from .terrain import TerrainCostMap
            self.terrain_map = TerrainCostMap(config.terrain_file, config)
        if config.terrain_file and self.terrain_map is None:
            raw = json.loads(Path(config.terrain_file).read_text())
            if raw.get('h3_resolution') != self.resolution:
                raise ValueError('terrain h3_resolution must match population cells')
            self.terrain = raw.get('cells', {})
            for key, value in self.terrain.items():
                if not h3.is_valid_cell(key) or h3.get_resolution(key) != self.resolution:
                    raise ValueError(f'invalid terrain H3 cell: {key}')
                if value.get('terrain_class') is not None and value['terrain_class'] not in config.terrain_cost_ranges_million:
                    raise ValueError(f'Unknown terrain_class: {value["terrain_class"]}')
                factor = value.get('multiplier', 1)
                if not math.isfinite(factor) or factor < 1:
                    raise ValueError('terrain multipliers must be finite and >= 1')
        if config.land_file:
            raw = json.loads(Path(config.land_file).read_text())
            geometries = [shape(f['geometry']) for f in raw['features']]
            self.land = prep(unary_union(geometries))

    def allowed(self, cell):
        if self.terrain.get(cell, {}).get('blocked', False):
            return False
        lat, lon = h3.cell_to_latlng(cell)
        if self.terrain_map is not None:
            try:
                self.terrain_map.terrain_class(lon, lat)
            except ValueError:
                return False
        return self.land is None or self.land.covers(Point(lon, lat))

    def cost(self, cell):
        c = self.config
        density = self.density.get(cell, 0)
        # Missing population is unknown; base cost here is an explicit optimistic
        # assumption, flagged in all outputs, rather than evidence of empty land.
        terrain = self.terrain.get(cell, {})
        if self.terrain_map is not None:
            lat, lon = h3.cell_to_latlng(cell)
            return self.terrain_map.cost_per_km(lon, lat, density)
        baseline = c.base_track_cost_per_km
        if terrain.get('terrain_class'):
            low, high = c.terrain_cost_ranges_million[terrain['terrain_class']]
            baseline = max(baseline, (low + c.terrain_cost_position * (high-low)) * 1_000_000)
        return (baseline + c.urban_cost_coefficient * density) * terrain.get('multiplier', 1)

    def route(self, a, b):
        c = self.config
        if not c.terrain_file and not c.land_file:
            count = max(2, math.ceil(haversine(a, b) / c.routing_sample_km) + 1)
            geometry = np.linspace(a, b, count).tolist()
        else:
            start, end = [h3.latlng_to_cell(p[1], p[0], self.resolution) for p in (a, b)]
            if not self.allowed(start) or not self.allowed(end):
                return None
            queue = [(0, start)]
            distance, previous = {start: 0}, {}
            visited = set()
            while queue and len(visited) < c.terrain_search_limit:
                _, current = heapq.heappop(queue)
                if current in visited:
                    continue
                visited.add(current)
                if current == end:
                    break
                lat, lon = h3.cell_to_latlng(current)
                for neighbor in h3.grid_disk(current, 1):
                    if neighbor == current or neighbor in visited or not self.allowed(neighbor):
                        continue
                    nlat, nlon = h3.cell_to_latlng(neighbor)
                    w = haversine((lon, lat), (nlon, nlat)) * (self.cost(current) + self.cost(neighbor)) / 2
                    trial = distance[current] + w
                    if trial < distance.get(neighbor, math.inf):
                        distance[neighbor] = trial
                        previous[neighbor] = current
                        heuristic = haversine((nlon, nlat), b) * c.base_track_cost_per_km
                        heapq.heappush(queue, (trial + heuristic, neighbor))
            if end not in visited:
                return None
            path = [end]
            while path[-1] != start:
                path.append(previous[path[-1]])
            middle = [[h3.cell_to_latlng(x)[1], h3.cell_to_latlng(x)[0]] for x in reversed(path)]
            geometry = [list(a)] + middle + [list(b)]
        length = cost = 0
        for x, y in zip(geometry, geometry[1:]):
            km = haversine(x, y)
            midpoint = [(x[i]+y[i])/2 for i in (0, 1)]
            cell = h3.latlng_to_cell(midpoint[1], midpoint[0], self.resolution)
            if not self.allowed(cell):
                return None
            length += km
            cost += km * self.cost(cell)
        return dict(km=length, track_capital=cost, geometry=geometry)


def prepare(config, progress=lambda message: None, publish=lambda problem: None):
    config.validate()
    warnings = [
        'Uncalibrated demand, uptake, fares and costs: results are scenarios, not forecasts.',
        'Population counts/coverage require source verification; absent cells are unknown.',
        'Population is distributed uniformly among child H3 cells for catchment estimates.',
        'Demand outside the bounding box is excluded; border effects can be large.',
        'Population seeds and demand are clipped by bbox; countries is legacy metadata and does not filter geography.',
        'Access uses fixed nearest-city origin zones; cross-zone access to another city station is excluded.',
        'Line patterns are a finite set of all-stop services; optimality is relative to these candidates.',
        'Hourly steady-state demand/capacity model; not a timetable or queue guarantee.',
        'The 06:00–22:00 window constrains departures; late arrivals may be after closing.',
        'Geometric crossing corridors are separate tracks unless sharing the same candidate edge.',
        'Terrain multipliers/exclusions do not enforce rail gradient or curvature engineering limits.',
    ]
    if not config.terrain_file or not config.land_file:
        warnings.append('Missing terrain and/or land mask: corridors may cross water or unsuitable land.')
    if config.terrain_file and Path(config.terrain_file).suffix.lower() in {'.tif', '.tiff'}:
        warnings.append('RGB relief-map colors approximate normal land, water and mountains; no measured elevations or slopes. Water uses bridge scenario costs, including open sea.')
    if not config.terrain_file:
        warnings.append('No elevation, gradient, curve, bridge or tunnel engineering constraints are available.')
    west, south, east, north = config.bbox
    raw = json.loads(Path(config.population_file).read_text())
    cells = []
    for feature in raw['features']:
        lon, lat = feature['geometry']['coordinates'][:2]
        if west <= lon <= east and south <= lat <= north:
            p = feature['properties']
            index = p['h3Index']
            population = float(p['population'])
            if not h3.is_valid_cell(index) or not math.isfinite(population) or population < 0:
                raise ValueError('invalid population cell')
            cells.append(dict(h3=index, lon=lon, lat=lat, population=population,
                              density=population/h3.cell_area(index, unit='km^2')))
    if not cells:
        raise ValueError('No population cells in bbox')
    if len({c['h3'] for c in cells}) != len(cells):
        raise ValueError('Duplicate H3 population cells would count residents twice')
    if len({h3.get_resolution(c['h3']) for c in cells}) != 1:
        raise ValueError('Population data must have one H3 resolution')
    progress(f'Loaded {len(cells)} population cells')
    from .seeds import population_seeds, label_markets, market_membership
    seeds = population_seeds(cells, config)
    if len(seeds) < 2:
        raise ValueError('At least two separated population markets are needed; change bbox or screening settings')
    # Local azimuthal equidistant projection; centered on the selected region.
    crs = f'+proj=aeqd +lat_0={(south+north)/2} +lon_0={(west+east)/2} +datum=WGS84 +units=m'
    forward = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    backward = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    sample_xy, sample_pop, sample_density = [], [], []
    for cell in cells:
        resolution = h3.get_resolution(cell['h3']) + config.population_subdivision
        if resolution > 15:
            raise ValueError('Population subdivision exceeds H3 resolution 15')
        children = sorted(h3.cell_to_children(cell['h3'], resolution))
        areas = np.array([h3.cell_area(x, unit='km^2') for x in children])
        weights = areas / areas.sum()
        for child, weight in zip(children, weights):
            lat, lon = h3.cell_to_latlng(child)
            if not west <= lon <= east or not south <= lat <= north:
                continue
            x, y = forward.transform(lon, lat)
            sample_xy.append((x/1000, y/1000))
            sample_pop.append(cell['population'] * weight)
            sample_density.append(cell['density'])
    xy, population, density = map(np.asarray, (sample_xy, sample_pop, sample_density))
    if not len(xy):
        raise ValueError('No population samples inside bbox')
    tree = cKDTree(xy)
    cities, locations = [], []
    for city_id, seed in enumerate(seeds):
        sx, sy = forward.transform(*seed['coordinate'])
        seed_xy = np.array([sx/1000, sy/1000])
        local = tree.query_ball_point(seed_xy, config.market_radius_km)
        if local:
            peak = local[int(np.argmax(density[local]))]
            radius = config.market_radius_km
            if config.center_method == 'access':
                center, _, _ = access_statistics(xy[local], population[local], seed_xy, config)
            elif config.center_method == 'peak':
                center = xy[peak]
            else:
                # Compare coverage using the unified radius; include the population seed.
                candidates = [seed_xy] + [xy[i] for i in local]
                center = max(candidates, key=lambda point: float(population[tree.query_ball_point(point, radius)].sum()))
        else:
            radius, center = 0.0, seed_xy
        city = dict(id=city_id, **seed, center=backward.transform(*(center*1000)),
                    search_radius_km=radius)
        cities.append(city)
        choices = [center]
        if np.linalg.norm(seed_xy-center) > 1:
            choices.append(seed_xy)
        # Additional alternatives use nearby population-cell centers.
        nearby = tree.query_ball_point(center, config.station_offset_km)
        choices.extend(xy[i] for i in sorted(nearby, key=lambda i: (np.linalg.norm(xy[i]-center), i)))
        accepted = []
        for point in choices:
            if all(np.linalg.norm(point-other) > 1 for other in accepted):
                accepted.append(point)
                locations.append(dict(id=len(locations), city=city_id,
                                      coordinate=backward.transform(*(point*1000)),
                                      xy=point.tolist(), center_offset_km=float(np.linalg.norm(point-center))))
                if len(accepted) == config.locations_per_city:
                    break
    label_markets(cities, config.cities_file)
    # Fixed origin zones use population seeds; labels never move their boundaries.
    seed_points = np.array([np.array(forward.transform(*s['coordinate'])) / 1000 for s in seeds])
    owner, eligible = market_membership(xy, seed_points, density, config)
    for city in cities:
        city['population'] = float(population[(owner == city['id']) & eligible].sum())
    stations = []
    for location in locations:
        for radius in (config.market_radius_km,):
            near = np.asarray(tree.query_ball_point(location['xy'], radius), dtype=int)
            near = near[(owner[near] == location['city']) & eligible[near]]
            covered = float(population[near].sum())
            if covered <= 0:
                continue
            _, mean_access_km, access_multiplier = access_statistics(
                xy[near], population[near], location['xy'], config)
            access = mean_access_km / config.access_speed_kmh
            total = cities[location['city']]['population']
            stations.append(dict(id=len(stations), location=location['id'], city=location['city'],
                                 radius_km=radius, covered_population=covered,
                                 uptake=uptake(radius, config.p_max, config.uptake_a, config.uptake_alpha),
                                 capture_fraction=covered / total * uptake(radius, config.p_max, config.uptake_a, config.uptake_alpha),
                                 access_hours=access, mean_access_km=mean_access_km,
                                 access_multiplier=access_multiplier))
    live_locations = {s['location'] for s in stations}
    # Publish a read-only population/city view before potentially lengthy terrain
    # routing, so the reporter can show useful geography when requested.
    publish(Problem(config, cells, cities, locations, stations, [], [], warnings))
    router = Router(config, cells)
    pairs = set()
    for i, city in enumerate(cities):
        others = sorted((haversine(city['center'], other['center']), j) for j, other in enumerate(cities) if i != j)
        for distance, j in others[:config.neighbors_per_city]:
            if distance <= config.maximum_edge_km:
                pairs.add(tuple(sorted((i, j))))
    edges = []
    for i, j in sorted(pairs):
        for a, b in itertools.product([l for l in locations if l['city'] == i and l['id'] in live_locations],
                                      [l for l in locations if l['city'] == j and l['id'] in live_locations]):
            route = router.route(a['coordinate'], b['coordinate'])
            if route and 0 < route['km'] <= config.maximum_edge_km:
                edges.append(dict(id=len(edges), a=a['id'], b=b['id'], **route))
        progress(f'Routed candidate market pair {i+1}–{j+1}')
    demand = []
    for i, origin in enumerate(cities):
        weights = {}
        for j, destination in enumerate(cities):
            if i == j:
                continue
            km = haversine(origin['coordinate'], destination['coordinate'])
            if km <= config.maximum_trip_distance_km and origin['population'] and destination['population']:
                weights[j] = (km, destination['population'] / max(km, config.distance_floor_km)**config.gravity_beta)
        denominator = sum(w for _, w in weights.values())
        for j, (km, weight) in sorted(weights.items()):
            if config.demand_model == 'gravity':
                volume = config.gravity_scale * origin['population'] * weight
            else:
                volume = config.trips_per_person_hour * origin['population'] * weight / denominator
            if volume > 1e-8:
                demand.append(dict(id=len(demand), origin=i, destination=j, km=km, volume=volume))
    if not demand or not stations or not edges:
        raise ValueError('No candidate demand, stations or corridors; inspect region/threshold/terrain settings')
    progress(f'{len(cities)} markets, {len(stations)} station options, {len(edges)} corridors, {len(demand)} directed OD pairs')
    return Problem(config, cells, cities, locations, stations, edges, demand, warnings)
