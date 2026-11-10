"""Data preparation, catchment approximation and optional coarse terrain routing."""
from dataclasses import dataclass
import heapq
import itertools
import json
import math
from pathlib import Path

import h3
import numpy as np
from pyproj import Transformer, Geod
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from shapely.geometry import Point, LineString, shape
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
        self.check = lambda: None
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
                if value.get('terrain_class') is not None and value['terrain_class'] not in {*config.terrain_cost_ranges_million, 'water'}:
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
        if terrain.get('terrain_class') == 'water':
            baseline = max(baseline, c.water_track_cost_per_km)
        elif terrain.get('terrain_class'):
            low, high = c.terrain_cost_ranges_million[terrain['terrain_class']]
            baseline = max(baseline, (low + c.terrain_cost_position * (high-low)) * 1_000_000)
        return (baseline + c.urban_cost_coefficient * max(0, density - c.urban_density_threshold)) * terrain.get('multiplier', 1)

    def alignment(self, points):
        """Integrate costs along geodesic segments, rather than grid directions."""
        geod = Geod(ellps='WGS84')
        spacing = min(self.config.routing_sample_km, self.config.water_routing_sample_km,
                      h3.average_hexagon_edge_length(self.resolution, unit='km') / 3)
        geometry, length, cost = [], 0.0, 0.0
        water_km, water_capital = 0.0, 0.0
        crossing_km, crossing_base = 0.0, 0.0

        def finish_crossing():
            nonlocal cost, water_capital, crossing_km, crossing_base
            # Cap the exponent only for numeric stability. The cap is still
            # astronomically above any feasible construction budget.
            premium = crossing_base * math.expm1(min(100, crossing_km / self.config.water_cost_growth_km))
            cost += premium
            water_capital += premium
            crossing_km, crossing_base = 0.0, 0.0
        for a, b in zip(points, points[1:]):
            km = geod.inv(*a, *b)[2] / 1000
            count = max(1, math.ceil(km / spacing))
            samples = [list(a)] + (geod.npts(*a, *b, count-1) if count > 1 else []) + [list(b)]
            if self.land is not None and not self.land.covers(LineString(samples)):
                return None
            self.check()
            for point in samples:
                if not self.allowed(h3.latlng_to_cell(point[1], point[0], self.resolution)):
                    return None
            for x, y in zip(samples, samples[1:]):
                self.check()
                segment_km = geod.inv(*x, *y)[2] / 1000
                midpoint = geod.npts(*x, *y, 1)[0] if segment_km > 0 else x
                cell = h3.latlng_to_cell(midpoint[1], midpoint[0], self.resolution)
                if not self.allowed(cell):
                    return None
                length += segment_km
                if self.terrain_map is not None:
                    # H3 centers can fall on the opposite side of a coastline.
                    # Price the sampled alignment itself, using H3 only for density.
                    try:
                        terrain_class = self.terrain_map.terrain_class(*midpoint)
                        unit_cost = self.terrain_map.cost_per_km(*midpoint, self.density.get(cell, 0))
                    except ValueError:
                        return None
                else:
                    terrain_class = self.terrain.get(cell, {}).get('terrain_class')
                    unit_cost = self.cost(cell)
                segment_cost = segment_km * unit_cost
                cost += segment_cost
                if terrain_class == 'water':
                    water_km += segment_km
                    water_capital += segment_cost
                    crossing_km += segment_km
                    crossing_base += segment_cost
                else:
                    finish_crossing()
            geometry.extend(samples if not geometry else samples[1:])
        finish_crossing()
        return dict(km=length, track_capital=cost, geometry=geometry,
                    water_km=water_km, water_capital=water_capital)

    def routes(self, a, b):
        """Direct plus a weighted Theta* any-angle construction-cost alternative."""
        direct = self.alignment([a, b])
        candidates = [dict(direct, alignment_kind='direct')] if direct else []
        start, end = [h3.latlng_to_cell(p[1], p[0], self.resolution) for p in (a, b)]
        if not self.allowed(start) or not self.allowed(end) or start == end:
            return candidates
        def coordinate(cell):
            if cell == start:
                return a
            if cell == end:
                return b
            lat, lon = h3.cell_to_latlng(cell)
            return (lon, lat)
        cache = {}
        def segment(u, v):
            key = tuple(sorted((u, v)))
            if key not in cache:
                result = self.alignment([coordinate(u), coordinate(v)])
                cache[key] = result['track_capital'] if result else math.inf
            return cache[key]
        queue, distance, previous, visited = [(0, start)], {start: 0}, {start: start}, set()
        while queue and len(visited) < self.config.terrain_search_limit:
            _, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                break
            self.check()
            for neighbor in sorted(h3.grid_disk(current, 1)):
                if neighbor == current or neighbor in visited or not self.allowed(neighbor):
                    continue
                # Compare both: a visible parent shortcut can cross costly land.
                options = [(distance[current] + segment(current, neighbor), current)]
                parent = previous[current]
                if parent != current:
                    options.append((distance[parent] + segment(parent, neighbor), parent))
                trial, predecessor = min(options)
                if trial < distance.get(neighbor, math.inf):
                    distance[neighbor], previous[neighbor] = trial, predecessor
                    heuristic = haversine(coordinate(neighbor), b) * self.config.base_track_cost_per_km * .99
                    heapq.heappush(queue, (trial + heuristic, neighbor))
        if end in visited:
            path = [end]
            while path[-1] != start:
                path.append(previous[path[-1]])
            alternative = self.alignment([coordinate(node) for node in reversed(path)])
            if alternative and (not direct or
                    alternative['track_capital'] < direct['track_capital'] - 1):
                candidates.append(dict(alternative, alignment_kind='terrain_any_angle'))
        return candidates

    def route(self, a, b):
        """Compatibility helper returning the cheapest generated alignment."""
        return min(self.routes(a, b), key=lambda route: route['track_capital'], default=None)


def candidate_market_pairs(cities, config):
    """Local corridors plus a bounded pool of demand-driven longer links."""
    pairs = set()
    for i, city in enumerate(cities):
        others = sorted((haversine(city['center'], other['center']), j) for j, other in enumerate(cities) if i != j)
        for distance, j in others[:config.neighbors_per_city]:
            if distance <= config.maximum_edge_km:
                pairs.add(tuple(sorted((i, j))))
    if config.solver_method == 'heuristic':
        # Nearest-neighbor corridors can strand large OD markets behind small
        # intermediate towns. Reserve additional pairs by potential fare volume.
        existing = set(pairs)
        for i, city in enumerate(cities):
            options = []
            for j, other in enumerate(cities):
                pair = tuple(sorted((i, j)))
                km = haversine(city['center'], other['center'])
                if i == j or pair in existing or km > min(config.maximum_edge_km, config.maximum_trip_distance_km):
                    continue
                potential = city['population']*other['population']*km / max(km, config.distance_floor_km)**config.gravity_beta
                options.append((-potential, j, pair))
            pairs.update(pair for _, _, pair in sorted(options)[:config.heuristic_demand_connections])
    return pairs


def country_mask(config):
    """Prepared country union, or None for bbox-only discovery."""
    if not config.countries:
        return None
    raw = json.loads(Path(config.country_boundaries_file).read_text())
    requested = set(config.countries)
    features = [f for f in raw['features'] if f['properties'].get('iso_a2') in requested]
    found = {f['properties']['iso_a2'] for f in features}
    if requested - found:
        raise ValueError(f'Country boundaries missing codes: {sorted(requested - found)}')
    return prep(unary_union([shape(f['geometry']) for f in features]))


def prepare(config, progress=lambda message: None, publish=lambda problem: None):
    config.validate()
    warnings = [
        'Uncalibrated demand, uptake, fares and costs: results are scenarios, not forecasts.',
        'Population counts/coverage require source verification; absent cells are unknown.',
        'Population is distributed uniformly among child H3 cells for catchment estimates.',
        'Demand outside the bounding box is excluded; border effects can be large.',
        'Population cells are filtered by bbox and country polygons; track routes are not constrained by political borders.',
        'Access uses fixed nearest-city origin zones; cross-zone access to another city station is excluded.',
        'Line patterns are a finite set of all-stop services; optimality is relative to these candidates.',
        'Direct geodesic and finite any-angle terrain alternatives use sampled costs; sub-sample obstacles may be missed.',
        'Hourly steady-state demand/capacity model; not a timetable or queue guarantee.',
        'The 06:00–22:00 window constrains departures; late arrivals may be after closing.',
        'Geometric crossing corridors are separate tracks unless sharing the same candidate edge.',
        'Terrain multipliers/exclusions do not enforce rail gradient or curvature engineering limits.',
    ]
    if not config.terrain_file or not config.land_file:
        warnings.append('Missing terrain and/or land mask: corridors may cross water or unsuitable land.')
    if config.terrain_file and Path(config.terrain_file).suffix.lower() in {'.tif', '.tiff'}:
        warnings.append('RGB relief-map colors approximate normal land, water and mountains; no measured elevations or slopes. Water uses exponential continuous-crossing scenario costs, including open sea.')
    if not config.terrain_file:
        warnings.append('No elevation, gradient, curve, bridge or tunnel engineering constraints are available.')
    west, south, east, north = config.bbox
    progress(f'Reading population data: {config.population_file}')
    raw = json.loads(Path(config.population_file).read_text())
    cells = []
    progress('Loading country boundaries and filtering population cells')
    region = country_mask(config)
    for feature in raw['features']:
        lon, lat = feature['geometry']['coordinates'][:2]
        if (west <= lon <= east and south <= lat <= north
                and (region is None or region.covers(Point(lon, lat)))):
            p = feature['properties']
            index = p['h3Index']
            population = float(p['population'])
            if not h3.is_valid_cell(index) or not math.isfinite(population) or population < 0:
                raise ValueError('invalid population cell')
            cells.append(dict(h3=index, lon=lon, lat=lat, population=population,
                              density=population/h3.cell_area(index, unit='km^2')))
    if not cells:
        raise ValueError('No population cells in bbox and configured countries')
    if len({c['h3'] for c in cells}) != len(cells):
        raise ValueError('Duplicate H3 population cells would count residents twice')
    if len({h3.get_resolution(c['h3']) for c in cells}) != 1:
        raise ValueError('Population data must have one H3 resolution')
    progress(f'Loaded {len(cells)} population cells')
    from .seeds import population_seeds, label_markets, market_membership
    progress('Finding population market seeds')
    seeds = population_seeds(cells, config)
    progress(f'Found {len(seeds)} market seeds; subdividing population cells')
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
            if (not west <= lon <= east or not south <= lat <= north
                    or (region is not None and not region.covers(Point(lon, lat)))):
                continue
            x, y = forward.transform(lon, lat)
            sample_xy.append((x/1000, y/1000))
            sample_pop.append(cell['population'] * weight)
            sample_density.append(cell['density'])
    xy, population, density = map(np.asarray, (sample_xy, sample_pop, sample_density))
    if not len(xy):
        raise ValueError('No population samples inside bbox')
    tree = cKDTree(xy)
    progress(f'Built {len(xy):,} population samples; selecting station locations')
    cities, locations = [], []
    for city_id, seed in enumerate(seeds):
        sx, sy = forward.transform(*seed['coordinate'])
        seed_xy = np.array([sx/1000, sy/1000])
        local = tree.query_ball_point(seed_xy, seed['radius_km'] + 1e-9)
        if local:
            peak = local[int(np.argmax(density[local]))]
            radius = seed['radius_km']
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
    progress(f'Labeling {len(cities)} markets and assigning population catchments')
    label_markets(cities, config.cities_file, countries=config.countries)
    # Fixed origin zones use population seeds; labels never move their boundaries.
    seed_points = np.array([np.array(forward.transform(*s['coordinate'])) / 1000 for s in seeds])
    owner, eligible = market_membership(xy, seed_points, density, config, [s['radius_km'] for s in seeds])
    for city in cities:
        city['population'] = float(population[(owner == city['id']) & eligible].sum())
    stations = []
    for location in locations:
        for radius in (cities[location['city']]['radius_km'],):
            near = np.asarray(tree.query_ball_point(location['xy'], radius), dtype=int)
            near = near[(owner[near] == location['city']) & eligible[near]]
            covered = float(population[near].sum())
            if covered <= 0:
                continue
            _, mean_access_km, access_multiplier = access_statistics(
                xy[near], population[near], location['xy'], config)
            access = mean_access_km / config.access_speed_kmh
            total = cities[location['city']]['population']
            uptake_radius = config.uptake_reference_radius_km or radius
            stations.append(dict(id=len(stations), location=location['id'], city=location['city'],
                                 radius_km=radius, covered_population=covered,
                                 uptake=uptake(uptake_radius, config.p_max, config.uptake_a, config.uptake_alpha),
                                 capture_fraction=covered / total * uptake(uptake_radius, config.p_max, config.uptake_a, config.uptake_alpha),
                                 access_hours=access, mean_access_km=mean_access_km,
                                 access_multiplier=access_multiplier))
    live_locations = {s['location'] for s in stations}
    if config.solver_method == 'heuristic':
        # Route only the station options the heuristic will actually evaluate.
        # Preparing every discarded location multiplies terrain searches.
        live_locations = {max((s for s in stations if s['city'] == city['id']),
                              key=lambda s: (s['capture_fraction'], -s.get('access_hours', 0), -s['id']))['location']
                          for city in cities if any(s['city'] == city['id'] for s in stations)}
    # Publish a read-only population/city view before potentially lengthy terrain
    # routing, so the reporter can show useful geography when requested.
    publish(Problem(config, cells, cities, locations, stations, [], [], warnings))
    progress('Initializing terrain router and candidate market pairs')
    router = Router(config, cells) if config.solver_method != 'annealing' else None
    pairs = candidate_market_pairs(cities, config) if router is not None else set()
    if router is None:
        progress('Deferring terrain routing until annealing proposes a corridor')
    edges = []
    for pair_number, (i, j) in enumerate(sorted(pairs), 1):
        progress(f'Routing candidate market pair {pair_number}/{len(pairs)} ({i+1}–{j+1})')
        for a, b in itertools.product([l for l in locations if l['city'] == i and l['id'] in live_locations],
                                      [l for l in locations if l['city'] == j and l['id'] in live_locations]):
            for route in router.routes(a['coordinate'], b['coordinate']):
                if 0 < route['km'] <= config.maximum_edge_km:
                    edges.append(dict(id=len(edges), a=a['id'], b=b['id'], **route))
        progress(f'Routed candidate market pair {i+1}–{j+1}')
    progress('Computing directed passenger demand')
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
    if not demand or not stations or (not edges and config.solver_method != 'annealing'):
        raise ValueError('No candidate demand, stations or corridors; inspect region/threshold/terrain settings')
    progress(f'{len(cities)} markets, {len(stations)} station options, {len(edges)} corridors, {len(demand)} directed OD pairs')
    return Problem(config, cells, cities, locations, stations, edges, demand, warnings)
