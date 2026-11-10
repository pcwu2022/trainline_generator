"""Lazy corridor generation and annealing over explicit operating services."""
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import highspy
import numpy as np

from .sampling import PHASE_ACTION_WEIGHTS, choose_action, pop_cost_candidate, sample_actions, sample_indices
from .finance import annuity_factor, asset_pv_factor
from .consolidation import annual_cost, candidates, materialize, service_protection
from .geography import Router, haversine
from pyproj import Geod
import h3
from .model import NetworkModel, SolverModelError
from .needs import rail_need_bound
from .network_growth import (connected_designs, connectivity, growth_energy, integration_pressure,
                             ready_to_consolidate, regional_proposal)


def input_signature(problem):
    """Include assumptions, discovered geography and source contents for resume."""
    data = dict(config=asdict(problem.config), cities=problem.cities,
                locations=problem.locations, stations=problem.stations, demand=problem.demand)
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode())
    for path in (problem.config.population_file, problem.config.cities_file,
                 problem.config.country_boundaries_file, problem.config.terrain_file,
                 problem.config.land_file):
        if path and Path(path).exists():
            with Path(path).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
    return digest.hexdigest()


class RoutePool:
    """An in-run cache, including failed searches; endpoints are immutable sites."""
    def __init__(self, problem, check=lambda: None, progress=lambda message: None):
        self.problem, self.check, self.progress = problem, check, progress
        self.router = None
        self.edges = [dict(e) for e in problem.edges]
        self.cache = {}
        self.searches = 0
        capture = {city['id']: max((s['capture_fraction'] for s in problem.stations
                                   if s['city'] == city['id']), default=0) for city in problem.cities}
        self.needs_upper_bound = rail_need_bound(problem)
        self.capture_upper_bound = sum(d['volume']*capture[d['origin']]*capture[d['destination']]
                                       for d in problem.demand)
        for e in self.edges:
            self.cache.setdefault(tuple(sorted((e['a'], e['b']))), []).append(e['id'])

    def demand_weights(self, indices=None):
        """Cache terrain screening only for requested demand records.

        The default still returns the complete array for legacy callers.
        Regional sampling requests only its bounded subset of indices.
        """
        c = self.problem.config
        if not hasattr(self, '_demand_weights'):
            self._demand_weights = np.full(len(self.problem.demand), np.nan)
            self.capital_estimates = {}
            fallback = {}
            for site in self.problem.locations:
                fallback.setdefault(site['city'], site['coordinate'])
            self._market_centers = {city['id']: city.get('center', fallback.get(city['id']))
                                    for city in self.problem.cities}
        centers = self._market_centers
        estimates = self.capital_estimates
        requested = range(len(self.problem.demand)) if indices is None else indices
        geod = Geod(ellps='WGS84')
        for index in requested:
            if not np.isnan(self._demand_weights[index]):
                continue
            od = self.problem.demand[index]
            pair = tuple(sorted((od['origin'], od['destination'])))
            if pair not in estimates:
                a, b = (centers[i] for i in pair)
                if a is None or b is None:
                    estimates[pair] = float('inf')
                else:
                    km = haversine(a, b)
                    capital = max(km, c.distance_floor_km)*c.base_track_cost_per_km
                    if c.annealing_proposal_cost_exponent and c.terrain_file and self.problem.cells:
                        self.check()
                        if self.router is None:
                            self.router = Router(c, self.problem.cells)
                            self.router.check = self.check
                        # Equal-length midpoint samples provide a cheap estimate;
                        # full routing and feasibility decide the actual alignment.
                        points = [a]+geod.npts(*a, *b, 31)+[b]
                        rates = []
                        for k in range(1, len(points), 2):
                            self.check()
                            lon, lat = points[k]
                            try:
                                if self.router.terrain_map is not None:
                                    rate = self.router.terrain_map.cost_per_km(lon, lat)
                                else:
                                    cell = h3.latlng_to_cell(lat, lon, self.router.resolution)
                                    rate = self.router.cost(cell)
                            except ValueError:
                                rate = max(c.base_track_cost_per_km, c.water_track_cost_per_km)
                            rates.append(rate)
                        capital = max(capital, km*sum(rates)/len(rates))
                    estimates[pair] = capital
            capital = estimates[pair]
            # Divide by a fixed reference before exponentiation for stable units.
            relative = max(1., capital/max(1., c.base_track_cost_per_km*c.distance_floor_km))
            self._demand_weights[index] = (od['volume']/relative**c.annealing_proposal_cost_exponent
                                           if math.isfinite(capital) else 0.)
        if indices is None:
            return self._demand_weights
        return {index: self._demand_weights[index] for index in requested}

    def get(self, a, b):
        self.check()
        key = tuple(sorted((a, b)))
        if key not in self.cache:
            pa, pb = [self.problem.locations[i]['coordinate'] for i in key]
            if haversine(pa, pb) > self.problem.config.maximum_edge_km:
                self.cache[key] = []
                return []
            self.progress(f'Routing proposed corridor {a+1}–{b+1}')
            if self.router is None:
                self.router = Router(self.problem.config, self.problem.cells)
                self.router.check = self.check
            routes = self.router.routes(pa, pb)
            self.searches += 1
            self.cache[key] = []
            for route in routes:
                if 0 < route['km'] <= self.problem.config.maximum_edge_km:
                    index = len(self.edges)
                    self.edges.append(dict(id=index, a=key[0], b=key[1], **route))
                    self.cache[key].append(index)
        return self.cache[key]


def make_lines(design, edges):
    """A design entry is (ordered station sites, ordered cached edge IDs, frequency)."""
    lines = []
    for stops, ids, frequency in design:
        geometry = []
        for a, edge_id in zip(stops, ids):
            edge = edges[edge_id]
            segment = edge['geometry'] if edge['a'] == a else list(reversed(edge['geometry']))
            geometry.extend(segment if not geometry else segment[1:])
        lines.append(dict(id=len(lines), a=stops[0], b=stops[-1], stops=list(stops),
                          edges=list(ids), segment_km=[edges[i]['km'] for i in ids],
                          alignment_kinds=[edges[i].get('alignment_kind', 'legacy') for i in ids],
                          km=sum(edges[i]['km'] for i in ids), geometry=geometry, frequencies=[frequency]))
    return lines


def canonical(design):
    result = []
    for stops, edges, frequency in design:
        if stops > tuple(reversed(stops)):
            stops, edges = tuple(reversed(stops)), tuple(reversed(edges))
        result.append((tuple(stops), tuple(edges), int(frequency)))
    return tuple(sorted(set(result)))


def financial_cost(model):
    """Financial-only coefficients in the final, dimensionless objective."""
    c = model.problem.config
    cost = np.zeros(len(model.names))
    for i, capital in model.capital.items():
        equivalent = (capital * asset_pv_factor(c.asset_lives_years[model.asset_kind[i]],
                      c.appraisal_years, c.discount_rate) / model.appraisal_annuity + model.operating[i])
        cost[i] = c.cost_weight * equivalent / 1e8
    cost[model.profit_shortfall] = c.profit_shortfall_weight / 1e5
    return cost


def coverage_deficit(model, vector, config):
    total = max(1, model.total_demand)
    served = sum(vector[i] for i in model.q.values())
    live = sum(vector[i] for i in model.x.values())
    deficit = max(0, config.minimum_served_fraction - served/total)
    deficit += max(0, config.minimum_city_served_fraction - live/max(1, len(model.problem.cities)))
    for city in model.problem.cities:
        origin = city['id']
        demand = sum(d['volume'] for d in model.problem.demand if d['origin'] == origin)
        if not demand:
            continue
        flow = sum(vector[i] for (od, a, b), i in model.q.items()
                   if model.problem.demand[od]['origin'] == origin)
        selected = sum(vector[model.x[s]] for s in model.by_city[origin])
        deficit += max(0, config.minimum_origin_served_fraction-flow/demand) / len(model.problem.cities)
        deficit += max(0, config.minimum_selected_origin_served_fraction*selected-flow/demand) / len(model.problem.cities)
    return deficit


def evaluate(problem, pool, design, financial_weight, remaining, interrupted, final=False, protection=None, network_pressure=0, conventional_only=False):
    """Fix the proposed network; optimize train technology and passenger assignment."""
    c = problem.config
    live = {site for stops, ids, f in design for site in stops}
    # One site per market; station moves must replace every incident service.
    if len({problem.locations[i]['city'] for i in live}) != len(live):
        return None
    used = sorted({i for stops, ids, f in design for i in ids})
    lookup = {old: new for new, old in enumerate(used)}
    edges = [dict(pool.edges[i], id=lookup[i]) for i in used]
    local = [(stops, tuple(lookup[i] for i in ids), f) for stops, ids, f in design]
    lines = make_lines(local, edges)
    relaxed = replace(c, minimum_served_fraction=0, minimum_city_served_fraction=0,
                      minimum_origin_served_fraction=0, minimum_selected_origin_served_fraction=0,
                      require_connected_network=False)
    if final and len(live) < math.ceil(c.minimum_city_served_fraction*len(problem.cities)-1e-10):
        return None
    stations = [dict(s, id=i) for i, s in enumerate(s for s in problem.stations if s['location'] in live)]
    p = replace(problem, config=c if final else relaxed, edges=edges, stations=stations)
    if final and c.require_connected_network:
        adjacency = {site: set() for site in live}
        for edge in edges:
            adjacency[edge['a']].add(edge['b'])
            adjacency[edge['b']].add(edge['a'])
        reached, pending = set(), [min(live)]
        while pending:
            site = pending.pop()
            if site not in reached:
                reached.add(site)
                pending.extend(adjacency[site]-reached)
        if reached != live:
            return None
    model = NetworkModel(p, lines=lines, capture_upper_bound=pool.capture_upper_bound, protection=protection,
                         needs_upper_bound=pool.needs_upper_bound)
    financial = financial_cost(model)
    costs = model.cost - (1-financial_weight)*financial
    for index in model.q.values():
        costs[index] -= network_pressure*c.annealing_network_effect_weight*c.served_weight/max(1, pool.capture_upper_bound)
    fixed = np.zeros(len(model.names))
    for station in p.stations:
        fixed[model.x[station['id']]] = int(station['location'] in live)
    for _, ids, _ in local:
        for edge in ids:
            fixed[model.y[edge]] = 1
    if model.problem.config.require_connected_network:
        fixed[model.names.index(f'root_{min(live)}')] = 1
    technology = set(model.z.values()) | set(model.hsr_tracks.values())
    indices = np.array([i for i in np.flatnonzero(model.integer) if i not in technology], dtype=np.int32)
    if remaining() <= 0 or interrupted.is_set():
        raise InterruptedError('Annealing search stopped')
    try:
        solver = model.solver(c, max(.001, remaining()))
    except SolverModelError as error:
        # A numerically unusable corridor must not discard the existing walk
        # or the validated incumbent. Do not alter or clamp its physical cost.
        pool.progress(str(error))
        return None
    solver.cbSimplexInterrupt.subscribe(lambda event: event.interrupt() if interrupted.is_set() else None)
    solver.cbIpmInterrupt.subscribe(lambda event: event.interrupt() if interrupted.is_set() else None)
    solver.cbMipInterrupt.subscribe(lambda event: event.interrupt() if interrupted.is_set() else None)
    solver.changeColsBounds(len(indices), indices, fixed[indices], fixed[indices])
    if conventional_only:
        disabled = list(model.hsr_tracks.values()) + [index for (line, frequency), index in model.z.items()
                    if model.lines[line]['train_type'] == 'HSR']
        if disabled:
            disabled = np.asarray(disabled, dtype=np.int32)
            solver.changeColsBounds(len(disabled), disabled, np.zeros(len(disabled)), np.zeros(len(disabled)))
    columns = np.arange(len(costs), dtype=np.int32)
    solver.changeColsCost(len(costs), columns, costs)
    solver.run()
    solution = solver.getSolution()
    if not solution.value_valid:
        return None
    vector = np.asarray(solution.col_value, dtype=float)
    if not model.validation(vector)['valid'] or not np.allclose(vector[indices], fixed[indices], atol=1e-5, rtol=0):
        return None
    deficit = coverage_deficit(model, vector, c)
    # Match LinearModel.objective: avoid spurious BLAS floating-point flags on
    # some Apple Accelerate / NumPy combinations.
    money = float(np.sum(financial * vector, dtype=np.float64))
    service = model.objective(vector) - money
    return model, vector, service, money, deficit, local


def propose(problem, pool, design, rng, frequencies, growth=False, stage=None, action_override=None):
    c = problem.config
    if stage in ('exploration', 'integration'):
        weights = PHASE_ACTION_WEIGHTS[stage]
        allowed = sample_actions(weights, c.annealing_action_fraction, rng, weights)
        if not design:
            allowed = ['seed_region']
        action = choose_action(allowed, stage, rng)
        if action in ('seed_region', 'frontier', 'bridge'):
            regional = [a for a in allowed if a in ('seed_region', 'frontier', 'bridge')]
            return regional_proposal(problem, pool, design, rng, frequencies, stage,
                                     allowed=regional, preferred_action=action)
        return propose(problem, pool, design, rng, frequencies, action_override=action)
    proposal = list(design)
    action = action_override or (str(rng.choice(['add', 'add', 'remove', 'frequency', 'through', 'split', 'move', 'reroute', 'bypass', 'hub', 'extend', 'close_station'])) if design else 'add')
    if growth and design and action_override is None:
        action = str(rng.choice(['add']*6 + ['extend']*2 + ['frequency', 'through']))
    if design and action_override is None and c.annealing_growth_probability and rng.random() < c.annealing_growth_probability:
        action = 'add'
    if action == 'add':
        # Sample demand, rather than enumerating a neighborhood of all pairs.
        live_cities = {problem.locations[site]['city'] for stops, ids, f in design for site in stops}
        weights = pool.demand_weights().copy()
        if c.require_connected_network and design:
            weights *= [d['origin'] in live_cities or d['destination'] in live_cities
                        for d in problem.demand]
        if weights.sum() <= 0:
            return design, action
        od = problem.demand[int(rng.choice(len(weights), p=weights/weights.sum()))]
        sites = {problem.locations[s]['city']: s for stops, ids, f in design for s in stops}
        endpoints = []
        for city in (od['origin'], od['destination']):
            options = [s['location'] for s in problem.stations if s['city'] == city]
            if not options:
                return design, action
            endpoints.append(sites.get(city, int(rng.choice(options))))
        a, b = endpoints
        routes = pool.get(a, b)
        if routes:
            proposal.append(((a, b), (int(rng.choice(routes)),), frequencies[0]))
    elif action == 'close_station':
        old = int(rng.choice(sorted({s for stops, ids, f in design for s in stops})))
        proposal = [line for line in proposal if old not in line[0]]
    elif action == 'extend':
        i = int(rng.integers(len(proposal)))
        stops, ids, f = proposal[i]
        if len(stops) < c.maximum_line_stops:
            choices = [s['location'] for s in problem.stations
                       if s['city'] not in {problem.locations[n]['city'] for n in stops}]
            if choices:
                new = int(rng.choice(choices))
                routes = pool.get(stops[-1], new)
                if routes:
                    proposal[i] = (stops+(new,), ids+(int(rng.choice(routes)),), f)
    elif action == 'hub':
        i = int(rng.integers(len(proposal)))
        stops, ids, f = proposal[i]
        hubs = sorted({s for path, links, freq in design for s in path} - set(stops))
        if hubs and len(stops) < c.maximum_line_stops:
            hub = int(rng.choice(hubs))
            k = int(rng.integers(len(ids)))
            left, right = pool.get(stops[k], hub), pool.get(hub, stops[k+1])
            if left and right:
                proposal[i] = (stops[:k+1]+(hub,)+stops[k+1:],
                               ids[:k]+(int(rng.choice(left)), int(rng.choice(right)))+ids[k+1:], f)
    elif action == 'remove':
        proposal.pop(int(rng.integers(len(proposal))))
    elif action == 'frequency':
        i = int(rng.integers(len(proposal)))
        stops, ids, f = proposal[i]
        proposal[i] = (stops, ids, int(rng.choice(frequencies)))
    elif action == 'through' and len(design) >= 2:
        i, j = rng.choice(len(design), size=2, replace=False)
        a, ea, fa = design[i]
        b, eb, fb = design[j]
        for aa, ae in ((a, ea), (a[::-1], ea[::-1])):
            for bb, be in ((b, eb), (b[::-1], eb[::-1])):
                stops = aa + bb[1:]
                if aa[-1] == bb[0] and len(stops) <= c.maximum_line_stops and len({problem.locations[s]['city'] for s in stops}) == len(stops):
                    proposal = [line for k, line in enumerate(design) if k not in (i, j)]
                    proposal.append((stops, ae+be, max(fa, fb)))
                    return canonical(proposal), action
    elif action == 'split':
        i = int(rng.integers(len(proposal)))
        stops, ids, f = proposal[i]
        if len(stops) > 2:
            k = int(rng.integers(1, len(stops)-1))
            proposal.pop(i)
            proposal.extend([(stops[:k+1], ids[:k], f), (stops[k:], ids[k:], f)])
    elif action == 'move':
        live = sorted({s for stops, ids, f in design for s in stops})
        old = int(rng.choice(live))
        city = problem.locations[old]['city']
        choices = [s['location'] for s in problem.stations if s['city'] == city and s['location'] != old]
        if choices:
            new = int(rng.choice(choices))
            proposal = []
            for stops, ids, f in design:
                updated = tuple(new if s == old else s for s in stops)
                links = list(ids)
                for k, (a, b) in enumerate(zip(updated, updated[1:])):
                    if old in stops[k:k+2]:
                        routes = pool.get(a, b)
                        if not routes:
                            return design, action
                        links[k] = int(rng.choice(routes))
                proposal.append((updated, tuple(links), f))
    elif action == 'reroute':
        i = int(rng.integers(len(proposal)))
        stops, ids, f = proposal[i]
        k = int(rng.integers(len(ids)))
        routes = pool.get(stops[k], stops[k+1])
        if routes:
            links = list(ids)
            links[k] = int(rng.choice(routes))
            proposal[i] = (stops, tuple(links), f)
    elif action == 'bypass':
        # Remove one intermediate stop and replace its two links by a shortcut.
        i = int(rng.integers(len(proposal)))
        stops, ids, f = proposal[i]
        if len(stops) > 2:
            k = int(rng.integers(1, len(stops)-1))
            routes = pool.get(stops[k-1], stops[k+1])
            if routes:
                proposal[i] = (stops[:k]+stops[k+1:], ids[:k-1]+(int(rng.choice(routes)),)+ids[k+1:], f)
    return canonical(proposal), action


def repair_origin(problem, pool, design, record, frequencies, excluded=None):
    """Connect underserved selected origins before opening still more markets."""
    excluded = excluded or set()
    if not record or problem.config.minimum_selected_origin_served_fraction <= 0:
        return design
    model, vector = record
    demand, served = {}, {}
    for od in problem.demand:
        demand[od['origin']] = demand.get(od['origin'], 0)+od['volume']
    for (od, a, b), index in model.q.items():
        origin = problem.demand[od]['origin']
        served[origin] = served.get(origin, 0)+vector[index]
    sites = {problem.locations[s]['city']: s for stops, ids, f in design for s in stops}
    capture = {s['city']: s['capture_fraction'] for s in model.problem.stations}
    volumes = {(d['origin'], d['destination']): d['volume'] for d in problem.demand}
    target = problem.config.minimum_selected_origin_served_fraction
    deficits = sorted((-(target-served.get(city, 0)/demand[city]), city)
                      for city in sites if demand.get(city, 0) > 0
                      and served.get(city, 0)/demand[city] < target-1e-6)
    if not deficits:
        return design
    city = deficits[0][1]
    a = sites[city]
    direct = {frozenset(stops): i for i, (stops, ids, f) in enumerate(design) if len(stops) == 2}
    options = []
    for od in problem.demand:
        if od['origin'] != city or od['destination'] not in sites:
            continue
        b = sites[od['destination']]
        if haversine(problem.locations[a]['coordinate'], problem.locations[b]['coordinate']) > problem.config.maximum_edge_km:
            continue
        options.append((-od['volume'], b))
    for weight, b in sorted(options):
        key = frozenset((a, b))
        if key in direct:
            i = direct[key]
            stops, ids, f = design[i]
            higher = [other for other in frequencies if other > f]
            capturable = volumes.get((city, problem.locations[b]['city']), 0)*capture[city]*capture[problem.locations[b]['city']]
            if higher and capturable > f*problem.config.train_seats+1e-6:
                proposal = list(design)
                proposal[i] = stops, ids, higher[0]
                candidate = canonical(proposal)
                if candidate not in excluded:
                    return candidate
        else:
            routes = pool.get(a, b)
            if routes:
                edge = min(routes, key=lambda e: pool.edges[e]['km'])
                candidate = canonical(list(design)+[((a, b), (edge,), frequencies[0])])
                if candidate not in excluded:
                    return candidate
    return design


def gravity_pairs(problem):
    """Rank unique OD pairs by population gravity, independently of demand model."""
    c = problem.config
    pairs = {}
    for od in problem.demand:
        a, b = sorted((od['origin'], od['destination']))
        pull = (problem.cities[a].get('population', 0)*problem.cities[b].get('population', 0)
                / max(od['km'], c.distance_floor_km)**c.gravity_beta)
        pairs[a, b] = pull
    return iter(sorted(pairs, key=lambda pair: (-pairs[pair], pair)))


def gravity_proposal(problem, pool, design, pair, frequencies):
    sites = {problem.locations[s]['city']: s for stops, ids, f in design for s in stops}
    endpoints = []
    for city in pair:
        options = [s for s in problem.stations if s['city'] == city]
        if not options:
            return design
        endpoints.append(sites.get(city, max(options, key=lambda s: (s['capture_fraction'], -s['location']))['location']))
    a, b = endpoints
    if any(a in stops and b in stops for stops, ids, f in design):
        return design
    routes = pool.get(a, b)
    if not routes:
        return design
    edge = min(routes, key=lambda i: (pool.edges[i]['track_capital'], i))
    return canonical(list(design)+[((a, b), (edge,), frequencies[0])])


def network_coverage(record, config, capture_upper_bound):
    """Coverage uses fixed regional denominators, independent of the active graph."""
    if record is None:
        return 0.
    model, vector = record
    if config.annealing_coverage_metric == 'markets':
        value = sum(vector[i] for i in model.x.values())/max(1, len(model.problem.cities))
    elif config.annealing_coverage_metric == 'population':
        value = sum(vector[model.x[s['id']]]*s.get('covered_population', 0)
                    for s in model.problem.stations)/model.population_normalization
    elif config.annealing_coverage_metric == 'needs':
        value = sum(vector[i] for i in model.q.values())/model.needs_upper_bound if model.needs_upper_bound > 0 else 0.
    else:
        value = sum(vector[i] for i in model.q.values())/max(1, capture_upper_bound)
    return max(0., min(1., float(value)))


def solve_annealing(problem, state, time_limit, interrupted, checkpoint=None, reporter=None, signature=None):
    c = problem.config
    signature = signature or input_signature(problem)
    start = time.monotonic()
    bounded = c.annealing_budget_mode == 'bounded'
    remaining = lambda: time_limit - (time.monotonic()-start) if bounded else time_limit
    def check():
        if interrupted.is_set() or remaining() <= 0 or (reporter and reporter.failure):
            raise InterruptedError('Annealing search stopped')
    pool = RoutePool(problem, check, lambda message: state.update(message=message))
    rng = np.random.default_rng(c.random_seed)
    frequencies = sorted(f for f in c.frequencies_per_hour if c.minimum_service_frequency <= f <= c.track_trains_per_hour_per_direction)
    empty_deficit = (c.minimum_served_fraction+c.minimum_city_served_fraction
                     + c.minimum_origin_served_fraction * len({d['origin'] for d in problem.demand})/len(problem.cities))
    empty_money = c.profit_shortfall_weight*c.minimum_horizon_profit/annuity_factor(c.appraisal_years, c.discount_rate)/1e8
    current, components = (), (0.0, empty_money, empty_deficit)
    best = math.inf
    best_design, best_record = (), None
    state.update(phase='solving', iteration_unit='annealing_proposal', message='Growing a network from an empty board; routing corridors on demand')
    if checkpoint:
        with np.load(checkpoint, allow_pickle=False) as saved:
            if 'annealing' not in saved:
                raise ValueError('Resume requires an annealing checkpoint')
            metadata = json.loads(str(saved['annealing']))
            if metadata['signature'] != signature:
                raise ValueError('Checkpoint belongs to different assumptions or input data')
            pool.edges = metadata['edges']
            pool.cache = {}
            for e in pool.edges:
                pool.cache.setdefault(tuple(sorted((e['a'], e['b']))), []).append(e['id'])
            current = canonical([(tuple(s), tuple(e), f) for s, e, f in metadata['design']])
            used = sorted({i for stops, ids, f in current for i in ids})
            lookup = {old: new for new, old in enumerate(used)}
            edges = [dict(pool.edges[i], id=lookup[i]) for i in used]
            local = [(s, tuple(lookup[i] for i in ids), f) for s, ids, f in current]
            live = {s for stops, ids, f in current for s in stops}
            stations = [dict(s, id=i) for i, s in enumerate(s for s in problem.stations if s['location'] in live)]
            restored_problem = replace(problem, edges=edges, stations=stations)
            restored = NetworkModel(restored_problem, lines=make_lines(local, edges),
                                    capture_upper_bound=pool.capture_upper_bound, protection=metadata.get('protection'),
                                    needs_upper_bound=pool.needs_upper_bound)
            vector = saved['vector'].copy()
            if str(saved['fingerprint']) != restored.fingerprint() or not restored.validation(vector)['valid']:
                raise ValueError('Checkpoint fails model fingerprint or feasibility validation')
            restored.annealing_checkpoint = str(saved['annealing'])
            best = restored.objective(vector)
            state.update(problem=restored_problem, model=restored, vector=vector)
            best_design, best_record = current, (restored, vector)
    accepted = improvements = tested = 0
    protection, pending, cost_current = None, [], None
    cost_at_start = None
    current_record = best_record
    growth_rejected = set()
    final_evaluated = set()
    stop_reason = 'proposal_limit'
    first_preview = True
    stage = 'exploration'
    gravity_active = c.gravity_heuristic and not checkpoint
    ranked_pairs = gravity_pairs(problem) if gravity_active else iter(())
    def publish(record, design, floors=None):
        nonlocal best, best_design, best_record
        model, vector = record
        best = model.objective(vector)
        best_design, best_record = design, (model, vector.copy())
        model.annealing_checkpoint = json.dumps(dict(signature=signature, edges=pool.edges,
                                                     design=design, protection=floors))
        state.update(problem=model.problem, model=model, vector=vector.copy())
    def show(record, deficit):
        nonlocal first_preview
        model, vector = record
        solver = dict(state.snapshot()[-1], coverage_metric=c.annealing_coverage_metric,
                      coverage_fraction=network_coverage(record, c, pool.capture_upper_bound),
                      coverage_target=c.annealing_consolidation_coverage, stage=stage,
                      **connectivity(record))
        state.update(preview=(model.problem, model, vector.copy(), deficit), solver=solver)
        if reporter:
            reporter.capture_growth()
            if first_preview:
                reporter.requested.set()
        first_preview = False
    def progress_report(temperature=None, weight=None, action=None):
        state.update(solver=dict(method='annealing', status='Searching', stage=stage,
                                budget_mode=c.annealing_budget_mode,
                                elapsed_search_seconds=time.monotonic()-start,
                                action=action, temperature=temperature, financial_weight=weight,
                                **connectivity(current_record),
                                connectivity_target=c.annealing_connectivity_target,
                                coverage_metric=c.annealing_coverage_metric,
                                coverage_fraction=network_coverage(current_record, c, pool.capture_upper_bound),
                                coverage_target=c.annealing_consolidation_coverage,
                                rail_eligible_needs_per_hour=pool.needs_upper_bound,
                                accepted=accepted, terrain_searches=pool.searches,
                                cost_candidates_tested=tested, cost_improvements=improvements,
                                pending_cost_candidates=len(pending), annual_cost=cost_current,
                                consolidation_start_annual_cost=cost_at_start,
                                primal_bound=best if math.isfinite(best) else None, relative_gap=None))
        if reporter:
            reporter.notify_iterations()
    if not frequencies:
        return 'Annealing search exhausted without a validated network'
    if c.annealing_coverage_metric == 'needs' and pool.needs_upper_bound <= 0:
        state.update(solver=dict(method='annealing', status='No rail-eligible passenger needs',
                                 stopping_reason='no_eligible_needs', coverage_metric='needs',
                                 rail_eligible_needs_per_hour=0, coverage_fraction=0,
                                 coverage_target=c.annealing_consolidation_coverage, coverage_target_met=False))
        return 'Annealing search exhausted without a validated network'
    try:
        for iteration in itertools.count():
            if bounded and iteration >= c.annealing_iterations:
                stop_reason = 'proposal_limit'
                break
            check()
            if gravity_active and current_record is not None:
                served = sum(current_record[1][i] for i in current_record[0].q.values())
                gravity_active = served < c.gravity_heuristic_demand_threshold*pool.capture_upper_bound
            if gravity_active:
                pair = next(ranked_pairs, None)
                if pair is None:
                    gravity_active = False
            coverage = network_coverage(current_record, c, pool.capture_upper_bound)
            best_coverage = network_coverage(best_record, c, pool.capture_upper_bound)
            coverage_progress = min(1., coverage/c.annealing_consolidation_coverage)
            if (not gravity_active and protection is None and ready_to_consolidate(best_record, best_coverage, c)
                    and ready_to_consolidate(current_record, coverage, c)):
                # Reset the walk to a useful railway and protect this fixed service
                # baseline; do not ratchet the allowed losses after each saving.
                current, current_record = best_design, best_record
                protection = service_protection(*best_record, c)
                protection.update(coverage_metric=c.annealing_coverage_metric,
                                  minimum_coverage=c.annealing_consolidation_coverage,
                                  coverage_denominator=(len(problem.cities) if c.annealing_coverage_metric == 'markets'
                                      else best_record[0].population_normalization if c.annealing_coverage_metric == 'population'
                                      else pool.needs_upper_bound if c.annealing_coverage_metric == 'needs'
                                      else max(1, pool.capture_upper_bound)))
                cost_current = annual_cost(*best_record)
                cost_at_start = cost_current
                pending = candidates(problem, pool, current, frequencies)
                if reporter and hasattr(reporter, 'capture_exploration'):
                    reporter.capture_exploration()
                state.update(preview=None)
            if protection is not None:
                stage = 'consolidation'
                if not pending:
                    stop_reason = 'bounded_cost_neighborhood_exhausted'
                    break
                if c.speed_up and len(pending) > 10:
                    sampled = [pending[i] for i in sample_indices(len(pending), 10, rng)]
                    candidate = pop_cost_candidate(sampled, c.annealing_action_fraction, rng)
                    pending.remove(candidate)
                else:
                    candidate = pop_cost_candidate(pending, c.annealing_action_fraction, rng)
                action = candidate[1]
                state.update(message=f'{stage.capitalize()}: testing {action}; {len(pending)} cost candidates remain')
                proposal = materialize(pool, current, candidate)
                state.advance_iteration()
                tested += 1
                if proposal and proposal != current:
                    result = evaluate(problem, pool, proposal, 1., remaining, interrupted,
                                      final=True, protection=protection)
                    if result is not None:
                        record = result[:2]
                        cost = annual_cost(*record)
                        if (ready_to_consolidate(record, network_coverage(record, c, pool.capture_upper_bound), c)
                                and cost < cost_current - max(.01, abs(cost_current)*1e-9)):
                            cost_current, current, current_record = cost, proposal, record
                            accepted += 1
                            improvements += 1
                            publish(record, proposal, protection)
                            # Rebuild the bounded neighborhood around the cheaper
                            # topology. Exact cached shared edge IDs are reused.
                            pending = candidates(problem, pool, current, frequencies)
                            state.update(preview=None)
                progress_report(action=action)
                continue
            stage = 'exploration' if coverage < min(c.annealing_integration_coverage, c.annealing_consolidation_coverage) else 'integration'
            if gravity_active:
                stage = 'gravity_heuristic'
            pressure = integration_pressure(coverage, c)
            weight = c.annealing_initial_financial_weight + (1-c.annealing_initial_financial_weight)*coverage_progress
            temperature = c.annealing_start_temperature*(c.annealing_end_temperature/c.annealing_start_temperature)**coverage_progress
            if gravity_active:
                proposal, action = gravity_proposal(problem, pool, current, pair, frequencies), 'gravity_add'
            elif checkpoint and iteration == 0:
                proposal, action = current, 'resume'
                components = (math.inf, 0, 0)
            else:
                proposal = repair_origin(problem, pool, current, current_record, frequencies, growth_rejected)
                if proposal != current:
                    action = 'repair_origin'
                else:
                    proposal, action = propose(problem, pool, current, rng, frequencies, growth=True, stage=stage)
            state.advance_iteration()
            result = None
            if proposal == current and action != 'resume':
                progress_report(temperature, weight, action)
                continue
            if not proposal:
                new = (0., empty_money, empty_deficit)
            else:
                result = evaluate(problem, pool, proposal, weight, remaining, interrupted, network_pressure=pressure, conventional_only=gravity_active)
                if result is None:
                    growth_rejected.add(proposal)
                    progress_report(temperature, weight, action)
                    continue
                model, vector, service, money, deficit, local = result
                new = service, money, deficit
                if action == 'seed_region':
                    added = {problem.locations[site]['city'] for stops, ids, f in proposal for site in stops} - {problem.locations[site]['city'] for stops, ids, f in current for site in stops}
                    served_new = sum(vector[index] for (od, a, b), index in model.q.items()
                                     if problem.demand[od]['origin'] in added and problem.demand[od]['destination'] in added)
                    if served_new <= 1e-6:
                        growth_rejected.add(proposal)
                        progress_report(temperature, weight, action)
                        continue
                final_designs = connected_designs(proposal) if c.require_connected_network else [proposal]
                for final_design in final_designs:
                    if final_design in final_evaluated:
                        continue
                    final = evaluate(problem, pool, final_design, 1., remaining, interrupted, final=True, conventional_only=gravity_active)
                    if not gravity_active:
                        final_evaluated.add(final_design)
                    if final is not None:
                        candidate_coverage = network_coverage(final[:2], c, pool.capture_upper_bound)
                        old_coverage = network_coverage(best_record, c, pool.capture_upper_bound)
                        target = c.annealing_consolidation_coverage
                        better_coverage = min(candidate_coverage, target) > min(old_coverage, target)+1e-9
                        same_coverage = abs(min(candidate_coverage, target)-min(old_coverage, target)) <= 1e-9
                        if better_coverage or (same_coverage and final[0].objective(final[1]) < best):
                            publish(final[:2], final_design)
            energy = lambda values, record: growth_energy(values, record, weight, pressure, c,
                         network_coverage(record, c, pool.capture_upper_bound), pool.capture_upper_bound)
            proposed_record = result[:2] if result is not None else None
            delta = energy(new, proposed_record)-energy(components, current_record)
            if gravity_active or delta <= 0 or rng.random() < math.exp(-delta/temperature):
                current, components = proposal, new
                growth_rejected.clear()
                accepted += 1
                if result is not None:
                    current_record = result[:2]
                    show(current_record, result[4])
                else:
                    current_record = None
                    state.update(preview=None)
            else:
                growth_rejected.add(proposal)
            label = 'passenger needs satisfied' if c.annealing_coverage_metric == 'needs' else c.annealing_coverage_metric+' coverage'
            state.update(message=f'{stage.capitalize()}: {action}; {network_coverage(current_record, c, pool.capture_upper_bound):.1%} {label} / {c.annealing_consolidation_coverage:.0%} target; {len(current)} services')
            progress_report(temperature, weight, action)
        else:
            stop_reason = 'proposal_limit'
    except InterruptedError:
        stop_reason = 'interrupted' if interrupted.is_set() else 'time_limit'
    status = 'Annealing feasible' if math.isfinite(best) else 'Annealing search exhausted without a validated network'
    # The final map is the validated incumbent; preserve an explicitly labelled
    # partial railway if no design ever met the original coverage requirements.
    if reporter and hasattr(reporter, 'capture_exploration'):
        solver = dict(state.snapshot()[-1], **connectivity(current_record))
        solver['coverage_fraction'] = network_coverage(current_record, c, pool.capture_upper_bound)
        state.update(solver=solver)
        reporter.capture_exploration()
    if best_record is not None:
        state.update(preview=None)
    state.update(solver=dict(method='annealing', status=status, stage=stage,
                            stopping_reason=stop_reason, accepted=accepted,
                            budget_mode=c.annealing_budget_mode,
                            **connectivity(best_record or current_record),
                            exploratory_component_count=connectivity(current_record)['component_count'],
                            exploratory_largest_component_population_share=connectivity(current_record)['largest_component_population_share'],
                            connectivity_target=c.annealing_connectivity_target,
                            exploratory_coverage_fraction=network_coverage(current_record, c, pool.capture_upper_bound),
                            coverage_metric=c.annealing_coverage_metric,
                            coverage_fraction=network_coverage(best_record or current_record, c, pool.capture_upper_bound),
                            coverage_target=c.annealing_consolidation_coverage,
                            rail_eligible_needs_per_hour=pool.needs_upper_bound,
                            coverage_target_met=network_coverage(best_record or current_record, c, pool.capture_upper_bound)+1e-9 >= c.annealing_consolidation_coverage,
                            terrain_searches=pool.searches, cost_candidates_tested=tested,
                            cost_improvements=improvements, pending_cost_candidates=len(pending),
                            annual_cost=cost_current, consolidation_start_annual_cost=cost_at_start,
                            primal_bound=best if math.isfinite(best) else None,
                            relative_gap=None))
    return status
