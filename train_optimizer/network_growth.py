"""Regional exploration and demand-ranked connections between existing networks."""
from collections import defaultdict

import numpy as np

from .geography import haversine
from .sampling import sample_indices, sample_pairs


def components(sites, links):
    adjacency = {site: set() for site in sites}
    for a, b in links:
        adjacency[a].add(b)
        adjacency[b].add(a)
    result = []
    unseen = set(sites)
    while unseen:
        reached, pending = set(), [min(unseen)]
        while pending:
            site = pending.pop()
            if site not in reached:
                reached.add(site)
                pending.extend(adjacency[site]-reached)
        unseen -= reached
        result.append(reached)
    return result


def connected_designs(design):
    """Individual valid-result candidates, without discarding the exploratory walk."""
    sites = {s for stops, ids, f in design for s in stops}
    groups = components(sites, [(a, b) for stops, ids, f in design for a, b in zip(stops, stops[1:])])
    return [tuple(entry for entry in design if entry[0][0] in group)
            for group in sorted(groups, key=lambda group: (-len(group), min(group)))]


def connectivity(record):
    if record is None:
        return dict(component_count=0, largest_component_population_share=0.)
    model, vector = record
    sites = {s['location'] for s in model.problem.stations if vector[model.x[s['id']]] > .5}
    links = [(e['a'], e['b']) for e in model.problem.edges if vector[model.y[e['id']]] > .5]
    groups = components(sites, links)
    population = {s['location']: s.get('covered_population',
                  model.problem.cities[s['city']].get('population', 1)) for s in model.problem.stations}
    total = sum(population[site] for site in sites)
    if total <= 0:
        population = {site: 1 for site in sites}
        total = len(sites)
    share = max((sum(population[site] for site in group)/max(1, total) for group in groups), default=0.)
    return dict(component_count=len(groups), largest_component_population_share=float(share))


def integration_pressure(coverage, config):
    start = min(config.annealing_integration_coverage, config.annealing_consolidation_coverage)
    if coverage < start:
        return 0.
    return min(1., (coverage-start)/max(1e-9, config.annealing_consolidation_coverage-start)) if start < config.annealing_consolidation_coverage else 1.


def ready_to_consolidate(record, coverage, config):
    if record is None or coverage+1e-9 < config.annealing_consolidation_coverage:
        return False
    stats = connectivity(record)
    return (stats['largest_component_population_share']+1e-9 >= config.annealing_connectivity_target
            and (not config.require_connected_network or stats['component_count'] == 1))


def growth_energy(values, record, financial_weight, pressure, config, coverage, capture_bound):
    model, vector = record if record is not None else (None, None)
    trips = sum(vector[i] for i in model.q.values()) if model is not None else 0.
    share = connectivity(record)['largest_component_population_share']
    return (values[0]+financial_weight*values[1]
            +config.annealing_coverage_penalty*(values[2]+max(0, config.annealing_consolidation_coverage-coverage))
            +config.annealing_connectivity_penalty*pressure*max(0, config.annealing_connectivity_target-share)
            -pressure*config.annealing_network_effect_weight*config.served_weight*trips/max(1, capture_bound))


def regional_proposal(problem, pool, design, rng, frequencies, stage, allowed=None, preferred_action=None):
    from .annealing import canonical
    c = problem.config
    allowed = set(allowed if allowed is not None else ('seed_region', 'frontier', 'bridge'))
    limit = min(c.annealing_candidate_pair_limit or 10, 10) if c.speed_up else c.annealing_candidate_pair_limit
    indices = sample_indices(len(problem.demand), limit, rng)
    live = {site for stops, ids, f in design for site in stops}
    active = {problem.locations[site]['city']: site for site in live}
    groups = components(live, [(a, b) for stops, ids, f in design for a, b in zip(stops, stops[1:])])
    component = {problem.locations[site]['city']: i for i, group in enumerate(groups) for site in group}
    existing = {frozenset(stops) for stops, ids, f in design if len(stops) == 2}
    city_pair = {frozenset(problem.locations[site]['city'] for site in stops)
                 for stops, ids, f in design if len(stops) == 2}
    if not hasattr(pool, '_market_centers'):
        fallback = {}
        for site in problem.locations:
            fallback.setdefault(site['city'], site['coordinate'])
        pool._market_centers = {city['id']: city.get('center', fallback.get(city['id']))
                                for city in problem.cities}
    centers = pool._market_centers
    base = pool.demand_weights(indices)
    if not hasattr(pool, '_market_distances'):
        pool._market_distances = {}
    def distance(a, b):
        key = tuple(sorted((a, b)))
        if key not in pool._market_distances:
            pool._market_distances[key] = haversine(centers[a], centers[b])
        return pool._market_distances[key]
    nearest = {}
    def novelty_distance(city):
        if city not in nearest:
            reference = sorted(active)
            sampled = sample_indices(len(reference), limit, rng)
            nearest[city] = min((distance(city, reference[i]) for i in sampled),
                                default=c.annealing_exploration_radius_km)
        return nearest[city]
    capture = {s['city']: s['capture_fraction'] for s in problem.stations if s['location'] in live}
    cross = defaultdict(float)
    for index in indices:
        od = problem.demand[index]
        a, b = od['origin'], od['destination']
        if a in component and b in component and component[a] != component[b]:
            cross[tuple(sorted((component[a], component[b])))] += od['volume']*capture[a]*capture[b]
    categories = dict(seed_region=[], frontier=[], bridge=[])
    for index in indices:
        od = problem.demand[index]
        a, b = od['origin'], od['destination']
        if centers[a] is None or centers[b] is None or base[index] <= 0 or frozenset((a, b)) in city_pair:
            continue
        km = distance(a, b)
        if km > c.maximum_edge_km:
            continue
        # A local seed must have a plausible passenger journey; full access,
        # capacity, waits and terrain are still evaluated by the LP afterward.
        if 'seed_region' in allowed and a not in active and b not in active:
            if km/c.train_speed_kmh+.5/frequencies[0] > c.maximum_journey_hours:
                continue
            novelty = .05+.95*min(1, min(novelty_distance(a), novelty_distance(b))/c.annealing_exploration_radius_km)
            categories['seed_region'].append((index, base[index]*novelty))
        elif 'frontier' in allowed and (a in active) != (b in active):
            categories['frontier'].append((index, base[index]))
    # Endpoints need not themselves form an OD market: a short bridge can
    # unlock demand between other stations on either side of the connection.
    for a, b in sample_pairs(sorted(active), limit, rng) if 'bridge' in allowed else ():
        if component[a] == component[b] or frozenset((a, b)) in city_pair:
            continue
        km = distance(a, b)
        if km > c.maximum_edge_km:
            continue
        aggregate = cross[tuple(sorted((component[a], component[b])))]
        capital = getattr(pool, 'capital_estimates', {}).get((a, b), km*c.base_track_cost_per_km)
        relative = max(1., capital/max(1, c.base_track_cost_per_km*c.distance_floor_km))
        weight = aggregate/relative**c.annealing_proposal_cost_exponent
        if weight > 0:
            categories['bridge'].append(((a, b), weight))
    if stage == 'exploration':
        preferred = 'seed_region' if rng.random() < .8 else 'frontier'
        order = [preferred, 'frontier' if preferred == 'seed_region' else 'seed_region', 'bridge']
    else:
        preferred = 'bridge' if rng.random() < .7 else 'frontier'
        order = [preferred, 'frontier' if preferred == 'bridge' else 'bridge', 'seed_region']
    if preferred_action is not None:
        order = [preferred_action] + [kind for kind in order if kind != preferred_action]
    for kind in order:
        options = categories[kind] if kind in allowed else []
        if not options:
            continue
        weights = np.asarray([w for _, w in options], dtype=float)
        selected = int(rng.choice(len(options), p=weights/weights.sum()))
        chosen = options[selected][0]
        if kind == 'bridge':
            pair = chosen
        else:
            od = problem.demand[chosen]
            pair = od['origin'], od['destination']
        endpoints = []
        for city in pair:
            choices = [s['location'] for s in problem.stations if s['city'] == city]
            if not choices:
                return design, kind
            endpoints.append(active[city] if city in active else int(rng.choice(choices)))
        a, b = endpoints
        routes = pool.get(a, b)
        if routes:
            # Cheapest generated alternative; speed/capacity tradeoffs are checked
            # by passenger evaluation and subsequent service adjustments.
            usable = ([i for i in routes if pool.edges[i]['km']/c.train_speed_kmh+.5/frequencies[0] <= c.maximum_journey_hours]
                      if kind == 'seed_region' else routes)
            if not usable:
                return design, kind
            edge = min(usable, key=lambda i: pool.edges[i]['track_capital'])
            if frozenset((a, b)) not in existing:
                return canonical(list(design)+[((a, b), (edge,), frequencies[0])]), kind
        return design, kind
    return design, 'no_regional_candidate'
