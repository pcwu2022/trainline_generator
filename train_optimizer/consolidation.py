"""Rank compound cost-saving moves before performing expensive terrain searches."""
from collections import Counter, defaultdict
import heapq
import itertools

from .finance import asset_pv_factor
from .geography import haversine


def annual_cost(model, vector):
    c = model.problem.config
    return sum(vector[i] * (capital * asset_pv_factor(c.asset_lives_years[model.asset_kind[i]],
               c.appraisal_years, c.discount_rate) / model.appraisal_annuity + model.operating[i])
               for i, capital in model.capital.items())


def service_protection(model, vector, config):
    trips = sum(vector[i] for i in model.q.values())
    origins = defaultdict(float)
    for (od, a, b), index in model.q.items():
        origins[str(model.problem.demand[od]['origin'])] += vector[index]
    hours = sum(vector[i]*value for i, value in model.time.items())
    return dict(minimum_trips=config.annealing_retained_ridership*trips,
                minimum_by_origin={i: config.annealing_retained_ridership*q for i, q in sorted(origins.items())},
                maximum_mean_hours=hours/max(1e-9, trips)*(1+config.annealing_journey_time_allowance))


def existing_path(edges, used, a, b, excluded, forbidden, maximum_edges):
    adjacency = defaultdict(list)
    for i in sorted(used-{excluded}):
        e = edges[i]
        adjacency[e['a']].append((e['b'], i, e['km']))
        adjacency[e['b']].append((e['a'], i, e['km']))
    queue = [(0., (a,), ())]
    best = {}
    while queue:
        distance, stops, ids = heapq.heappop(queue)
        node = stops[-1]
        if node == b:
            return stops, ids
        if len(ids) >= maximum_edges:
            continue
        for neighbor, edge, km in adjacency[node]:
            if neighbor in stops or neighbor in forbidden:
                continue
            key = neighbor, len(ids)+1
            if distance+km < best.get(key, float('inf')):
                best[key] = distance+km
                heapq.heappush(queue, (distance+km, stops+(neighbor,), ids+(edge,)))
    return None


def candidates(problem, pool, design, frequencies):
    """Bounded, diverse lists of cheap estimates; no terrain routing occurs here."""
    c = problem.config
    counts = Counter(e for stops, ids, f in design for e in ids)
    used = set(counts)
    live = {s for stops, ids, f in design for s in stops}
    active_city = {problem.locations[s]['city']: s for s in live}
    sites = sorted({s['location'] for s in problem.stations
                    if s['city'] not in active_city or s['location'] == active_city[s['city']]})
    groups = defaultdict(list)
    def distance(a, b):
        return haversine(problem.locations[a]['coordinate'], problem.locations[b]['coordinate'])
    def estimate(a, b):
        if any({pool.edges[e]['a'], pool.edges[e]['b']} == {a, b} for e in used):
            return 0.
        return distance(a, b)*c.base_track_cost_per_km
    def add(kind, args, saving):
        groups[kind].append((float(saving), kind, args))
    for old in sorted(used):
        edge = pool.edges[old]
        for new in pool.cache.get(tuple(sorted((edge['a'], edge['b']))), []):
            if new != old and pool.edges[new]['track_capital'] < edge['track_capital']:
                add('alignment', (old, new), edge['track_capital']-pool.edges[new]['track_capital'])
    segments = []
    for i, (stops, ids, f) in enumerate(design):
        for k, edge_id in enumerate(ids):
            a, b = stops[k:k+2]
            edge = pool.edges[edge_id]
            credit = edge['track_capital'] if counts[edge_id] == 1 else 0.
            segments.append((credit, i, k, a, b))
            path = existing_path(pool.edges, used, a, b, edge_id, set(stops)-{a, b},
                                 c.maximum_line_stops-len(stops)+1)
            if path and len({problem.locations[s]['city'] for s in path[0]}) == len(path[0]):
                add('reuse', (i, k, path), credit)
            if len(stops) < c.maximum_line_stops:
                for hub in sorted(live-set(stops)):
                    detour = distance(a, hub)+distance(hub, b)
                    if detour <= c.maximum_shared_track_detour*max(1e-9, edge['km']):
                        saving = credit-estimate(a, hub)-estimate(hub, b)
                        if saving > 0:
                            add('hub', (i, k, hub), saving)
        for frequency in frequencies:
            if frequency < f:
                add('frequency', (i, frequency), (f-frequency)*sum(pool.edges[e]['km'] for e in ids)*c.train_cost_per_km)
        add('remove', (i,), sum(pool.edges[e]['track_capital'] for e in ids if counts[e] == 1))
    # Move a site's complete bundle of incident services, preserving one site/market.
    for old in sorted(live):
        city = problem.locations[old]['city']
        neighbors = {b if a == old else a for stops, ids, f in design
                     for a, b in zip(stops, stops[1:]) if old in (a, b)}
        for station in problem.stations:
            new = station['location']
            if station['city'] == city and new != old:
                saving = sum(distance(old, n)-distance(new, n) for n in neighbors)*c.base_track_cost_per_km
                add('move', (old, new), saving)
    # Open one prepared intermediate station and reroute two branches together.
    # This compound action creates a common trunk even if either edit alone loses.
    for first, second in itertools.combinations(sorted(segments, reverse=True)[:30], 2):
        credit1, i, k, a, b = first
        credit2, j, t, x, y = second
        if i == j or not {a, b} & {x, y}:
            continue
        common = min({a, b} & {x, y})
        left, right = (b if a == common else a), (y if x == common else x)
        if left == right:
            continue
        if any(len(design[n][0]) >= c.maximum_line_stops for n in (i, j)):
            continue
        for hub in sites:
            if hub in set(design[i][0]) | set(design[j][0]):
                continue
            if any(distance(common, hub)+distance(hub, end) > c.maximum_shared_track_detour*max(1e-9, distance(common, end)) for end in (left, right)):
                continue
            saving = credit1+credit2-estimate(common, hub)-estimate(hub, left)-estimate(hub, right)
            if hub not in live:
                saving -= c.station_capital_cost
            if saving > 0:
                add('junction_station', (i, k, j, t, hub), saving)
    for i, j in itertools.combinations(range(len(design)), 2):
        a, ea, fa = design[i]
        b, eb, fb = design[j]
        for aa, ae in ((a, ea), (a[::-1], ea[::-1])):
            for bb, be in ((b, eb), (b[::-1], eb[::-1])):
                stops = aa+bb[1:]
                if aa[-1] == bb[0] and len(stops) <= c.maximum_line_stops and len({problem.locations[s]['city'] for s in stops}) == len(stops):
                    add('through', (i, j, (stops, ae+be, max(fa, fb))), 0.)
    # Give each move family a quota so large hub lists do not crowd out relocation.
    limit = c.annealing_consolidation_candidates
    quota = max(1, limit//max(1, len(groups)))
    selected, rest = [], []
    for kind in sorted(groups):
        ordered = sorted(groups[kind], key=lambda item: (-item[0], repr(item[2])))
        selected.extend(ordered[:quota])
        rest.extend(ordered[quota:])
    selected.extend(sorted(rest, key=lambda item: (-item[0], item[1], repr(item[2])))[:max(0, limit-len(selected))])
    return sorted(selected, key=lambda item: (-item[0], item[1], repr(item[2])))[:limit]


def materialize(pool, design, candidate):
    from .annealing import canonical
    saving, kind, args = candidate
    used = {e for stops, ids, f in design for e in ids}
    def link(a, b):
        choices = [e for e in sorted(used) if {pool.edges[e]['a'], pool.edges[e]['b']} == {a, b}]
        choices = choices or pool.get(a, b)
        return min(choices, key=lambda e: pool.edges[e]['track_capital']) if choices else None
    proposal = list(design)
    def via(i, k, hub):
        stops, ids, f = proposal[i]
        left, right = link(stops[k], hub), link(hub, stops[k+1])
        if left is None or right is None:
            return False
        proposal[i] = (stops[:k+1]+(hub,)+stops[k+1:], ids[:k]+(left, right)+ids[k+1:], f)
        return True
    if kind == 'alignment':
        old, new = args
        proposal = [(stops, tuple(new if e == old else e for e in ids), f)
                    for stops, ids, f in design]
    elif kind == 'reuse':
        i, k, (path, edges) = args
        stops, ids, f = proposal[i]
        proposal[i] = (stops[:k]+path+stops[k+2:], ids[:k]+edges+ids[k+1:], f)
    elif kind == 'hub':
        if not via(*args):
            return design
    elif kind == 'junction_station':
        i, k, j, t, hub = args
        if not via(i, k, hub) or not via(j, t, hub):
            return design
    elif kind == 'move':
        old, new = args
        for i, (stops, ids, f) in enumerate(design):
            path = tuple(new if s == old else s for s in stops)
            edges = list(ids)
            for k, (a, b) in enumerate(zip(path, path[1:])):
                if old in stops[k:k+2]:
                    edges[k] = link(a, b)
                    if edges[k] is None:
                        return design
            proposal[i] = path, tuple(edges), f
    elif kind == 'frequency':
        i, f = args
        proposal[i] = proposal[i][0], proposal[i][1], f
    elif kind == 'remove':
        proposal.pop(args[0])
    elif kind == 'through':
        i, j, line = args
        proposal = [entry for k, entry in enumerate(proposal) if k not in (i, j)] + [line]
    return canonical(proposal)
