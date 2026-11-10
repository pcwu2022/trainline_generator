"""Deterministic finite all-stop line patterns on the candidate track graph."""
from .geography import haversine
from itertools import combinations, product, islice


def generate_lines(problem):
    config = problem.config
    adjacency = {loc['id']: [] for loc in problem.locations}
    edge_lookup, alternatives = {}, {}
    for edge in problem.edges:
        a, b = edge['a'], edge['b']
        key = tuple(sorted((a, b)))
        alternatives.setdefault(key, []).append(edge)
    for key, choices in alternatives.items():
        a, b = key
        adjacency[a].append(b)
        adjacency[b].append(a)
        edge_lookup[key] = min(choices, key=lambda e: (e['km'], e['track_capital']))
    patterns = set()
    extra = set()
    for edge in problem.edges:
        patterns.add(tuple(sorted((edge['a'], edge['b']))))
    # Enumerate simple paths without repeating a city. Stop once a bounded
    # candidate pool is reached; larger regions should increase the explicit cap.
    pool_limit = max(1000, config.maximum_extra_lines * 100)
    def extend(path):
        if len(extra) >= pool_limit or len(path) >= config.maximum_line_stops:
            return
        cities = {problem.locations[node]['city'] for node in path}
        for neighbor in sorted(adjacency[path[-1]]):
            if problem.locations[neighbor]['city'] in cities:
                continue
            candidate = path + (neighbor,)
            canonical = min(candidate, tuple(reversed(candidate)))
            if len(candidate) >= 3:
                extra.add(canonical)
            extend(candidate)
    if config.maximum_extra_lines:
        for location in sorted(adjacency):
            extend((location,))
    def efficiency(pattern):
        km = sum(edge_lookup[tuple(sorted((a, b)))]['km'] for a, b in zip(pattern, pattern[1:]))
        direct = haversine(problem.locations[pattern[0]]['coordinate'], problem.locations[pattern[-1]]['coordinate'])
        return (km/max(direct, 1), -len(pattern), km, pattern)
    # Favor through lines with little detour. All two-stop corridors remain
    # available even when the optional multi-stop candidate budget is zero.
    patterns.update(sorted(extra, key=efficiency)[:config.maximum_extra_lines])
    # Reserve a separate pool for hub services. Generic low-detour ranking can
    # discard these before the MILP ever gets to evaluate shared infrastructure.
    # Score is an optimistic saving if one spoke is already built for another
    # service, not a discount to the actual objective or construction costs.
    hubs = []
    if config.maximum_line_stops >= 3 and config.maximum_shared_track_lines:
        for hub in sorted(adjacency):
            for a, b in combinations(sorted(adjacency[hub]), 2):
                if len({problem.locations[n]['city'] for n in (a, hub, b)}) != 3:
                    continue
                pattern = min((a, hub, b), (b, hub, a))
                if pattern in patterns or efficiency(pattern)[0] > config.maximum_shared_track_detour:
                    continue
                spokes = [edge_lookup[tuple(sorted((n, hub)))] for n in (a, b)]
                direct = edge_lookup.get(tuple(sorted((a, b))))
                avoided = direct['track_capital'] if direct else sum(e['track_capital'] for e in spokes)
                saving = avoided - min(e['track_capital'] for e in spokes) - config.station_capital_cost
                hubs.append((-saving, efficiency(pattern), pattern))
        patterns.update(pattern for _, _, pattern in sorted(hubs)[:config.maximum_shared_track_lines])
    lines = []
    for stops in sorted(patterns, key=lambda p: (len(p), p)):
        choices = [sorted(alternatives[tuple(sorted((a, b)))],
                          key=lambda e: (e['track_capital'], e['km'], e['id']))
                   for a, b in zip(stops, stops[1:])]
        # Keep fastest and cheapest combinations, then bounded mixed choices.
        fastest = tuple(min(options, key=lambda e: (e['km'], e['track_capital'])) for options in choices)
        cheapest = tuple(options[0] for options in choices)
        variants = [fastest, cheapest]
        variants.extend(islice(product(*choices), config.maximum_alignment_variants_per_line))
        seen = set()
        for edges in variants:
            ids = tuple(e['id'] for e in edges)
            if ids in seen:
                continue
            if len(seen) >= config.maximum_alignment_variants_per_line:
                break
            seen.add(ids)
            geometry = []
            for a, b, edge in zip(stops, stops[1:], edges):
                segment = edge['geometry'] if edge['a'] == a else list(reversed(edge['geometry']))
                geometry.extend(segment if not geometry else segment[1:])
            lines.append(dict(id=len(lines), a=stops[0], b=stops[-1], stops=list(stops),
                              edges=list(ids), segment_km=[e['km'] for e in edges],
                              alignment_kinds=[e.get('alignment_kind', 'legacy') for e in edges],
                              km=sum(e['km'] for e in edges), geometry=geometry))
    return lines
