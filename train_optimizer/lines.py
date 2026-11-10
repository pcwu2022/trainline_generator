"""Deterministic finite all-stop line patterns on the candidate track graph."""
from .geography import haversine


def generate_lines(problem):
    config = problem.config
    adjacency = {loc['id']: [] for loc in problem.locations}
    edge_lookup = {}
    for edge in problem.edges:
        a, b = edge['a'], edge['b']
        adjacency[a].append(b)
        adjacency[b].append(a)
        edge_lookup[tuple(sorted((a, b)))] = edge
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
    lines = []
    for stops in sorted(patterns, key=lambda p: (len(p), p)):
        edges = [edge_lookup[tuple(sorted((a, b)))] for a, b in zip(stops, stops[1:])]
        geometry = []
        for a, b, edge in zip(stops, stops[1:], edges):
            segment = edge['geometry'] if edge['a'] == a else list(reversed(edge['geometry']))
            geometry.extend(segment if not geometry else segment[1:])
        lines.append(dict(id=len(lines), a=stops[0], b=stops[-1], stops=list(stops),
                          edges=[e['id'] for e in edges], segment_km=[e['km'] for e in edges],
                          km=sum(e['km'] for e in edges), geometry=geometry))
    return lines
