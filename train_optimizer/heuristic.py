"""Construct connected networks and improve them with fixed-design LP evaluations.

Station and service decisions are heuristic; HiGHS only assigns passengers and
connectivity flows for each fixed design. No branch-and-bound search is used.
"""
from dataclasses import replace
import math
import time

import highspy
import numpy as np


def heuristic_problem(problem):
    """Choose one station per market and retain its candidate corridors."""
    stations = []
    for city in problem.cities:
        options = [s for s in problem.stations if s['city'] == city['id']]
        if options:
            best = max(options, key=lambda s: (s['capture_fraction'],
                       -s.get('access_hours', 0), -s['id']))
            stations.append(dict(best, id=len(stations)))
    live = {s['location'] for s in stations}
    edges = [dict(edge, id=i) for i, edge in enumerate(
             e for e in problem.edges if e['a'] in live and e['b'] in live)]
    config = replace(problem.config, maximum_line_stops=min(3, problem.config.maximum_line_stops),
                     maximum_extra_lines=min(problem.config.maximum_extra_lines, problem.config.heuristic_through_lines),
                     maximum_shared_track_lines=0, maximum_alignment_variants_per_line=2)
    return replace(problem, config=config, stations=stations, edges=edges,
                   warnings=list(problem.warnings) + [
                       'Heuristic search fixes one station location per market and uses a bounded direct/through-service pool; no optimality guarantee.'])


def spanning_tree(model, rng, distance=False, randomized=False):
    """Kruskal tree on the largest component; infeasible designs are rejected later."""
    edges = model.problem.edges
    parent = {s['location']: s['location'] for s in model.problem.stations}
    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def weight(edge):
        value = edge['km'] if distance else edge['track_capital']
        return value * (rng.uniform(.6, 1.4) if randomized else 1)
    ordered = sorted(edges, key=lambda e: (weight(e), e['id']))
    chosen = set()
    for edge in ordered:
        a, b = root(edge['a']), root(edge['b'])
        if a != b:
            parent[a] = b
            chosen.add(edge['id'])
    # Restrict to one connected component; required coverage is still checked by LP.
    groups = {}
    for node in parent:
        groups.setdefault(root(node), set()).add(node)
    if not groups:
        return set()
    largest = max(groups.values(), key=lambda group: (len(group), -min(group)))
    return {i for i in chosen if edges[i]['a'] in largest}


def service_design(model, edges, frequency):
    """Convert a physical backbone to direct services, preserving alignment IDs."""
    direct = {line['edges'][0]: line['id'] for line in model.lines if len(line['edges']) == 1}
    return {direct[e]: frequency for e in edges}


def journey_times(model, design):
    """Fast all-pairs generalized train times, including waits and transfers."""
    c = model.problem.config
    n = len(model.problem.locations)
    times = np.full((n, n), np.inf)
    np.fill_diagonal(times, 0)
    for line_id, frequency in design.items():
        line = model.lines[line_id]
        for i, a in enumerate(line['stops']):
            for j, b in enumerate(line['stops']):
                if i == j:
                    continue
                lo, hi = sorted((i, j))
                hours = (sum(line['segment_km'][lo:hi])/line['speed_kmh']
                         + (hi-lo-1)*c.dwell_minutes/60 + .5/frequency + c.transfer_minutes/60)
                times[a, b] = min(times[a, b], hours)
    for k in range(n):
        times = np.minimum(times, times[:, k, None] + times[None, k, :])
    return times


def neighborhood(model, design, frequencies):
    """Demand/time-guided shortcuts, through replacements, bypasses and pruning.

    Ranking guides evaluation only; acceptance always uses the full LP objective.
    """
    p, c = model.problem, model.problem.config
    station = {s['city']: s for s in p.stations}
    ods = [(station[d['origin']]['location'], station[d['destination']]['location'],
            d['volume']*station[d['origin']]['capture_fraction']*station[d['destination']]['capture_fraction'])
           for d in p.demand if d['origin'] in station and d['destination'] in station]
    def proxy(candidate):
        times = journey_times(model, candidate)
        total = 0
        for a, b, volume in ods:
            hours = times[a, b] - c.transfer_minutes/60
            if hours <= c.maximum_journey_hours:
                total += volume*(c.served_weight-c.time_weight*hours)/max(1, model.capture_upper_bound)
        # Charge incremental fleet and unique infrastructure in the ranking.
        edges = {e for line in candidate for e in model.lines[line]['edges']}
        total -= sum(model.cost[model.z[line, frequency]] for line, frequency in candidate.items())
        total -= sum(model.cost[model.y[e]] for e in edges)
        high_speed_edges = {e for line in candidate if model.lines[line]['train_type'] == 'HSR'
                            for e in model.lines[line]['edges']}
        total -= sum(model.cost[model.hsr_tracks[e]] for e in high_speed_edges)
        return total
    candidates = {}
    def add(candidate):
        key = tuple(sorted(candidate.items()))
        if key != tuple(sorted(design.items())):
            candidates[key] = candidate
    for line_id, frequency in sorted(design.items()):
        removed = dict(design)
        del removed[line_id]
        add(removed)
        for other in frequencies:
            if other != frequency:
                add(design | {line_id: other})
    for line in model.lines:
        if line['id'] in design:
            continue
        # Through trains replace constituent shuttles, eliminating transfers.
        replaced = {i: f for i, f in design.items()
                    if not (set(model.lines[i]['edges']) <= set(line['edges'])
                            or model.lines[i]['stops'] == line['stops'])}
        frequency = max((f for i, f in design.items()
                         if set(model.lines[i]['edges']) <= set(line['edges'])), default=frequencies[0])
        add(replaced | {line['id']: frequency})
        add(design | {line['id']: frequencies[0]})
        # Replace an alignment variant without keeping the old service running.
        if len(line['stops']) == 2:
            a, b = line['stops']
            incident_a = {node for i in design for node in model.lines[i]['stops'] if a in model.lines[i]['stops']}
            incident_b = {node for i in design for node in model.lines[i]['stops'] if b in model.lines[i]['stops']}
            for hub in (incident_a & incident_b) - {a, b}:
                bypass = {i: f for i, f in design.items() if hub not in model.lines[i]['stops']}
                add(bypass | {line['id']: frequency})
    return sorted(candidates.values(), key=lambda proposal: (-proxy(proposal), tuple(sorted(proposal.items()))))


def solve_heuristic(model, state, time_limit, interrupted, checkpoint=None, reporter=None):
    config = model.problem.config
    state.update(iteration_unit='heuristic_design_evaluation')
    rng = np.random.default_rng(config.random_seed)
    start = time.monotonic()
    solver = model.solver(config, time_limit)
    indices = np.flatnonzero(model.integer).astype(np.int32)
    frequencies = sorted(f for f in config.frequencies_per_hour
                         if f >= config.minimum_service_frequency)
    # Physical track throughput is a hard limit even in heuristic mode.
    frequencies = [f for f in frequencies if f <= config.track_trains_per_hour_per_direction]
    if not frequencies:
        state.update(solver=dict(method='heuristic', status='Configured frequencies exceed track capacity', relative_gap=None))
        return 'Heuristic search exhausted without a validated network'
    best_vector, best_design, best_value = None, None, math.inf
    if checkpoint:
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['fingerprint']) != model.fingerprint():
                raise ValueError('Checkpoint belongs to a different model/configuration/input dataset')
            vector = saved['vector'].copy()
        if not model.validation(vector)['valid']:
            raise ValueError('Checkpoint fails current model feasibility validation')
        best_vector, best_value = vector, model.objective(vector)
        best_design = {line: f for (line, f), index in model.z.items() if vector[index] > .5}
        state.update(vector=vector.copy())
    def stop(event):
        if interrupted.is_set() or (reporter and reporter.failure):
            event.interrupt()
    solver.cbSimplexInterrupt.subscribe(stop)
    solver.cbIpmInterrupt.subscribe(stop)
    cheap = spanning_tree(model, rng)
    short = spanning_tree(model, rng, distance=True)
    # Include a dense direct-service fallback before local improvement: trees
    # can fail journey limits or bottleneck demand despite adequate corridors.
    pairs = {}
    for edge in sorted(model.problem.edges, key=lambda e: (e['km'], e['track_capital'], e['id'])):
        pairs.setdefault(tuple(sorted((edge['a'], edge['b']))), edge['id'])
    initial = [service_design(model, set(pairs.values()), f) for f in (frequencies[0], frequencies[-1])]
    initial.extend(service_design(model, tree, frequencies[0]) for tree in (cheap, short))
    seen = set()
    pending = []
    evaluations = 0
    state.update(phase='solving', message='Demand-guided search: direct routes, through services and network improvements')
    for iteration in range(config.heuristic_iterations):
        if interrupted.is_set() or (reporter and reporter.failure) or time.monotonic()-start >= time_limit:
            break
        if iteration < len(initial):
            design = initial[iteration]
        elif pending:
            design = pending.pop(0)
        else:
            tree = spanning_tree(model, rng, distance=bool(iteration % 2), randomized=True)
            design = service_design(model, tree, int(rng.choice(frequencies)))
        key = tuple(sorted(design.items()))
        if key in seen:
            continue
        seen.add(key)
        live = {node for line in design for node in model.lines[line]['stops']}
        fixed = np.zeros(len(model.names))
        for s in model.problem.stations:
            fixed[model.x[s['id']]] = int(s['location'] in live)
        for line, frequency in design.items():
            for edge in model.lines[line]['edges']:
                fixed[model.y[edge]] = 1
            fixed[model.z[line, frequency]] = 1
            if model.lines[line]['train_type'] == 'HSR':
                for edge in model.lines[line]['edges']:
                    fixed[model.hsr_tracks[edge]] = 1
        if config.require_connected_network and live:
            fixed[model.names.index(f'root_{min(live)}')] = 1
        bounds = fixed[indices]
        if solver.changeColsBounds(len(indices), indices, bounds, bounds) != highspy.HighsStatus.kOk:
            raise RuntimeError('Could not fix heuristic design')
        # Every integer decision is fixed: the remaining problem is continuous.
        solver.run()
        evaluations += 1
        solution = solver.getSolution()
        if solution.value_valid:
            vector = np.asarray(solution.col_value, dtype=float)
            if (np.allclose(vector[indices], bounds, atol=1e-5, rtol=0)
                    and model.validation(vector)['valid']):
                value = model.objective(vector)
                if value < best_value - 1e-9:
                    best_value, best_vector, best_design = value, vector.copy(), dict(design)
                    state.update(vector=best_vector.copy())
                    pending = [proposal for proposal in neighborhood(model, best_design, frequencies)
                               if tuple(sorted(proposal.items())) not in seen]
        state.advance_iteration()
        state.update(solver=dict(method='heuristic', status='Searching', evaluations=evaluations,
                                primal_bound=best_value if best_vector is not None else None,
                                dual_bound=None, relative_gap=None, nodes=0))
        if reporter:
            reporter.notify_iterations()
    status = 'Heuristic feasible' if best_vector is not None else 'Heuristic search exhausted without a validated network'
    state.update(solver=dict(method='heuristic', status=status, evaluations=evaluations,
                            primal_bound=best_value if best_vector is not None else None,
                            dual_bound=None, relative_gap=None, nodes=0))
    return status
