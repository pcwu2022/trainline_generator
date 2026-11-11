"""Grow one persistent network, then reduce its cost while retaining service.

Every neighbourhood is made from the previous accepted genome. Evaluated
alternatives also populate a Pareto archive, independently of the live network.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import math
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .evaluator import Evaluator
from .search import SearchResult, SearchUpdate, bridge_components, gravity_scores, nondominated


def neighbours(current, evaluator):
    """Single-edge edits plus a cheapest path joining existing components."""
    x = current.genome
    proposals = {}
    served = np.zeros(evaluator.n, dtype=bool)
    active = np.flatnonzero(x)
    served[evaluator.a[active]] = True
    served[evaluator.b[active]] = True
    demand = gravity_scores(evaluator)
    for i, edge in enumerate(evaluator.graph.edges):
        capital = min(a.capex_meur for a in edge.alignments.values())
        preference = max(1e-12, demand[i] / max(1, capital))
        preference *= 1 + 4 * (int(served[edge.a]) + int(served[edge.b]))
        for state in np.flatnonzero(evaluator.allowed[i]):
            if state == x[i]:
                continue
            proposal = x.copy()
            proposal[i] = state
            proposals[tuple(proposal)] = (proposal, preference)
    bridge = bridge_components(x.copy(), evaluator, None)
    if not np.array_equal(bridge, x):
        proposals[tuple(bridge)] = (bridge, max((v[1] for v in proposals.values()), default=1))
    return proposals


def growth_choice(current, candidates):
    """Prefer useful ridership gains per added annual euro; never reset the network."""
    useful = []
    for candidate in candidates:
        delta_trips = candidate.metrics['passengers'] - current.metrics['passengers']
        delta_population = candidate.metrics['population_served'] - current.metrics['population_served']
        delta_cost = candidate.objectives[0] - current.objectives[0]
        if delta_trips < -1e-6 or delta_population < -1e-6:
            continue
        if delta_trips <= 1e-6 and delta_cost >= -.01:
            continue
        # A cost-saving edit that preserves service is always attractive.
        rank = (delta_cost < -.01, delta_trips / max(1e6, delta_cost),
                delta_population, -candidate.objectives[0])
        useful.append((rank, candidate))
    return max(useful, key=lambda value: value[0])[1] if useful else None


def consolidation_choice(current, candidates, baseline, config):
    """Retain a fixed growth baseline, preventing repeated percentage erosion."""
    p = config['search']
    eligible = [candidate for candidate in candidates
        if candidate.objectives[0] < current.objectives[0] - .01
        and candidate.metrics['passengers'] + 1e-6 >= baseline['passengers'] * p['retained_ridership']
        and candidate.metrics['population_served'] + 1e-6 >= baseline['population_served'] * p['retained_ridership']
        and candidate.metrics['mean_generalized_hours'] <= baseline['mean_generalized_hours'] * (1 + p['journey_time_allowance']) + 1e-9]
    return min(eligible, key=lambda candidate: (candidate.objectives[0], -candidate.metrics['passengers'])) if eligible else None


def topology(genome, evaluator):
    """Count missing candidate stations and excess active components separately."""
    active = np.flatnonzero(genome)
    served = np.unique(np.r_[evaluator.a[active], evaluator.b[active]])
    links = sparse.csr_matrix((np.ones(2 * len(active)),
        (np.r_[evaluator.a[active], evaluator.b[active]],
         np.r_[evaluator.b[active], evaluator.a[active]])),
        shape=(evaluator.n, evaluator.n))
    _, labels = connected_components(links, directed=False)
    return evaluator.n - len(served), max(0, len(np.unique(labels[served])) - 1)


def integration_choice(current, candidates, evaluator):
    missing, components = topology(current.genome, evaluator)
    eligible = []
    for candidate in candidates:
        new_missing, new_components = topology(candidate.genome, evaluator)
        # Do not trade existing coverage away to join other components.
        if new_missing <= missing and new_missing + new_components < missing + components:
            eligible.append(candidate)
    return min(eligible, key=lambda candidate: (
        sum(topology(candidate.genome, evaluator)), candidate.objectives[0],
        -candidate.metrics['passengers'])) if eligible else None


def search_incremental(graph, config, progress=print, observer=None, interrupted=None):
    p = config['search']
    evaluator = Evaluator(graph, config)
    rng = np.random.default_rng(p['seed'])
    started = time.perf_counter()
    evaluations = 1
    current = evaluator.evaluate(np.zeros(evaluator.m, dtype=int), repair=True)
    archive = {tuple(current.genome): current.objectives.copy()} if current.feasible else {}
    history = []
    attempted = set()
    cache = OrderedDict()
    baseline = None
    status = 'completed'
    workers = ThreadPoolExecutor(p['workers']) if p['workers'] > 1 else None

    def publish(iteration, phase, previous=None, accepted=False):
        changes = tuple((int(i), int(previous[i]), int(current.genome[i]))
            for i in np.flatnonzero(previous != current.genome)) if previous is not None else ()
        elapsed = time.perf_counter() - started
        record = dict(generation=iteration, iteration=iteration, phase=phase,
            parent_iteration=iteration - 1 if iteration else None,
            parent_genome=previous.tolist() if previous is not None else None,
            genome=current.genome.tolist(), accepted=accepted,
            changes=[list(change) for change in changes], metrics=current.metrics.copy(),
            front_size=len(archive), evaluations=evaluations, seconds=elapsed)
        if baseline is not None:
            record['growth_baseline'] = baseline.copy()
        history.append(record)
        if observer:
            observer(SearchUpdate(iteration, phase, current, len(archive), evaluations,
                                  elapsed, accepted, changes, previous))
        if progress:
            progress(f'Iteration {iteration}/{p["generations"]}: {phase}; '
                f'{current.metrics["active_edges"]} edges, {current.metrics["passengers"]/1e6:.2f}M trips/year, '
                f'need satisfaction {current.metrics["need_satisfaction_ratio"]:.2%}, '
                f'{current.metrics["unserved_stations"]} unserved cities, {current.metrics["active_components"]} components, '
                f'net EUR {current.metrics["net_eur"]/1e6:.1f}M/year; '
                f'{"accepted change" if accepted else "retained network"}', flush=True)

    publish(0, 'growth')
    try:
        for iteration in range(1, p['generations'] + 1):
            if interrupted is not None and interrupted.is_set():
                status = 'interrupted'
                break
            if p['time_limit_seconds'] and time.perf_counter() - started >= p['time_limit_seconds']:
                status = 'time_limit'
                break
            complete = topology(current.genome, evaluator) == (0, 0)
            if baseline is None and complete and iteration > math.ceil(p['generations'] * p['growth_fraction']):
                baseline = current.metrics.copy()
                attempted.clear()
            phase = ('consolidation' if baseline is not None else
                     'integration' if iteration > math.ceil(p['generations'] * p['growth_fraction']) else 'growth')
            previous = current.genome.copy()
            proposals = neighbours(current, evaluator)
            pending = [(key, proposal, weight) for key, (proposal, weight) in proposals.items() if key not in attempted]
            if not pending:
                if baseline is None:
                    if not complete:
                        publish(iteration, phase, previous)
                        status = 'coverage_incomplete'
                        break
                    baseline = current.metrics.copy()
                    attempted.clear()
                    publish(iteration, 'consolidation', previous)
                    continue
                publish(iteration, phase, previous)
                status = 'neighbourhood_exhausted'
                break
            weights = np.array([value[2] for value in pending])
            weights /= weights.max()
            weights = np.maximum(weights, 1e-12)
            chosen = rng.choice(len(pending), min(p['population'], len(pending)),
                                replace=False, p=weights / weights.sum())
            batch = [pending[i] for i in chosen]
            attempted.update(key for key, _, _ in batch)
            missing = [(key, proposal) for key, proposal, _ in batch if key not in cache]
            def score(proposal):
                return evaluator.evaluate(proposal, repair=True)
            results = list(workers.map(score, [x for _, x in missing])) if workers else [score(x) for _, x in missing]
            evaluations += len(results)
            cache.update((key, result) for (key, _), result in zip(missing, results))
            candidates = [cache[key] for key, _, _ in batch if cache[key].feasible
                          and not np.array_equal(cache[key].genome, current.genome)]
            for candidate in candidates:
                archive[tuple(candidate.genome)] = candidate.objectives.copy()
            if archive:
                keys = sorted(archive)
                front = nondominated([archive[key] for key in keys])
                archive = {keys[i]: archive[keys[i]] for i in front}
            if baseline is not None:
                candidates = [candidate for candidate in candidates
                              if topology(candidate.genome, evaluator) == (0, 0)]
                selected = consolidation_choice(current, candidates, baseline, config)
            elif phase == 'integration':
                selected = integration_choice(current, candidates, evaluator)
            else:
                selected = growth_choice(current, candidates)
                if selected is None:
                    selected = integration_choice(current, candidates, evaluator)
                    if selected is not None:
                        phase = 'integration'
            if selected is not None:
                current = selected
                attempted.clear()
            publish(iteration, phase, previous, selected is not None)
            while len(cache) > max(128, 2 * p['population']):
                cache.popitem(last=False)
    finally:
        if workers:
            workers.shutdown()
    keys = sorted(archive)
    if status == 'completed' and topology(current.genome, evaluator) != (0, 0):
        status = 'coverage_incomplete'
    dimensions = 4 if p['include_time_objective'] else 3
    return SearchResult(np.array(keys, dtype=int).reshape(-1, evaluator.m),
        np.array([archive[key] for key in keys]).reshape(-1, dimensions), history,
        evaluations, time.perf_counter() - started, current.genome.copy(), status)
