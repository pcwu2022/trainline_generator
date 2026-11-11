"""Integer NSGA-II/III, structured network moves, repair and Pareto archiving."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path

from .evaluator import Evaluator


@dataclass
class SearchResult:
    genomes: np.ndarray
    objectives: np.ndarray
    history: list
    evaluations: int
    elapsed_seconds: float


def nondominated(objectives):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    return NonDominatedSorting().do(np.asarray(objectives), only_non_dominated_front=True)


def greedy_seeds(evaluator, count):
    """Marginal net value / incremental capital, sampled at several growth budgets."""
    m = evaluator.m
    empty = np.zeros(m, dtype=int)
    if count <= 0:
        return []
    standalone = []
    for edge in range(m):
        x = empty.copy()
        x[edge] = 1 if evaluator.allowed[edge, 1] else 2
        score = evaluator.evaluate(x, repair=True)
        standalone.append((score.metrics['passengers'], edge, x[edge]))
    ranked = sorted(standalone, reverse=True)
    pool = [edge for _, edge, state in ranked if state > 0]
    if not pool:
        return [empty.copy() for _ in range(count)]
    trajectory = [empty]
    # Also preserve attractive isolated corridors as initial population members.
    for _, edge, state in ranked[:min(5, count)]:
        x = empty.copy()
        x[edge] = state
        trajectory.append(x)
    x = trajectory[1].copy()
    for _ in range(min(m, max(12, count))):
        current = evaluator.evaluate(x)
        candidates = []
        for edge in [e for e in pool if x[e] == 0][:12]:
            for state in (1, 2):
                if evaluator.allowed[edge, state]:
                    proposal = x.copy()
                    proposal[edge] = state
                    score = evaluator.evaluate(proposal, repair=True)
                    delta_capital = max(1, score.metrics['upfront_eur'] - current.metrics['upfront_eur'])
                    benefit = (score.metrics['net_eur'] - current.metrics['net_eur']) / delta_capital
                    candidates.append((benefit, score.metrics['passengers'], score.genome))
        if not candidates:
            break
        x = max(candidates, key=lambda value: value[:2])[2].copy()
        if np.array_equal(x, trajectory[-1]):
            break
        trajectory.append(x)
    indices = np.linspace(0, len(trajectory) - 1, count).round().astype(int)
    return [trajectory[i].copy() for i in indices]


def bridge_components(x, evaluator, rng):
    active = np.flatnonzero(x)
    if not len(active):
        return x
    links = sparse.csr_matrix((np.ones(2 * len(active)),
        (np.r_[evaluator.a[active], evaluator.b[active]], np.r_[evaluator.b[active], evaluator.a[active]])),
        shape=(evaluator.n, evaluator.n))
    _, labels = connected_components(links, directed=False)
    served = np.flatnonzero(np.asarray(links.sum(axis=1)).ravel() > 0)
    choices = [(a, b) for a in served for b in served if a < b and labels[a] != labels[b]]
    if not choices:
        return x
    costs = np.full((evaluator.n, evaluator.n), np.inf)
    np.fill_diagonal(costs, 0)
    lookup = {}
    for i, edge in enumerate(evaluator.graph.edges):
        cost = .000001 if x[i] else min(a.capex_meur for a in edge.alignments.values())
        costs[edge.a, edge.b] = costs[edge.b, edge.a] = cost
        lookup[edge.a, edge.b] = lookup[edge.b, edge.a] = i
    distances, previous = shortest_path(costs, method='FW', directed=False, return_predecessors=True)
    a, b = min(choices, key=lambda ab: distances[ab])
    if not np.isfinite(distances[a, b]):
        return x
    while b != a:
        parent = previous[a, b]
        edge = lookup[parent, b]
        if not x[edge]:
            alignments = evaluator.graph.edges[edge].alignments
            track = min(alignments, key=lambda t: alignments[t].capex_meur)
            x[edge] = 1 if track == 'CR' else 2
        b = parent
    return x


def structured_mutation(x, evaluator, rng):
    action = int(rng.integers(4))
    active = np.flatnonzero(x)
    if action == 0:
        return bridge_components(x, evaluator, rng)
    if not len(active):
        return x
    score = evaluator.evaluate(x, details=True)
    if action == 1:
        # Full annual marginal allocation includes shared infrastructure and stations.
        x[active[np.argmin(score.edge_net[active])]] = 0
    elif action == 2:
        choices = np.flatnonzero((x == 1) & evaluator.allowed[:, 2])
        if len(choices):
            x[choices[np.argmax(score.directional_load[choices])]] = 2
    else:
        choices = np.flatnonzero((x == 2) & evaluator.allowed[:, 1])
        if len(choices):
            x[choices[np.argmin(score.directional_load[choices])]] = 1
    return x


def search(graph, config, progress=print):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.callback import Callback
    from pymoo.core.mutation import Mutation
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.indicators.hv import HV
    from pymoo.operators.crossover.ux import UniformCrossover
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    p = config['search']
    evaluator = Evaluator(graph, config)
    rng = np.random.default_rng(p['seed'])
    started = time.perf_counter()
    archive = {}
    history = []
    workers = ThreadPoolExecutor(p['workers']) if p['workers'] > 1 else None

    class NetworkProblem(Problem):
        def __init__(self):
            super().__init__(n_var=evaluator.m, n_obj=4 if p['include_time_objective'] else 3,
                             n_ieq_constr=4, xl=0, xu=2, vtype=int)
            self.evaluations = 0

        def _evaluate(self, X, out, *args, **kwargs):
            def score(x):
                return evaluator.evaluate(x, repair=True)
            results = list(workers.map(score, X)) if workers else [score(x) for x in X]
            self.evaluations += len(results)
            out['F'] = np.array([r.objectives for r in results])
            out['G'] = np.array([r.constraints for r in results])
            # pymoo's evaluator writes returned X back to each individual.
            out['X'] = np.array([r.genome for r in results], dtype=int)
            for result in results:
                if result.feasible:
                    archive[tuple(result.genome)] = result.objectives
            if archive:
                keys = list(archive)
                front = nondominated([archive[key] for key in keys])
                selected = {keys[i]: archive[keys[i]] for i in front}
                archive.clear()
                archive.update(selected)

    class AvailabilityRepair(Repair):
        def _do(self, problem, X, **kwargs):
            X = np.clip(np.rint(X), 0, 2).astype(int)
            for row in X:
                invalid = ~evaluator.allowed[np.arange(evaluator.m), row]
                row[invalid] = 0
            return X

    class EdgeMutation(Mutation):
        def __init__(self):
            super().__init__(prob=1.0)

        def _do(self, problem, X, **kwargs):
            X = X.astype(int, copy=True)
            rate = p['mutation_rate'] or 1 / evaluator.m
            for row in X:
                selected = np.flatnonzero(rng.random(evaluator.m) < rate)
                for gene in selected:
                    states = np.flatnonzero(evaluator.allowed[gene])
                    choices = states[states != row[gene]]
                    if len(choices):
                        row[gene] = rng.choice(choices)
                if rng.random() < p['structured_probability']:
                    structured_mutation(row, evaluator, rng)
            return X

    class FrontProgress(Callback):
        def __init__(self):
            super().__init__()
            self.ideal = self.scale = self.reference = None
            self.best_hv = 0.0
            self.stagnant = 0

        def notify(self, algorithm):
            hv = 0.0
            if archive:
                F = np.array(list(archive.values()))
                if self.ideal is None:
                    self.ideal = F.min(axis=0)
                    self.scale = np.maximum(F.max(axis=0) - self.ideal, 1)
                    self.reference = np.full(F.shape[1], 1.1)
                hv = float(HV(ref_point=self.reference)((F - self.ideal) / self.scale))
                improvement = (hv - self.best_hv) / max(1, abs(self.best_hv))
                if improvement > p['hypervolume_tolerance']:
                    self.best_hv = hv
                    self.stagnant = 0
                else:
                    self.stagnant += 1
                if p['hypervolume_tolerance'] and self.stagnant >= p['stagnation_generations']:
                    algorithm.termination.terminate()
            history.append(dict(generation=int(algorithm.n_gen), front_size=len(archive),
                evaluations=problem.evaluations, hypervolume=hv, seconds=time.perf_counter() - started))
            if progress:
                progress(f'Generation {algorithm.n_gen}/{p["generations"]}: {len(archive)} Pareto designs, {problem.evaluations} evaluations', flush=True)

    initial = greedy_seeds(evaluator, int(p['population'] * p['greedy_fraction']))
    # Include the empty design and demand-ranked sparse subsets reproducibly.
    initial.insert(0, np.zeros(evaluator.m, dtype=int))
    gravity = gravity_scores(evaluator)
    while len(initial) < p['population']:
        x = np.zeros(evaluator.m, dtype=int)
        probability = rng.uniform(.02, .10)
        bias = gravity / max(1e-12, gravity.max()) if len(initial) % 2 else np.ones(evaluator.m)
        selected = np.flatnonzero(rng.random(evaluator.m) < probability * bias)
        for gene in selected:
            x[gene] = rng.choice(np.flatnonzero(evaluator.allowed[gene, 1:]) + 1)
        initial.append(x)
    problem = NetworkProblem()
    options = dict(pop_size=p['population'], sampling=np.array(initial[:p['population']]),
                   crossover=UniformCrossover(), mutation=EdgeMutation(),
                   repair=AvailabilityRepair(), eliminate_duplicates=True)
    if p['algorithm'] == 'nsga2':
        algorithm = NSGA2(**options)
    elif p['algorithm'] == 'nsga3':
        directions = get_reference_directions('energy', problem.n_obj, p['population'], seed=p['seed'])
        algorithm = NSGA3(ref_dirs=directions, **options)
    else:
        raise ValueError('Supported algorithms are nsga2 and nsga3 (pymoo MOEA/D lacks constraint handling)')
    try:
        minimize(problem, algorithm, ('n_gen', p['generations']), seed=p['seed'],
                 callback=FrontProgress(), verbose=False, copy_algorithm=False)
    finally:
        if workers:
            workers.shutdown()
    if not archive:
        return SearchResult(np.empty((0, evaluator.m), dtype=int),
            np.empty((0, problem.n_obj)), history, problem.evaluations, time.perf_counter() - started)
    keys = sorted(archive)
    return SearchResult(np.array(keys, dtype=int), np.array([archive[key] for key in keys]),
                        history, problem.evaluations, time.perf_counter() - started)


def gravity_scores(evaluator):
    from .demand import gravity_matrix
    base = gravity_matrix(evaluator.graph.cities, evaluator.config)
    return np.array([base[e.a, e.b] for e in evaluator.graph.edges])


def representative_indices(objectives, limit, subsidy):
    """K-medoids in normalized objective space, preserving financial/policy anchors."""
    from scipy.spatial.distance import cdist
    F = np.asarray(objectives)
    if not len(F):
        return []
    limit = min(len(F), max(3, limit))
    normalized = (F - F.min(axis=0)) / np.maximum(1, np.ptp(F, axis=0))
    best_net = int(np.argmin(F[:, 0]))
    eligible = np.flatnonzero(F[:, 0] <= subsidy)
    best_passengers = int(eligible[np.argmin(F[eligible, 1])]) if len(eligible) else best_net
    knee = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    anchors = list(dict.fromkeys([best_net, best_passengers, knee]))
    medoids = anchors.copy()
    distance = cdist(normalized, normalized)
    while len(medoids) < limit:
        nearest = distance[:, medoids].min(axis=1)
        nearest[medoids] = -1
        medoids.append(int(np.argmax(nearest)))
    for _ in range(10):
        assigned = np.argmin(distance[:, medoids], axis=1)
        previous = medoids.copy()
        for cluster in range(len(anchors), len(medoids)):
            members = np.flatnonzero(assigned == cluster)
            if len(members):
                medoids[cluster] = int(members[np.argmin(distance[np.ix_(members, members)].sum(axis=1))])
        if medoids == previous:
            break
    return medoids
