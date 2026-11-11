"""Closed-form assignment, operations and annual finance; no LP/MILP or routing."""
from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path

from .alignment import TRACKS
from .demand import fare, gravity_matrix, rail_share
from .prep import distances


def capital_recovery_factor(rate, life):
    if life <= 0 or rate < 0:
        raise ValueError('Capital recovery requires nonnegative rate and positive life')
    return 1 / life if rate == 0 else rate / -math.expm1(-life * math.log1p(rate))


@dataclass
class Evaluation:
    genome: np.ndarray
    objectives: np.ndarray
    constraints: np.ndarray
    metrics: dict
    frequency: np.ndarray
    directional_load: np.ndarray
    edge_revenue: np.ndarray
    edge_net: np.ndarray
    trips: np.ndarray
    generalized_time: np.ndarray

    @property
    def feasible(self):
        return bool(np.all(self.constraints <= 1e-9))


class Evaluator:
    def __init__(self, graph, config):
        self.graph, self.config = graph, config
        self.n, self.m = len(graph.cities), len(graph.edges)
        self.a = np.array([e.a for e in graph.edges], dtype=int)
        self.b = np.array([e.b for e in graph.edges], dtype=int)
        if len({tuple(sorted((e.a, e.b))) for e in graph.edges}) != self.m:
            raise ValueError('One genome gene is required per endpoint pair')
        self.xy = np.array([c.xy for c in graph.cities])
        self.population = np.array([c.population * c.kappa for c in graph.cities])
        self.access = np.array([c.access_hours for c in graph.cities])
        self.distance = distances(graph.cities)
        base = gravity_matrix(graph.cities, config)
        self.origins, self.destinations = np.where(np.triu(base, 1) > 0)
        self.base = base[self.origins, self.destinations]
        self.od_distance = self.distance[self.origins, self.destinations]
        self.allowed = np.zeros((self.m, 3), dtype=bool)
        self.allowed[:, 0] = True
        self.lengths = np.zeros((self.m, 3))
        for i, edge in enumerate(graph.edges):
            for t, track in enumerate(TRACKS, 1):
                if track in edge.alignments:
                    self.allowed[i, t] = True
                    self.lengths[i, t] = edge.alignments[track].length_km
        self.station_capital = np.minimum(config['finance']['station_cap_meur'],
            config['finance']['station_base_meur'] + config['finance']['station_per_million_meur'] *
            np.array([c.population for c in graph.cities]) / 1e6) * 1e6
        f = config['finance']
        self.crf = {asset: capital_recovery_factor(f['discount_rate'], f[f'{asset}_life_years'])
                    for asset in ('track', 'station', 'fleet')}

    def _paths(self, active, time):
        adjacency = np.full((self.n, self.n), np.inf)
        np.fill_diagonal(adjacency, 0)
        adjacency[self.a[active], self.b[active]] = time[active]
        adjacency[self.b[active], self.a[active]] = time[active]
        duration, predecessor = shortest_path(adjacency, method='FW', directed=False, return_predecessors=True)
        ivt = duration[self.origins, self.destinations]
        connected = np.isfinite(ivt)
        lookup = np.full((self.n, self.n), -1, dtype=int)
        lookup[self.a[active], self.b[active]] = active
        lookup[self.b[active], self.a[active]] = active
        degree = np.bincount(np.r_[self.a[active], self.b[active]], minlength=self.n)
        cursor = self.destinations.copy()
        following = np.full(len(cursor), -1, dtype=int)
        transfers = np.zeros(len(cursor))
        rows, columns = [], []
        cosine_limit = math.cos(math.radians(self.config['operations']['transfer_angle_degrees']))
        for _ in range(self.n):
            pairs = np.flatnonzero(connected & (cursor != self.origins))
            if not len(pairs):
                break
            node = cursor[pairs]
            previous = predecessor[self.origins[pairs], node]
            if np.any(previous < 0):
                raise RuntimeError('Inconsistent shortest-path predecessors')
            edge_ids = lookup[previous, node]
            rows.extend(edge_ids.tolist())
            columns.extend(pairs.tolist())
            turns = (following[pairs] >= 0) & (degree[node] >= 3)
            selected = pairs[turns]
            if len(selected):
                incoming = self.xy[cursor[selected]] - self.xy[previous[turns]]
                outgoing = self.xy[following[selected]] - self.xy[cursor[selected]]
                cosine = np.einsum('ij,ij->i', incoming, outgoing) / np.maximum(1e-12,
                    np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1))
                transfers[selected] += cosine < cosine_limit
            following[pairs] = node
            cursor[pairs] = previous
        incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                                       shape=(self.m, len(self.base)))
        return incidence, ivt, transfers, degree

    def evaluate(self, genome, repair=False, details=False):
        raw = np.asarray(genome)
        if raw.shape != (self.m,) or not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
            raise ValueError('Genome must contain one integer state per candidate edge')
        x = raw.astype(int, copy=True)
        if np.any((x < 0) | (x > 2)) or not self.allowed[np.arange(self.m), x].all():
            raise ValueError('Genome selects an unavailable alignment or invalid track state')
        result = self._evaluate(x, details)
        if repair:
            unused = (x > 0) & (result.directional_load <= self.config['operations']['zero_load_threshold'])
            if unused.any():
                x[unused] = 0
                result = self._evaluate(x, details)
        return result

    def _evaluate(self, x, details):
        c, o, f = self.config, self.config['operations'], self.config['finance']
        tracks = c['tracks']
        active = np.flatnonzero(x)
        length = self.lengths[np.arange(self.m), x]
        def parameter(key):
            return np.choose(x, [0, tracks['CR'][key], tracks['HSR'][key]])
        speed = parameter('speed_kmh')
        time = np.divide(length, speed, out=np.zeros(self.m), where=speed > 0) + (x > 0) * o['dwell_hours']
        incidence, ivt, transfers, degree = self._paths(active, time)
        ridden = np.asarray(incidence.T @ length).ravel()
        weighted_yield = np.asarray(incidence.T @ (length * parameter('yield_eur_pkm'))).ravel()
        path_yield = np.divide(weighted_yield, ridden, out=np.zeros_like(ridden), where=ridden > 0)
        fares = fare(self.od_distance, path_yield, c['demand'])
        overhead_time = self.access[self.origins] + self.access[self.destinations] + transfers * o['transfer_hours'] + fares / c['demand']['value_of_time_eur']
        path_edges, path_pairs = incidence.nonzero()
        max_frequency = parameter('max_frequency')
        frequency = np.minimum(max_frequency, np.maximum(o['minimum_frequency'], o['initial_frequency'])) * (x > 0)
        hours = o['hours_per_day'] * o['days_per_year']
        seats = parameter('seats')
        def assign():
            bottleneck = np.full(len(self.base), np.inf)
            np.minimum.at(bottleneck, path_pairs, frequency[path_edges])
            wait = np.minimum(1, np.divide(.5, bottleneck, out=np.zeros_like(bottleneck), where=bottleneck > 0))
            gt = overhead_time + wait + ivt
            # Each upper-triangle market represents TWO symmetric one-way ODs.
            return 2 * self.base * rail_share(gt, c['demand']), gt
        for _ in range(o['assignment_iterations']):
            trips, gt = assign()
            load = np.asarray(incidence @ (trips / 2)).ravel()
            needed = np.ceil(o['peak_factor'] * load / hours / np.maximum(1, seats * o['target_load_factor']))
            updated = np.where(x > 0, np.clip(needed, o['minimum_frequency'], max_frequency), 0)
            if np.array_equal(updated, frequency):
                break
            frequency = updated
        trips, gt = assign()  # Always evaluate demand at the returned frequencies.
        unspilled = float(trips.sum())
        load = np.asarray(incidence @ (trips / 2)).ravel()
        capacity = seats * frequency * o['maximum_load_factor'] * hours / o['peak_factor']
        ratios = np.minimum(1, np.divide(capacity, load, out=np.ones(self.m), where=load > 0))
        spill_scale = np.ones(len(trips))
        np.minimum.at(spill_scale, path_pairs, ratios[path_edges])
        trips *= spill_scale
        load = np.asarray(incidence @ (trips / 2)).ravel()
        spilled = unspilled - float(trips.sum())
        spill_fraction = spilled / unspilled if unspilled else 0.0
        train_km = 2 * frequency * length * hours
        train_opex = train_km * parameter('opex_eur_trainkm')
        in_service = 2 * frequency * time
        fleets = {}
        fleet_capital = 0.0
        for state, track in enumerate(TRACKS, 1):
            needed = math.ceil(float(in_service[x == state].sum()) * (1 + o['turnaround_overhead']))
            fleets[track] = math.ceil(needed * (1 + o['spare_ratio']))
            fleet_capital += fleets[track] * tracks[track]['train_meur'] * 1e6
        track_capital = self.graph.track_capital(x) * 1e6
        served = degree > 0
        station_capital = float(self.station_capital[served].sum())
        annual_capital = track_capital * self.crf['track'] + station_capital * self.crf['station'] + fleet_capital * self.crf['fleet']
        operating_cost = float(train_opex.sum()) * (1 + f['overhead_rate']) + track_capital * f['track_maintenance_rate'] + station_capital * f['station_opex_rate']
        revenue = float(np.sum(trips * fares))
        total_cost = annual_capital + operating_cost
        net = revenue - total_cost
        passengers = float(trips.sum())
        eligible_needs = float(2 * self.base.sum())
        mean_gt = float(np.sum(trips[trips > 0] * gt[trips > 0]) / passengers) if passengers else 0.0
        upfront = track_capital + station_capital + fleet_capital
        if active.size:
            links = sparse.csr_matrix((np.ones(2 * len(active)),
                (np.r_[self.a[active], self.b[active]], np.r_[self.b[active], self.a[active]])),
                shape=(self.n, self.n))
            _, labels = connected_components(links, directed=False)
            active_components = len(np.unique(labels[served]))
            component_share = float(np.bincount(labels[served]).max() / served.sum())
        else:
            component_share = 0.0
            active_components = 0
        limits = c['constraints']
        constraints = np.array([
            upfront / limits['upfront_budget_eur'] - 1 if limits['upfront_budget_eur'] else 0,
            -net / limits['maximum_subsidy_eur'] - 1 if limits['maximum_subsidy_eur'] else 0,
            spill_fraction - limits['maximum_spill_fraction'],
            limits['minimum_component_share'] - component_share if active.size else 0,
        ])
        objectives = [total_cost - revenue, -passengers, -float(self.population[served].sum())]
        if c['search']['include_time_objective']:
            # Empty networks must not win the time objective by reporting zero time.
            objectives.append(mean_gt if passengers else 1e6)
        metrics = dict(net_eur=net, revenue_eur=revenue, annual_cost_eur=total_cost,
            annual_capital_eur=annual_capital, operating_cost_eur=operating_cost,
            train_opex_eur=float(train_opex.sum()), upfront_eur=upfront,
            track_capital_eur=track_capital, station_capital_eur=station_capital,
            fleet_capital_eur=fleet_capital, passengers=passengers,
            eligible_need_trips=eligible_needs,
            need_satisfaction_ratio=passengers / eligible_needs if eligible_needs else 0.0,
            population_served=float(self.population[served].sum()), mean_generalized_hours=mean_gt,
            spill_fraction=spill_fraction, lost_trips=spilled, component_share=component_share,
            fleet_CR=fleets['CR'], fleet_HSR=fleets['HSR'],
            track_km=float(length.sum()), active_edges=int(len(active)),
            unserved_stations=int(self.n - served.sum()), active_components=int(active_components),
            active_stations=int(served.sum()), train_km_per_year=float(train_km.sum()))
        edge_revenue = np.zeros(self.m)
        edge_net = np.zeros(self.m)
        if details:
            # Allocate fare revenue by ridden distance, without earning extra fare on detours.
            per_km = np.divide(trips * fares, ridden, out=np.zeros_like(trips), where=ridden > 0)
            edge_revenue = np.asarray(incidence @ per_km).ravel() * length
            edge_track = np.zeros(self.m)
            hsr_used = np.asarray(self.graph.matrices[1].T @ (x == 2)).ravel() > 0
            for state, matrix in enumerate(self.graph.matrices, 1):
                count = np.asarray(matrix.T @ (x == state)).ravel()
                selected = (count > 0) & (hsr_used if state == 2 else ~hsr_used)
                split = np.zeros_like(count)
                split[selected] = self.graph.tile_costs[state - 1, selected] * 1e6 / count[selected]
                edge_track += np.asarray(matrix @ split).ravel() * (x == state)
            station_split = np.divide(self.station_capital, degree, out=np.zeros(self.n), where=degree > 0)
            edge_station = (station_split[self.a] + station_split[self.b]) * (x > 0)
            edge_fleet = np.zeros(self.m)
            for state, track in enumerate(TRACKS, 1):
                mask = x == state
                total = in_service[mask].sum()
                if total:
                    edge_fleet[mask] = in_service[mask] / total * fleets[track] * tracks[track]['train_meur'] * 1e6
            edge_net = edge_revenue - train_opex * (1 + f['overhead_rate']) - edge_track * (self.crf['track'] + f['track_maintenance_rate']) - edge_station * (self.crf['station'] + f['station_opex_rate']) - edge_fleet * self.crf['fleet']
        return Evaluation(x, np.asarray(objectives), constraints, metrics, frequency, load,
                          edge_revenue, edge_net, trips, gt)
