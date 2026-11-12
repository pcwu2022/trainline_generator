"""Service-based assignment and annual finance on a newly built railway network."""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .alignment import TRACKS
from .track_layout import LAYOUT_BY_COUNT, MODEL_VERSION
from .demand import gravity_matrix
from .prep import distances
from .services import Service, ServiceNetwork, generate_plans


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
    frequency: np.ndarray  # Aggregate trains/hour/direction on infrastructure edges.
    directional_load: np.ndarray
    edge_revenue: np.ndarray
    edge_net: np.ndarray
    trips: np.ndarray
    generalized_time: np.ndarray
    train_units: np.ndarray  # Maximum formation using an edge, not an edge decision.
    capital_multiplier: np.ndarray
    track_counts: np.ndarray
    service_plan: tuple
    services: list
    edge_incidence: object
    transfers: np.ndarray
    waiting_hours: np.ndarray
    in_vehicle_hours: np.ndarray
    edge_capacity: np.ndarray
    segment_load: np.ndarray
    segment_capacity: np.ndarray

    @property
    def feasible(self):
        return bool(np.all(self.constraints <= 1e-9))


@dataclass
class Infrastructure:
    """Candidate-local resources shared by all service trials; never globally cached."""
    selected: object
    edge_caps: np.ndarray
    tile_caps: np.ndarray
    edge_track: np.ndarray
    active: np.ndarray
    length: np.ndarray
    degree: np.ndarray
    served: np.ndarray
    active_components: int
    component_share: float
    multipliers: np.ndarray


class Evaluator:
    def __init__(self, graph, config):
        self.graph, self.config = graph, config
        self.n, self.m = len(graph.cities), len(graph.edges)
        self.a = np.array([e.a for e in graph.edges], dtype=int)
        self.b = np.array([e.b for e in graph.edges], dtype=int)
        if len({tuple(sorted((e.a, e.b))) for e in graph.edges}) != self.m:
            raise ValueError('One infrastructure gene is required per endpoint pair')
        self.population = np.array([c.population * c.kappa for c in graph.cities])
        self.access = np.array([c.access_hours for c in graph.cities])
        self.distance = distances(graph.cities)
        base = gravity_matrix(graph.cities, config)
        self.origins, self.destinations = np.where(np.triu(base, 1) > 0)
        self.base = base[self.origins, self.destinations]
        self.sources = np.unique(self.origins)
        self.source_row = np.searchsorted(self.sources, self.origins)
        self.ranked_od = sorted(range(len(self.base)),
            key=lambda j: (-self.base[j], int(self.origins[j]), int(self.destinations[j])))
        self.gravity_scores = np.array([base[e.a, e.b] for e in graph.edges])
        self.edge_lookup = {tuple(sorted((e.a, e.b))): i for i, e in enumerate(graph.edges)}
        self.od_distance = self.distance[self.origins, self.destinations]
        self.allowed = np.zeros((self.m, 3), dtype=bool)
        self.allowed[:, 0] = True
        self.lengths = np.zeros((self.m, 3))
        for i, edge in enumerate(graph.edges):
            for state, track in enumerate(TRACKS, 1):
                if track in edge.alignments and not (config['infrastructure_layout'] == 'single' and state == 2):
                    self.allowed[i, state] = True
                if track in edge.alignments:
                    self.lengths[i, state] = edge.alignments[track].length_km
        self.station_capital = np.minimum(config['finance']['station_cap_meur'],
            config['finance']['station_base_meur'] + config['finance']['station_per_million_meur'] *
            np.array([c.population for c in graph.cities]) / 1e6) * 1e6
        f = config['finance']
        self.crf = {asset: capital_recovery_factor(f['discount_rate'], f[f'{asset}_life_years'])
                    for asset in ('track', 'station', 'fleet')}

    def selected_tiles(self, x):
        return self.graph.selected_tiles(x)

    def edge_capacities(self, x, counts):
        capacities = np.zeros(self.m)
        for i in np.flatnonzero(x):
            track = TRACKS[x[i] - 1]
            layout = LAYOUT_BY_COUNT[int(counts[i])]
            limit = self.config['track_layouts'][layout]['max_frequency'][track]
            # Old max_frequency remains an infrastructure limit, never a train limit.
            ratio = limit / self.config['track_layouts']['double']['max_frequency'][track]
            capacities[i] = min(limit, self.config['tracks'][track]['max_frequency'] * ratio)
        return capacities

    def tile_capacities(self, x, counts):
        matrix = self.selected_tiles(x).tocoo()
        caps = np.zeros(self.graph.tile_costs.shape[1])
        edge_caps = self.edge_capacities(x, counts)
        np.maximum.at(caps, matrix.col, edge_caps[matrix.row])
        return caps

    def infrastructure(self, x, counts):
        selected = self.selected_tiles(x)
        edge_caps = self.edge_capacities(x, counts)
        matrix = selected.tocoo()
        tile_caps = np.zeros(self.graph.tile_costs.shape[1])
        np.maximum.at(tile_caps, matrix.col, edge_caps[matrix.row])
        edge_track = self.graph.track_allocation(x, self.config['track_layouts'], counts,
                                                selected=selected) * 1e6
        active = np.flatnonzero(x)
        degree = np.bincount(np.r_[self.a[active], self.b[active]], minlength=self.n)
        served = degree > 0
        links = sparse.csr_matrix((np.ones(2 * len(active)),
            (np.r_[self.a[active], self.b[active]], np.r_[self.b[active], self.a[active]])),
            shape=(self.n, self.n))
        _, labels = connected_components(links, directed=False)
        components = len(np.unique(labels[served])) if active.size else 0
        share = float(np.bincount(labels[served]).max() / served.sum()) if active.size else 0.0
        multipliers = np.array([self.config['track_layouts'][LAYOUT_BY_COUNT[int(count)]]['cost_multiplier']
                                if count else 0 for count in counts])
        return Infrastructure(selected, edge_caps, tile_caps, edge_track, active,
            self.lengths[np.arange(self.m), x], degree, served, components, share, multipliers)

    def evaluate(self, genome, repair=False, details=False, services=None, track_counts=None):
        raw = np.asarray(genome)
        if raw.shape != (self.m,) or not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
            raise ValueError('Genome must contain one integer infrastructure state per candidate edge')
        x = raw.astype(int, copy=True)
        if np.any((x < 0) | (x > 2)):
            raise ValueError('Infrastructure states are 0/1/2; use migrate_legacy_genome for old layout states 3/4/5')
        # Explicit counts may override the global layout, but not alignment availability.
        for i in np.flatnonzero(x):
            if TRACKS[x[i] - 1] not in self.graph.edges[i].alignments:
                raise ValueError('Genome selects an unavailable alignment')
        default_count = {'single': 1, 'double': 2, 'four': 4}[self.config['infrastructure_layout']]
        counts = np.where(x > 0, default_count, 0) if track_counts is None else np.asarray(track_counts)
        if counts.shape != x.shape or not np.isfinite(counts).all() or np.any((x > 0) & ~np.isin(counts, [1, 2, 4])) or np.any((x == 0) & (counts != 0)):
            raise ValueError('Track counts must be 1/2/4 on built edges and zero on unbuilt edges')
        counts = counts.astype(int, copy=True)
        if np.any((counts == 1) & (x == 2)):
            raise ValueError('HSR infrastructure is prohibited on single track')
        infrastructure = self.infrastructure(x, counts)
        if services is not None:
            plan = tuple(Service.from_dict(s) if isinstance(s, dict) else s for s in services)
            return self._evaluate_plan(x, counts, plan, optimize=False, details=details, infrastructure=infrastructure)
        networks = generate_plans(self, x, counts, infrastructure=infrastructure, return_networks=True)
        results = (self._evaluate_plan(x, counts, network.services, optimize=True, details=details,
                   infrastructure=infrastructure, network=network) for network in networks)
        result = min(results, key=lambda r: (float(np.maximum(r.constraints - 1e-9, 0).sum()), r.objectives[0], -r.metrics['passengers']))
        if repair:
            unused = (x > 0) & (result.directional_load <= self.config['operations']['zero_load_threshold'])
            if unused.any():
                x[unused], counts[unused] = 0, 0
                # Regenerate the whole service plan when infrastructure changes.
                return self.evaluate(x, repair=False, details=details, track_counts=counts)
        return result

    def _evaluate_plan(self, x, counts, plan, optimize, details, infrastructure=None, network=None):
        c, o, f = self.config, self.config['operations'], self.config['finance']
        infrastructure = infrastructure if infrastructure is not None else self.infrastructure(x, counts)
        network = network if network is not None else ServiceNetwork(self, x, plan, counts, infrastructure)
        assignment = None
        ns = len(plan)
        hours = o['hours_per_day'] * o['days_per_year']
        frequency = np.array([s.frequency for s in plan], dtype=float)
        units = np.array([s.formation for s in plan], dtype=int)
        stock = [c['rolling_stock'][s.train_type] for s in plan]
        seats = np.array([s['seats'] for s in stock])
        if optimize and ns:
            frequency = network.fit_frequencies(np.full(ns, o['initial_frequency']))
            max_cap = max(c['tracks'][t]['max_frequency'] * 2 for t in TRACKS)
            levels = np.array(sorted(set(o['low_frequencies'] + [o['minimum_frequency']] + list(range(1, math.ceil(max_cap) + 1)))))
            if c['services']['optimize_frequencies']:
                for _ in range(o['assignment_iterations']):
                    assignment = network.assign(frequency)
                    load = np.asarray(assignment['incidence'] @ (assignment['trips'] / 2)).ravel()
                    maximum = np.zeros(ns)
                    np.maximum.at(maximum, network.segment_service, load)
                    required = o['peak_factor'] * maximum / hours / (seats * o['target_load_factor'])
                    requested = levels[np.minimum(np.searchsorted(levels, required), len(levels) - 1)]
                    updated = network.fit_frequencies(requested)
                    if np.array_equal(updated, frequency):
                        break
                    frequency = updated
                    assignment = None
        if assignment is None:
            assignment = network.assign(frequency)
        incidence = assignment['incidence']
        trips = assignment['trips'].copy()
        unspilled = float(trips.sum())
        raw_load = np.asarray(incidence @ (trips / 2)).ravel()
        unit_capacity = seats * frequency * o['maximum_load_factor'] * hours / o['peak_factor']
        if optimize and ns:
            maximum = np.zeros(ns)
            np.maximum.at(maximum, network.segment_service, raw_load)
            units = np.clip(np.ceil(maximum / unit_capacity), 1, o['maximum_train_units']).astype(int)
        service_capacity = unit_capacity * units
        segment_capacity = service_capacity[network.segment_service]
        ratios = np.minimum(1, np.divide(segment_capacity, raw_load, out=np.ones_like(raw_load), where=raw_load > 0))
        scale = np.ones(len(trips))
        rows, pairs = incidence.nonzero()
        np.minimum.at(scale, pairs, ratios[rows])
        trips *= scale
        segment_load = np.asarray(incidence @ (trips / 2)).ravel()
        edge_incidence = assignment['edge_incidence']
        edge_load = np.asarray(edge_incidence @ (trips / 2)).ravel()
        edge_frequency = np.asarray(network.edge_services @ frequency).ravel()
        edge_capacity = np.asarray(network.edge_services @ service_capacity).ravel()
        movements = np.asarray(network.resource_services @ frequency).ravel()
        track_overload = float(np.max(np.divide(movements, network.resource_caps,
            out=np.zeros_like(movements), where=network.resource_caps > 0) - 1, initial=0))
        fleets = dict(CR=0, HSR=0)
        edge_opex, edge_fleet = np.zeros(self.m), np.zeros(self.m)
        services, final_plan = [], []
        train_km_total, unit_km_total = 0.0, 0.0
        max_units = np.zeros(self.m, dtype=int)
        fleet_capital = 0.0
        for sid, service in enumerate(plan):
            selected = np.flatnonzero(network.segment_service == sid)
            edges = network.segment_edges[selected]
            section_times = network.times[selected]
            one_way = float(section_times.sum())
            cycle = 2 * one_way * (1 + o['turnaround_overhead'])
            active_formations = math.ceil(cycle * frequency[sid])
            base_fleet = active_formations * int(units[sid])
            fleet = math.ceil(base_fleet * (1 + o['spare_ratio']))
            fleets[service.train_type] += fleet
            capital = fleet * stock[sid]['train_meur'] * 1e6
            fleet_capital += capital
            section_km = 2 * frequency[sid] * network.lengths[selected] * hours
            # The first set incurs the full operating rate; additional coupled
            # sets share some costs (e.g. driver), but add energy and maintenance.
            operating_multiplier = 1 + o['additional_unit_cost_fraction'] * (units[sid] - 1)
            section_opex = section_km * operating_multiplier * stock[sid]['opex_eur_trainkm']
            np.add.at(edge_opex, edges, section_opex)
            np.add.at(edge_fleet, edges, capital * section_times / one_way)
            np.maximum.at(max_units, edges, units[sid])
            train_km_total += float(section_km.sum())
            unit_km_total += float(section_km.sum() * units[sid])
            final = replace(service, frequency=float(frequency[sid]), formation=int(units[sid]))
            final_plan.append(final)
            services.append(dict(service_id=sid, **final.to_dict(), edge_ids=edges.tolist(),
                section_speeds_kmh=network.speeds[selected].tolist(), one_way_hours=one_way,
                cycle_hours=cycle, active_formations=active_formations, active_train_units=base_fleet, fleet_units=fleet, fleet_capital_eur=capital,
                operating_cost_eur=float(section_opex.sum()),
                directional_annual_load=segment_load[selected].tolist(),
                directional_annual_capacity=float(service_capacity[sid])))
        active, length = infrastructure.active, infrastructure.length
        degree, served = infrastructure.degree, infrastructure.served
        edge_track = infrastructure.edge_track
        track_capital = float(edge_track.sum())
        station_capital = float(self.station_capital[served].sum())
        annual_capital = track_capital * self.crf['track'] + station_capital * self.crf['station'] + fleet_capital * self.crf['fleet']
        train_opex = float(edge_opex.sum())
        operating_cost = train_opex * (1 + f['overhead_rate']) + track_capital * f['track_maintenance_rate'] + station_capital * f['station_opex_rate']
        revenue = float(np.sum(trips * assignment['fares']))
        total_cost = annual_capital + operating_cost
        net = revenue - total_cost
        passengers = float(trips.sum())
        eligible_needs = float(2 * self.base.sum())
        gt = assignment['generalized_time']
        mean_gt = float(np.sum(trips[trips > 0] * gt[trips > 0]) / passengers) if passengers else 0.0
        # Normalize before multiplying so the weighted mean cannot overflow
        # merely because annual passenger counts are large. Use an explicit
        # reduction rather than dispatching this scalar statistic to BLAS.
        mean_transfers = float(np.sum((trips / passengers) * assignment['transfers'])) if passengers else 0.0
        upfront = track_capital + station_capital + fleet_capital
        active_components = infrastructure.active_components
        component_share = infrastructure.component_share
        spilled = unspilled - passengers
        spill_fraction = spilled / unspilled if unspilled else 0.0
        limits = c['constraints']
        constraints = np.array([
            upfront / limits['upfront_budget_eur'] - 1 if limits['upfront_budget_eur'] else 0,
            -net / limits['maximum_subsidy_eur'] - 1 if limits['maximum_subsidy_eur'] else 0,
            spill_fraction - limits['maximum_spill_fraction'],
            limits['minimum_component_share'] - component_share if active.size else 0,
            track_overload,
        ])
        objectives = [total_cost - revenue, -passengers, -float(self.population[served].sum())]
        if c['search']['include_time_objective']:
            objectives.append(mean_gt if passengers else 1e6)
        metrics = dict(model_version=MODEL_VERSION, net_eur=net, revenue_eur=revenue, annual_cost_eur=total_cost,
            annual_capital_eur=annual_capital, operating_cost_eur=operating_cost,
            train_opex_eur=train_opex, upfront_eur=upfront, track_capital_eur=track_capital,
            station_capital_eur=station_capital, fleet_capital_eur=fleet_capital, passengers=passengers,
            eligible_need_trips=eligible_needs, need_satisfaction_ratio=passengers / eligible_needs if eligible_needs else 0.0,
            population_served=float(self.population[served].sum()), mean_generalized_hours=mean_gt,
            spill_fraction=spill_fraction, lost_trips=spilled, component_share=component_share,
            fleet_CR=fleets['CR'], fleet_HSR=fleets['HSR'], track_km=float(length.sum()),
            active_edges=int(len(active)), unserved_stations=int(self.n - served.sum()),
            active_components=int(active_components), active_stations=int(served.sum()),
            train_km_per_year=train_km_total, train_unit_km_per_year=unit_km_total,
            multi_unit_edges=int(np.count_nonzero(max_units > 1)),
            maximum_train_units_used=int(max_units.max()) if self.m else 0,
            service_count=ns, shared_track_overload=track_overload,
            mean_transfers=mean_transfers)
        per_km = np.divide(trips * assignment['fares'], assignment['ridden'], out=np.zeros_like(trips), where=assignment['ridden'] > 0)
        edge_revenue = np.asarray(edge_incidence @ per_km).ravel() * length
        station_split = np.divide(self.station_capital, degree, out=np.zeros(self.n), where=degree > 0)
        edge_station = (station_split[self.a] + station_split[self.b]) * (x > 0)
        edge_net = edge_revenue - edge_opex * (1 + f['overhead_rate']) - edge_track * (self.crf['track'] + f['track_maintenance_rate']) - edge_station * (self.crf['station'] + f['station_opex_rate']) - edge_fleet * self.crf['fleet']
        multipliers = infrastructure.multipliers.copy()
        return Evaluation(x.copy(), np.asarray(objectives), constraints, metrics, edge_frequency, edge_load,
            edge_revenue, edge_net, trips, gt, max_units, multipliers, counts.copy(), tuple(final_plan), services,
            edge_incidence, assignment['transfers'], assignment['wait'], assignment['ivt'], edge_capacity,
            segment_load, segment_capacity)
