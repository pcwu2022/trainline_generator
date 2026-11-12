"""Deterministic bounded service plans and frequency-based passenger journeys.

Every service is symmetric in both directions and stops at every route node.
The expanded graph charges a wait on boarding, never on an infrastructure change.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class Service:
    route: tuple
    train_type: str
    frequency: float
    formation: int = 1

    def __post_init__(self):
        object.__setattr__(self, "route", tuple(self.route))

    def to_dict(self):
        return dict(route=[int(node) for node in self.route], train_type=self.train_type,
                    frequency=float(self.frequency), formation=int(self.formation))

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value['route']), value['train_type'], value['frequency'], value['formation'])


class ServiceNetwork:
    def __init__(self, evaluator, x, services, track_counts, infrastructure=None):
        self.e = e = evaluator
        self.x = x
        self.services = tuple(services)
        self.edges, self.segment_edges, self.segment_service = [], [], []
        self.speeds, self.times, self.yields, self.offsets = [], [], [], []
        self.lookup = e.edge_lookup
        for sid, service in enumerate(services):
            if not isinstance(service, Service):
                raise ValueError('Service plans must contain Service objects or decision dictionaries')
            route = service.route
            if (service.train_type not in e.config['rolling_stock'] or len(route) < 2 or
                any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v < e.n for v in route) or
                len(set(route)) != len(route)):
                raise ValueError('Services need a simple continuous route and a known train type')
            if (isinstance(service.frequency, (bool, np.bool_)) or not isinstance(service.frequency, (int, float, np.integer, np.floating)) or not np.isfinite(service.frequency) or
                service.frequency < e.config['operations']['minimum_frequency'] or
                isinstance(service.formation, bool) or not isinstance(service.formation, (int, np.integer)) or
                not 1 <= service.formation <= e.config['operations']['maximum_train_units']):
                raise ValueError('Invalid service frequency or formation')
            edges = []
            self.offsets.append(len(self.segment_edges))
            stock = e.config['rolling_stock'][service.train_type]
            for a, b in zip(route, route[1:]):
                edge = self.lookup.get(tuple(sorted((a, b))))
                if edge is None or not x[edge]:
                    raise ValueError('Service route is disconnected or uses unbuilt infrastructure')
                # The single-track policy prohibits HSR trains, even at reduced speed.
                if track_counts[edge] == 1 and service.train_type == 'HSR':
                    raise ValueError('HSR services are prohibited on single track')
                edges.append(edge)
                speed = min(e.config['tracks']['CR' if x[edge] == 1 else 'HSR']['speed_kmh'], stock['speed_kmh'])
                self.speeds.append(speed)
                self.times.append(e.lengths[edge, x[edge]] / speed + e.config['operations']['dwell_hours'])
                self.yields.append(stock['yield_eur_pkm'])
                self.segment_edges.append(edge)
                self.segment_service.append(sid)
            self.edges.append(tuple(edges))
        self.segment_edges = np.asarray(self.segment_edges, dtype=int)
        self.segment_service = np.asarray(self.segment_service, dtype=int)
        self.times = np.asarray(self.times)
        self.speeds = np.asarray(self.speeds)
        self.yields = np.asarray(self.yields)
        self.lengths = e.lengths[self.segment_edges, x[self.segment_edges]]
        ns = len(services)
        self.edge_services = sparse.csr_matrix((np.ones(len(self.segment_edges)),
            (self.segment_edges, self.segment_service)), shape=(e.m, ns))
        # One traversal per physical tile per service, including where adjacent
        # abstract edges overlap the same tile. This avoids duplicate movements.
        infrastructure = infrastructure if infrastructure is not None else e.infrastructure(x, track_counts)
        selected = infrastructure.selected
        tile_services = selected.T @ self.edge_services
        tile_services.data[:] = 1
        tile_caps = infrastructure.tile_caps
        self.resource_services = sparse.vstack([self.edge_services, tile_services], format='csr')
        self.resource_caps = np.r_[infrastructure.edge_caps, tile_caps]
        self._resource_coo = self.resource_services.tocoo()
        # This map has one column per segment, independent of service frequency.
        self.segment_to_edge = sparse.csr_matrix((np.ones(len(self.segment_edges)),
            (self.segment_edges, np.arange(len(self.segment_edges)))),
            shape=(e.m, len(self.segment_edges)))
        self._adjacency = None

    def fit_frequencies(self, requested):
        """Keep existing minimum allocations, proportionally ration extra paths."""
        requested = np.asarray(requested, dtype=float)
        if not len(requested):
            return requested
        minimum = self.e.config['operations']['minimum_frequency']
        floor = np.full(len(requested), minimum)
        extra = np.maximum(0, requested - floor)
        remaining = np.maximum(0, self.resource_caps - np.asarray(self.resource_services @ floor).ravel())
        usage = np.asarray(self.resource_services @ extra).ravel()
        ratios = np.minimum(1, np.divide(remaining, usage, out=np.ones_like(usage), where=usage > 0))
        scale = np.ones(len(requested))
        coo = self._resource_coo
        np.minimum.at(scale, coo.col, ratios[coo.row])
        return floor + extra * scale

    def _prepare_assignment(self):
        """Build fixed arcs once, preserving explicit zero-cost alighting arcs."""
        e = self.e
        rows, cols, weights, arc_segment = [], [], [], []
        size = e.n + sum(len(s.route) for s in self.services)
        cursor = e.n
        transfer = e.config['operations']['transfer_hours']
        for sid, service in enumerate(self.services):
            for pos, city in enumerate(service.route):
                node = cursor + pos
                rows.extend((city, node))
                cols.extend((node, city))
                weights.extend((transfer, 0))
                arc_segment.extend((-1, -1))
            for j, _ in enumerate(self.edges[sid]):
                segment = self.offsets[sid] + j
                t = self.times[segment] + self.lengths[segment] * self.yields[segment] / e.config['demand']['value_of_time_eur']
                rows.extend((cursor + j, cursor + j + 1))
                cols.extend((cursor + j + 1, cursor + j))
                weights.extend((t, t))
                arc_segment.extend((segment, segment))
            cursor += len(service.route)
        adjacency = sparse.csr_matrix((weights, (rows, cols)), shape=(size, size))
        self._adjacency = adjacency
        self._size = size
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        keys = rows * size + cols
        order = np.argsort(keys)
        self._arc_keys = keys[order]
        self._arc_segments = np.asarray(arc_segment)[order]
        boardings = np.flatnonzero((rows < e.n) & (cols >= e.n))
        self._boarding_indices = np.searchsorted(self._arc_keys, keys[boardings])
        node_services = [np.full(len(service.route), sid, dtype=int)
                         for sid, service in enumerate(self.services)]
        self._node_services = np.concatenate(node_services) if node_services else np.empty(0, dtype=int)
        self._boarding_services = self._node_services[cols[boardings] - e.n]

    def assign(self, frequencies):
        """All-or-nothing shortest generalized journeys on explicit services.

        Routing uses distance-based fare as an additive proxy; final fare still
        uses OD distance, preserving the no-extra-revenue-for-detours convention.
        """
        e = self.e
        q = len(e.base)
        frequencies = np.asarray(frequencies, dtype=float)
        if self._adjacency is None:
            self._prepare_assignment()
        adjacency, size = self._adjacency, self._size
        # CSR entries are sorted by the same row/column keys as _arc_keys.
        adjacency.data[self._boarding_indices] = (.5 / frequencies[self._boarding_services]
                                                  + e.config['operations']['transfer_hours'])
        sources = e.sources
        if not len(sources):
            incidence = sparse.csr_matrix((len(self.segment_edges), q))
            return self._journey_arrays(incidence, np.zeros(q), np.zeros(q), np.zeros(q, dtype=bool))
        duration, previous = shortest_path(adjacency, method='D', directed=True, indices=sources, return_predecessors=True)
        source_row = e.source_row
        connected = np.isfinite(duration[source_row, e.destinations])
        keys, values = self._arc_keys, self._arc_segments
        positions = e.destinations.copy()
        ri, ci = [], []
        boardings, wait = np.zeros(q), np.zeros(q)
        frequency_at_node = np.zeros(size)
        frequency_at_node[e.n:] = frequencies[self._node_services]
        for _ in range(size):
            pairs = np.flatnonzero(connected & (positions != e.origins))
            if not len(pairs):
                break
            nodes = positions[pairs]
            parents = previous[source_row[pairs], nodes]
            if np.any(parents < 0):
                raise RuntimeError('Invalid service journey predecessor')
            segments = values[np.searchsorted(keys, parents.astype(np.int64) * size + nodes)]
            riding = segments >= 0
            ri.extend(segments[riding].tolist())
            ci.extend(pairs[riding].tolist())
            boarding = (parents < e.n) & (nodes >= e.n)
            boardings[pairs[boarding]] += 1
            wait[pairs[boarding]] += .5 / frequency_at_node[nodes[boarding]]
            positions[pairs] = parents
        incidence = sparse.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(self.segment_edges), q))
        return self._journey_arrays(incidence, np.maximum(0, boardings - 1), wait, connected)

    def _journey_arrays(self, incidence, transfers, wait, connected):
        from .demand import fare, rail_share
        e = self.e
        distance = np.asarray(incidence.T @ self.lengths).ravel()
        weighted_yield = np.asarray(incidence.T @ (self.lengths * self.yields)).ravel()
        unit_yield = np.divide(weighted_yield, distance, out=np.zeros_like(distance), where=distance > 0)
        fares = fare(e.od_distance, unit_yield, e.config['demand'])
        ivt = np.asarray(incidence.T @ self.times).ravel()
        gt = e.access[e.origins] + e.access[e.destinations] + ivt + wait + transfers * e.config['operations']['transfer_hours'] + fares / e.config['demand']['value_of_time_eur']
        gt[~connected] = np.inf
        trips = 2 * e.base * rail_share(gt, e.config['demand'])
        edge_incidence = self.segment_to_edge @ incidence
        return dict(incidence=incidence, edge_incidence=edge_incidence, transfers=transfers, wait=wait,
                    fares=fares, ivt=ivt, generalized_time=gt, trips=trips, ridden=distance)


def generate_plans(e, x, track_counts, *, infrastructure=None, return_networks=False):
    """At most three plans: hybrid through, edge feeders, conventional through.

    A bounded set of high-demand shortest paths is used as through routes, then
    uncovered sections get edge feeders. No route invents unbuilt track.
    """
    active = np.flatnonzero(x)
    if not len(active):
        return [ServiceNetwork(e, x, (), track_counts, infrastructure)] if return_networks else [()]
    adjacency = np.full((e.n, e.n), np.inf)
    np.fill_diagonal(adjacency, 0)
    for i in active:
        speed = e.config['tracks']['CR' if x[i] == 1 else 'HSR']['speed_kmh']
        adjacency[e.a[i], e.b[i]] = adjacency[e.b[i], e.a[i]] = e.lengths[i, x[i]] / speed
    duration, previous = shortest_path(adjacency, method='FW', directed=False, return_predecessors=True)
    limit = e.config['services']['max_route_edges']
    routes, seen = [], set()
    ranked = e.ranked_od
    for j in ranked:
        a, b = int(e.origins[j]), int(e.destinations[j])
        if not np.isfinite(duration[a, b]):
            continue
        route = [b]
        while route[-1] != a and len(route) <= limit:
            route.append(int(previous[a, route[-1]]))
        if route[-1] != a or len(route) < 3:
            continue
        route = tuple(route[::-1])
        if route not in seen:
            routes.append(route)
            seen.add(route)
        if len(routes) >= e.config['services']['max_through_routes']:
            break
    if e.config['services']['max_through_routes'] == 0:
        routes = []
    lookup = {tuple(sorted((int(e.a[i]), int(e.b[i])))): i for i in active}
    edge_routes = [(int(e.a[i]), int(e.b[i])) for i in active]
    covered = {lookup[tuple(sorted(ab))] for route in routes for ab in zip(route, route[1:])}
    through = routes + [route for i, route in zip(active, edge_routes) if i not in covered]
    infrastructure = infrastructure if infrastructure is not None else e.infrastructure(x, track_counts)
    plans, networks, attempted = [], [], set()
    for paths, conventional in [(through, False), (edge_routes, False), (through, True)]:
        services = []
        # Admit routes only if their minimum frequency fits all shared resources.
        for route in paths:
            edges = [lookup[tuple(sorted(ab))] for ab in zip(route, route[1:])]
            train = 'HSR' if not conventional and any(x[i] == 2 for i in edges) and all(track_counts[i] != 1 for i in edges) else 'CR'
            services.append(Service(tuple(route), train, e.config['operations']['minimum_frequency']))
        # Resource admission is deterministic. It can leave built sections without
        # service when the bounds are impossible; construction is still charged.
        proposed = tuple(services)
        if proposed in attempted:
            continue
        attempted.add(proposed)
        network = ServiceNetwork(e, x, services, track_counts, infrastructure)
        usage = np.zeros(len(network.resource_caps))
        admitted = []
        resources = network.resource_services.tocsc()
        for sid, service in enumerate(services):
            indices = resources.indices[resources.indptr[sid]:resources.indptr[sid + 1]]
            if np.all(usage[indices] + service.frequency <= network.resource_caps[indices] + 1e-9):
                usage[indices] += service.frequency
                admitted.append(service)
        plan = tuple(admitted)
        if plan not in plans:
            plans.append(plan)
            if return_networks:
                networks.append(network if len(admitted) == len(services) else
                                ServiceNetwork(e, x, plan, track_counts, infrastructure))
        if len(plans) >= e.config['services']['plan_trials']:
            break
    return networks if return_networks else plans
