"""Joint station, all-stop service, capacity and multi-commodity passenger MILP."""
import hashlib
import math

import highspy
import numpy as np
from scipy.sparse import coo_matrix
from .lines import generate_lines
from .finance import annuity_factor, asset_pv_factor, lifecycle_report


class LinearModel:
    def __init__(self):
        self.names, self.cost, self.lower, self.upper, self.integer = [], [], [], [], []
        self.row_lower, self.row_upper = [], []
        self.ri, self.ci, self.values = [], [], []

    def var(self, name, cost=0, upper=math.inf, integer=False):
        index = len(self.names)
        self.names.append(name)
        self.cost.append(cost)
        self.lower.append(0)
        self.upper.append(upper)
        self.integer.append(integer)
        return index

    def row(self, terms, lower=-math.inf, upper=math.inf):
        index = len(self.row_lower)
        self.row_lower.append(lower)
        self.row_upper.append(upper)
        for column, value in terms.items():
            if value:
                self.ri.append(index)
                self.ci.append(column)
                self.values.append(value)

    def finalize(self):
        self.matrix = coo_matrix((self.values, (self.ri, self.ci)),
                                 shape=(len(self.row_lower), len(self.names))).tocsr()
        for key in ('cost', 'lower', 'upper', 'row_lower', 'row_upper'):
            setattr(self, key, np.asarray(getattr(self, key), dtype=float))
        self.integer = np.asarray(self.integer, dtype=bool)

    def fingerprint(self):
        digest = hashlib.sha256()
        for array in (self.cost, self.lower, self.upper, self.row_lower, self.row_upper,
                      self.integer, self.matrix.indptr, self.matrix.indices, self.matrix.data):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def objective(self, vector):
        # Elementwise reduction also avoids platform-specific BLAS floating-point
        # flag warnings on some Apple Accelerate / NumPy combinations.
        return float(np.sum(self.cost * vector, dtype=np.float64))

    def validation(self, vector, tolerance=1e-5):
        v = np.asarray(vector, dtype=float)
        if v.shape != self.cost.shape or not np.all(np.isfinite(v)):
            return dict(valid=False, error='Non-finite or wrong-length solution')
        activity = self.matrix @ v
        bound = max(float(np.max(self.lower-v, initial=0)), float(np.max(v-self.upper, initial=0)))
        rows = max(float(np.max(self.row_lower-activity, initial=0)), float(np.max(activity-self.row_upper, initial=0)))
        integer = float(np.max(abs(v[self.integer]-np.rint(v[self.integer])), initial=0))
        return dict(valid=max(bound, rows, integer) <= tolerance,
                    maximum_bound_violation=bound, maximum_row_violation=rows,
                    maximum_integrality_violation=integer)

    def solver(self, config, time_limit):
        h = highspy.Highs()
        for key, value in dict(output_flag=False, time_limit=time_limit,
                               mip_rel_gap=config.mip_relative_gap,
                               random_seed=config.random_seed, threads=1).items():
            if h.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f'Could not set solver option {key}')
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.matrix.shape[1], self.matrix.shape[0]
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = self.cost, self.lower, self.upper
        lp.row_lower_, lp.row_upper_ = self.row_lower, self.row_upper
        lp.integrality_ = [highspy.HighsVarType.kInteger if x else highspy.HighsVarType.kContinuous for x in self.integer]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = self.matrix.indptr.astype(np.int32)
        lp.a_matrix_.index_ = self.matrix.indices.astype(np.int32)
        lp.a_matrix_.value_ = self.matrix.data
        if h.passModel(lp) != highspy.HighsStatus.kOk:
            raise RuntimeError('HiGHS rejected model')
        return h


def add(terms, index, value):
    terms[index] = terms.get(index, 0) + value


class NetworkModel(LinearModel):
    def __init__(self, problem, progress=lambda message: None):
        super().__init__()
        self.problem = problem
        c = problem.config
        self.x, self.y, self.z, self.q, self.flow = {}, {}, {}, {}, {}
        self.lines = generate_lines(problem)
        station_counts = {city['id']: sum(s['city'] == city['id'] for s in problem.stations) for city in problem.cities}
        estimate = (1 + len(problem.demand) * len(c.frequencies_per_hour)
                    * sum(len(line['stops'])*(len(line['stops'])-1) for line in self.lines)
                    + sum(station_counts[d['origin']]*station_counts[d['destination']] for d in problem.demand)
                    + len(problem.stations) + len(problem.edges) + len(self.lines)*len(c.frequencies_per_hour))
        if c.require_connected_network:
            estimate += 2*len(problem.locations) + 2*len(problem.edges)
        if estimate > c.maximum_model_variables:
            raise ValueError(f'Estimated {estimate:,} variables exceeds maximum_model_variables={c.maximum_model_variables:,}; '
                             'reduce max_cities, locations_per_city, maximum_extra_lines or frequencies, or raise the explicit limit')
        self.service = {}
        self.capital, self.operating, self.revenue, self.time = {}, {}, {}, {}
        self.asset_kind = {}
        self.total_demand = sum(d['volume'] for d in problem.demand)
        maximum_capture = {city['id']: max((s['capture_fraction'] for s in problem.stations if s['city'] == city['id']), default=0)
                           for city in problem.cities}
        self.capture_upper_bound = sum(d['volume']*maximum_capture[d['origin']]*maximum_capture[d['destination']] for d in problem.demand)
        # Normalize rider/time objectives by a fixed optimistic capture bound,
        # rather than by millions of inaccessible potential trips. Otherwise small
        # center-location penalties can overwhelm all achievable rider benefit.
        normalization = max(1, self.capture_upper_bound)
        self.by_city = {city['id']: [] for city in problem.cities}
        self.by_location = {loc['id']: [] for loc in problem.locations}
        self.incident = {loc['id']: [] for loc in problem.locations}
        for station in problem.stations:
            loc = problem.locations[station['location']]
            index = self.var(f'station_{station["id"]}',
                             cost=(c.center_weight * loc['center_offset_km'] / max(1, c.station_offset_km)
                                   + c.access_weight * (station.get('access_multiplier', 1) - 1)) / len(problem.cities),
                             upper=1, integer=True)
            self.x[station['id']] = index
            self.by_city[station['city']].append(station['id'])
            self.by_location[station['location']].append(station['id'])
            self.asset_kind[index] = 'station'
            self.capital[index] = c.station_capital_cost
            self.operating[index] = c.station_annual_cost
        for city, options in self.by_city.items():
            self.row({self.x[s]: 1 for s in options}, upper=1)
        # Circles are mutually exclusive. Hence overlapping demand is never counted
        # twice, and the larger-station tie rule is unnecessary for selected sites.
        for i, a in enumerate(problem.stations):
            for b in problem.stations[i+1:]:
                if a['city'] == b['city']:
                    continue
                distance = np.linalg.norm(np.array(problem.locations[a['location']]['xy'])
                                          - problem.locations[b['location']]['xy'])
                if distance < a['radius_km'] + b['radius_km']:
                    self.row({self.x[a['id']]: 1, self.x[b['id']]: 1}, upper=1)
        for edge in problem.edges:
            index = self.var(f'track_{edge["id"]}', upper=1, integer=True)
            self.y[edge['id']] = index
            self.asset_kind[index] = 'track'
            self.capital[index] = edge['track_capital']
            self.operating[index] = c.maintenance_fraction * edge['track_capital']
        track_services = {edge['id']: {} for edge in problem.edges}
        track_options = {edge['id']: {} for edge in problem.edges}
        for line in self.lines:
            choices = {}
            for frequency in sorted(c.frequencies_per_hour):
                index = self.var(f'line_{line["id"]}_f{frequency}', upper=1, integer=True)
                self.z[line['id'], frequency] = index
                choices[index] = 1
                runtime = line['km'] / c.train_speed_kmh + (len(line['stops'])-2)*c.dwell_minutes/60
                fleet = max(1, math.ceil(2 * (runtime + c.turnaround_minutes/60) * frequency - 1e-10))
                capex = fleet * c.train_capital_cost
                departures = 2 * frequency * c.annual_hours
                opex = departures * (line['km'] * c.train_cost_per_km + c.train_cost_per_departure)
                self.asset_kind[index] = 'fleet'
                self.capital[index], self.operating[index] = capex, opex
                self.service[line['id'], frequency] = dict(fleet=fleet, runtime_hours=runtime,
                                                          annual_train_km=departures*line['km'],
                                                          annual_departures=departures)
                for edge_id in line['edges']:
                    self.row({index: 1, self.y[edge_id]: -1}, upper=0)
                    track_services[edge_id][index] = frequency
                    track_options[edge_id][index] = -1
                for location in line['stops']:
                    self.incident[location].append(index)
                    terms = {index: 1}
                    for s in self.by_location[location]:
                        add(terms, self.x[s], -1)
                    self.row(terms, upper=0)
            self.row(choices, upper=1)
        for edge_id, terms in track_services.items():
            self.row(terms, upper=c.track_trains_per_hour_per_direction)
            used = dict(track_options[edge_id])
            used[self.y[edge_id]] = 1
            self.row(used, upper=0)
        for location, options in self.by_location.items():
            for station in options:
                terms = {self.x[station]: 1}
                for index in self.incident[location]:
                    add(terms, index, -1)
                self.row(terms, upper=0)
        if c.require_connected_network:
            self._connectivity()
        capacities = {}
        all_served, served_by_origin = {}, {city['id']: {} for city in problem.cities}
        for od in problem.demand:
            conservation = {loc['id']: {} for loc in problem.locations}
            served, journey = {}, {}
            for a in self.by_city[od['origin']]:
                for b in self.by_city[od['destination']]:
                    sa, sb = problem.stations[a], problem.stations[b]
                    captured = od['volume'] * sa['capture_fraction'] * sb['capture_fraction']
                    if captured < 1e-8:
                        continue
                    index = self.var(f'passengers_{od["id"]}_{a}_{b}', upper=captured)
                    self.q[od['id'], a, b] = index
                    self.row({index: 1, self.x[a]: -captured}, upper=0)
                    self.row({index: 1, self.x[b]: -captured}, upper=0)
                    served[index] = all_served[index] = 1
                    served_by_origin[od['origin']][index] = 1
                    # outflow - inflow = injected passengers at origins.
                    add(conservation[sa['location']], index, -1)
                    add(conservation[sb['location']], index, 1)
                    self.revenue[index] = od['km'] * c.ticket_per_km * c.annual_hours
                    # Arc time includes a transfer penalty per boarding; remove
                    # one penalty here because the initial boarding is not a transfer.
                    self.time[index] = sa['access_hours'] + sb['access_hours'] - c.transfer_minutes / 60
                    journey[index] = self.time[index] - c.maximum_journey_hours
                    self.cost[index] = (-c.served_weight + c.time_weight*self.time[index]) / normalization
            self.row(served, upper=od['volume'])
            for line in self.lines:
                for frequency in sorted(c.frequencies_per_hour):
                    # A ride can board and alight at any ordered stop pair on
                    # this line. Staying aboard incurs no extra waiting/transfer.
                    for board in range(len(line['stops'])):
                        for alight in range(len(line['stops'])):
                            if board == alight:
                                continue
                            index = self.var(f'flow_{od["id"]}_{line["id"]}_{frequency}_{board}_{alight}', upper=od['volume'])
                            self.flow[od['id'], line['id'], frequency, board, alight] = index
                            add(conservation[line['stops'][board]], index, 1)
                            add(conservation[line['stops'][alight]], index, -1)
                            lo, hi = sorted((board, alight))
                            km = sum(line['segment_km'][lo:hi])
                            dwell = (hi-lo-1)*c.dwell_minutes/60
                            self.time[index] = km/c.train_speed_kmh + dwell + 0.5/frequency + c.transfer_minutes/60
                            journey[index] = self.time[index]
                            self.cost[index] = c.time_weight * self.time[index] / normalization
                            direction = int(board > alight)
                            for segment in range(lo, hi):
                                capacities.setdefault((line['id'], frequency, segment, direction), {})[index] = 1
            for terms in conservation.values():
                if terms:
                    self.row(terms, lower=0, upper=0)
            self.row(journey, upper=0)
            progress(f'Built passenger commodity {od["id"]+1}/{len(problem.demand)}')
        for (line, frequency, segment, direction), terms in capacities.items():
            terms[self.z[line, frequency]] = -c.train_seats * frequency
            self.row(terms, upper=0)
        self.row(all_served, lower=c.minimum_served_fraction * self.total_demand)
        for city, terms in served_by_origin.items():
            origin_demand = sum(d['volume'] for d in problem.demand if d['origin'] == city)
            if origin_demand:
                self.row(terms, lower=c.minimum_origin_served_fraction*origin_demand)
        # Thousands of currency units keep a small positive profit floor meaningful
        # at the solver's feasibility tolerance while scaling down capital values.
        self.appraisal_annuity = annuity_factor(c.appraisal_years, c.discount_rate)
        financial = {index: value / 1000 for index, value in self.revenue.items()}
        for index in self.capital:
            factor = asset_pv_factor(c.asset_lives_years[self.asset_kind[index]],
                                     c.appraisal_years, c.discount_rate)
            annual_equivalent = self.capital[index] * factor / self.appraisal_annuity + self.operating[index]
            financial[index] = -(1 + c.financial_margin) * annual_equivalent / 1000
            self.cost[index] += c.cost_weight * annual_equivalent / 1e8
        # Annual shortfall in thousands of euros relaxes the profitability target.
        self.profit_shortfall = self.var('profit_shortfall', cost=c.profit_shortfall_weight / 1e5)
        financial[self.profit_shortfall] = 1
        self.row(financial, lower=c.minimum_horizon_profit/self.appraisal_annuity/1000)
        if c.upfront_budget:
            self.row({i: value/1e6 for i, value in self.capital.items()}, upper=c.upfront_budget/1e6)
        self.finalize()

    def _connectivity(self):
        """Single-commodity root flow connects every selected station by track."""
        n = len(self.problem.locations)
        balances = {loc['id']: {} for loc in self.problem.locations}
        roots, supplies = {}, {}
        for location, options in self.by_location.items():
            root = self.var(f'root_{location}', upper=1, integer=True)
            supply = self.var(f'root_supply_{location}', upper=n)
            roots[root] = 1
            supplies[supply] = 1
            terms = {root: 1}
            for station in options:
                add(terms, self.x[station], -1)
                add(balances[location], self.x[station], 1)
                add(supplies, self.x[station], -1)
            self.row(terms, upper=0)
            self.row({supply: 1, root: -n}, upper=0)
            add(balances[location], supply, -1)
        self.row(roots, lower=1, upper=1)
        self.row(supplies, lower=0, upper=0)
        for edge in self.problem.edges:
            for a, b in ((edge['a'], edge['b']), (edge['b'], edge['a'])):
                flow = self.var(f'connect_{edge["id"]}_{a}_{b}', upper=n)
                self.row({flow: 1, self.y[edge['id']]: -n}, upper=0)
                add(balances[a], flow, 1)
                add(balances[b], flow, -1)
        for terms in balances.values():
            self.row(terms, lower=0, upper=0)

    def describe(self, vector):
        p, c = self.problem, self.problem.config
        validation = self.validation(vector)
        if not validation['valid']:
            return dict(feasible=False, validation=validation)
        v = np.asarray(vector)
        dot = lambda terms: float(sum(v[i]*value for i, value in terms.items()))
        served = float(sum(v[i] for i in self.q.values()))
        revenue, opex, capex = dot(self.revenue), dot(self.operating), dot(self.capital)
        horizon_revenue = c.accounting_years * revenue
        horizon_cost = capex + c.accounting_years * opex
        capitals = {kind: sum(v[i]*value for i, value in self.capital.items()
                              if self.asset_kind[i] == kind) for kind in ('track', 'station', 'fleet')}
        lifecycle = lifecycle_report(capitals, revenue, opex, c, c.discount_rate)
        sensitivity = [lifecycle_report(capitals, revenue, opex, c, rate)
                       for rate in sorted(set(c.discount_rates + [c.discount_rate]))]
        required = (1+c.financial_margin)*lifecycle['pv_cost']+c.minimum_horizon_profit
        surplus = lifecycle['pv_revenue']-required
        budget_ok = surplus >= -max(0.01, abs(required)*1e-9)
        validation['financial_ok'] = budget_ok
        stations, lines, tracks, pairs = [], [], [], []
        for station_id, index in self.x.items():
            if v[index] > 0.5:
                station = p.stations[station_id]
                location = p.locations[station['location']]
                stations.append(dict(**station, name=p.cities[station['city']]['name'],
                                     coordinate=location['coordinate'],
                                     center_offset_km=location['center_offset_km']))
        active_flows = [(key, float(v[index])) for key, index in self.flow.items() if v[index] > 1e-6]
        segment_loads = {}
        for (od, line_id, frequency, board, alight), volume in active_flows:
            lo, hi = sorted((board, alight))
            for segment in range(lo, hi):
                key = (line_id, frequency, segment, int(board > alight))
                segment_loads[key] = segment_loads.get(key, 0) + volume
        for (line_id, frequency), index in self.z.items():
            if v[index] > 0.5:
                line = self.lines[line_id]
                loads = [[segment_loads.get((line_id, frequency, segment, d), 0) for d in (0, 1)]
                         for segment in range(len(line['edges']))]
                stop_names = [p.cities[p.locations[location]['city']]['name'] for location in line['stops']]
                lines.append(dict(**line, stop_names=stop_names, frequency_per_hour_per_direction=frequency,
                                  capacity_per_direction_per_hour=c.train_seats*frequency,
                                  directional_loads=[max(load[d] for load in loads) for d in (0, 1)],
                                  segment_directional_loads=loads, **self.service[line_id, frequency],
                                  capital_cost=self.capital[index], annual_operating_cost=self.operating[index]))
        for edge_id, index in self.y.items():
            if v[index] > 0.5:
                tracks.append(dict(**p.edges[edge_id], capital_cost=self.capital[index], annual_maintenance=self.operating[index]))
        selected_capture = {city['id']: 0 for city in p.cities}
        for station in stations:
            selected_capture[station['city']] = station['capture_fraction']
            location = station['location']
            station['boardings_per_hour_including_transfers'] = sum(
                volume for (od, line, frequency, board, alight), volume in active_flows
                if self.lines[line]['stops'][board] == location)
            station['departures_per_hour'] = sum(
                line['frequency_per_hour_per_direction'] * (1 if location in (line['a'], line['b']) else 2)
                for line in lines if location in line['stops'])
        accessible = 0
        for od in p.demand:
            volume = float(sum(v[index] for (d, _, _), index in self.q.items() if d == od['id']))
            captured = od['volume']*selected_capture[od['origin']]*selected_capture[od['destination']]
            accessible += captured
            pairs.append(dict(**od, served=volume, unmet=max(0, od['volume']-volume),
                              accessible=captured, unserved_accessible=max(0, captured-volume)))
        time = dot(self.time)
        return dict(feasible=True, validation=validation, stations=stations, lines=lines, tracks=tracks,
                    passenger_flows=[dict(od=od, line=line, frequency=frequency, board_stop=board,
                                          alight_stop=alight, trips_per_hour=volume)
                                     for (od, line, frequency, board, alight), volume in active_flows],
                    od_pairs=pairs, metrics=dict(
                        objective=self.objective(v), potential_trips_per_hour=self.total_demand,
                        served_trips_per_hour=served, unmet_trips_per_hour=max(0, self.total_demand-served),
                        accessible_trips_per_hour=accessible,
                        unserved_accessible_trips_per_hour=max(0, accessible-served),
                        outside_access_or_uptake_trips_per_hour=max(0, self.total_demand-accessible),
                        served_fraction_of_accessible_demand=served/accessible if accessible else None,
                        served_fraction=served/self.total_demand,
                        passenger_hours_per_hour=time, mean_journey_hours=time/served if served else None,
                        total_track_km=sum(track['km'] for track in tracks),
                        fleet_trainsets=sum(line['fleet'] for line in lines),
                        annual_train_km=sum(line['annual_train_km'] for line in lines),
                        annual_revenue=revenue, annual_operating_cost=opex, capital_cost=capex,
                        horizon_revenue=horizon_revenue, horizon_cost=horizon_cost,
                        horizon_profit=horizon_revenue-horizon_cost,
                        lifecycle=lifecycle, discount_rate_sensitivity=sensitivity,
                        profit_target_basis='lifecycle_present_value',
                        profit_target_met=bool(budget_ok),
                        profit_shortfall=max(0, -surplus),
                        profit_shortfall_slack=v[self.profit_shortfall]*1000*self.appraisal_annuity,
                        profit_shortfall_penalty=self.cost[self.profit_shortfall]*v[self.profit_shortfall],
                        financial_surplus_after_margin=surplus,
                        payback_years=capex/(revenue-opex) if revenue > opex else None))
