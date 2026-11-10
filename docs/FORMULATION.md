# Railway planning algorithm and mathematical formulation

This document describes the algorithm implemented in `train_optimizer/`.
[README.md](../README.md) covers installation, commands and output files.
[config.json](../config.json) holds the current assumptions. The optimizer is a
strategic scenario model with a finite set of station and line candidates.

## Solver modes

`solver_method="heuristic"` is the default. Choose the station with greatest
capture fraction in each market, breaking ties by lowest access time and ID.
Route only those station sites. Add up to `heuristic_demand_connections` extra
corridors per market, ranked by potential passenger-km, within the existing
edge/trip distance limits. Generate direct lines and up to
`heuristic_through_lines` three-stop lines (also respecting maximum_extra_lines,
maximum_line_stops and the shared-track detour limit), prioritizing endpoint OD
demand. Alignment variants remain bounded. Use dense networks and cheap/short
trees as starting points. Rank neighborhood proposals using all-pairs generalized
travel times weighted by captured demand, then evaluate using the full LP
objective. Moves include shortcuts, through-service replacements, small-city
bypasses, pruning, frequency changes and alignment replacements. A tree is not
required for the final network.

Every station, track, service and connectivity-root binary is fixed for each LP.
Published vectors pass the full row, bound and integrality checks. Fixed stations
and the finite through-service pool can still miss better or feasible designs.
The heuristic has no dual bound or optimality guarantee. `heuristic_iterations`
limits proposals and `random_seed` controls restarts; interruption preserves the
incumbent. Failure to find a design is not proof of infeasibility.

When `maximum_journey_detour_ratio > 0`, the OD-average time limit is the smaller
of `maximum_journey_hours` and `access_a + access_b + 0.5/minimum_allowed_frequency
+ maximum_journey_detour_ratio * direct_OD_km / train_speed_kmh`. This includes
waiting and transfers in actual journey time. The root config uses ratio 2;
zero disables the relative cap. It applies to both solver modes. Financial
shortfall remains a soft penalty, not a guarantee of profitability.

`--solver milp` (or JSON `solver_method="milp"`) retains the full formulation,
station alternatives and line patterns, and the existing HiGHS incumbent search.
Checkpoints are fingerprinted against each generated model, so switching modes
normally requires a fresh output directory. Preparation and construction time
are additional to the solver time limit in both modes.

## 1. Configuration, data and units

Run from the project directory:

```sh
.venv/bin/python optimize.py --output runs/my-scenario
```

The CLI loads `config.json`, then applies any file passed with `--config` as
top-level overrides. CLI solver-duration and animation options override JSON.
Paths resolve relative to the working directory. `--write-default-config PATH`
exports the built-in `Config` defaults, rather than the currently edited JSON.
Unknown keys and invalid parameter values are rejected.

Money is in euros, length in km, time in hours, density in people/km² and demand
in directional trips/hour. Defaults include 500 seats, 120 km/h train speed,
25 km/h access speed, 06:00–22:00 operation and ten accounting years.

The current population source is `data/europe_3km.geojson`, with H3 resolution 6.
The older `europe_22km.geojson` is also available. Both contain population points
with `h3Index`; their names describe approximate cell sizes, not square rasters.
For each supplied H3 cell g:

```text
n_g = population_g
D_g = n_g / H3_area_g_km²
```

Use actual H3 area, not a single area inferred from the filename. Population
units, coverage and reference year require source verification. Missing cells
are unknown. City records provide optional labels only; their populations never affect seeds
or demand and are never added to the population cells.

The bounding box and `countries` polygons filter population points. Seeds are
derived only from those points. `country_boundaries_file` supplies country
polygons; an empty country list disables the polygon filter. Missing country
codes are an error. Cells and subdivided samples are included by their centers.
Routes are not constrained to political boundaries. Local distances use a regional azimuthal equidistant projection;
corridor and direct OD distances use great-circle distances.

## 2. Population samples, city centers and station candidates

When `population_subdivision=0` (default), each cell contributes its H3 center
and full population. Values 1 or 2 distribute population among finer H3 children
in proportion to their areas, assuming uniform population within the parent.
Child centers outside the bounding box are excluded.

All seed candidates are eligible population cell centers. City records never
participate in seeding. Score centers using a distance-weighted sum of eligible
population within `market_radius_km`:

```text
sigma = market_radius_km / 3
weight(d) = exp(-0.5 * (d / sigma)**2) for d <= market_radius_km
score(center) = sum(population_g * weight(distance(center, g)))
```

This favors compact urban centers over locations between multiple towns.
`seed_population` reports the unweighted neighborhood total; `seed_score` and
`seed_marginal_population` report weighted scores. After every selection, remove its scoring neighborhood
from the remaining candidates' scores. This greedy marginal-coverage score
rewards newly represented population rather than repeatedly rewarding the same
metropolitan population. Require at least `minimum_city_population` of new
unweighted population. H3 IDs break ties deterministically. Retain up to `max_cities` seeds,
separated by `minimum_seed_spacing_km` (zero means `market_radius_km`). Spacing
is independent of the smaller density-adaptive catchment radius.

For each selected seed, set `radius_km = center_coefficient * sqrt(seed density)`
using the seed population cell density in people/km², with no minimum or maximum.
The default coefficient 0.16136 gives Brussels approximately 20 km at a seed
density of 15,362.67 people/km². `market_radius_km` controls seed scoring only;
`minimum_city_population` screens seeds, without setting their radius.
At least two markets are required. City names are optional labels.

The MILP chooses which candidate markets to serve. Seeds are fixed during a solve:
moving them would change origin populations and demand and invalidate comparisons
between incumbent solutions. Greedy coverage improves geographic representation,
not guaranteed network optimality; optimality remains relative to the candidates.

For `center_method="access"`, collect population samples within
the adaptive radius of the population seed and numerically minimize:

```text
center_i = argmin_z sum(g in local_i, n_g * distance(g, z)) / sum(g in local_i, n_g)
```

This is a population-weighted geometric median, computed with Powell's method
and `access_optimizer_tolerance_km`. Local center generation uses the local
population samples before demand eligibility and origin-zone restrictions.
If there are no local samples, the city seed is the center.

Two legacy options remain configurable: `peak` picks the highest-density sample;
`circle` uses the market’s adaptive radius and compares
covered population at the seed and local sample centers. Neither is the default.

Station locations include the computed center, the seed when sufficiently
separated, and nearby population-sample centers ordered by proximity to the
computed center. Sites must be more than 1 km apart within a city, and at most
`locations_per_city` are retained. Every location has exactly one catchment with the market’s adaptive radius. The optimization selects among these finite options;
it does not move stations continuously.

## 3. Fixed origin zones, catchments and uptake

Assign each population sample to its nearest population seed, independently of
station-opening decisions. Only samples within the adaptive radius of that seed and with density
at least `density_threshold` contribute to the city's eligible population N_i and demand.
A station cannot capture residents assigned to another city's zone. The target
is evaluated before exclusive ownership; final market population and station
coverage can therefore be below the target. Population is counted in whole cells.

The regional scenarios set `uptake_reference_radius_km=25`, separating modeled
willingness to use rail from the density-adaptive geometry. This remains an
uncalibrated assumption; zero preserves catchment-dependent uptake.

For station/radius option s belonging to city i:

```text
G_s = eligible samples in zone_i with distance(g, station_s) <= radius_s
covered_s = sum(g in G_s, n_g)
uptake_radius_s = uptake_reference_radius_km if positive, otherwise radius_s
uptake_s = p_max * (1 - exp(-uptake_a * uptake_radius_s**uptake_alpha))
capture_s = (covered_s / N_i) * uptake_s
```

Zero-coverage options are omitted. Uptake is an aggregate fraction for a whole
catchment, not an individual distance-decay curve. It requires calibration.

At most one station/location/radius option can open per city. Circles from
different markets may overlap. Each population cell belongs to one fixed nearest
seed zone, and only its zone's station can capture it, preventing duplicate demand.
Cross-zone station access is not modeled. The optimizer does not use the standalone
`rail_model.py` size-rank assignment helper for station selection.

## 4. Population-weighted station access

For the same covered residents G_s, define:

```text
mean_access_km_s = sum(g in G_s, n_g * distance(g, station_s)) / covered_s
ideal_access_km_s = min_z sum(g in G_s, n_g * distance(g, z)) / covered_s
access_multiplier_s = max(1, mean_access_km_s /
                             max(ideal_access_km_s, access_normalization_floor_km))
access_hours_s = mean_access_km_s / access_speed_kmh
```

The ideal distance is numerically estimated at the weighted geometric median of
those same residents. Their membership is held fixed during this normalization.
A symmetric center has multiplier 1, as does any weighted geometric median;
less favorable positions incur a larger multiplier, subject to the small
numerical floor. This ratio measures placement efficiency, while the absolute
access hours measure how long residents take to reach the station. Distances
are straight-line projected distances, not road-network travel times.

The objective charges `access_weight * (access_multiplier_s - 1)` per selected
station, divided by the number of city seeds. Subtracting 1 removes a constant
opening penalty. Access and egress hours also enter served passenger journey
time. `center_weight` is an optional legacy center-offset penalty and defaults
to zero.

## 5. Fixed potential demand and rail capture

Distances d_ij are between population seed coordinates. Exclude pairs farther than
`maximum_trip_distance_km`. For default `demand_model="production"`:

```text
w_ij = N_j / max(d_ij, distance_floor_km)**gravity_beta
Q_ij = trips_per_person_hour * N_i * w_ij / sum(k eligible destinations, w_ik)
```

If there are no eligible destinations, that origin generates no OD records.
Optional `gravity` mode instead sets:

```text
Q_ij = gravity_scale * N_i * N_j / max(d_ij, distance_floor_km)**gravity_beta
```

These are directed potential trips/hour. For selected station options a in i
and b in j, served endpoint flow q_ijab is bounded by:

```text
0 <= q_ijab <= Q_ij * capture_a * capture_b
q_ijab <= Q_ij * capture_a * capture_b * x_a
q_ijab <= Q_ij * capture_a * capture_b * x_b
sum(a,b, q_ijab) <= Q_ij
```

Here x_s is a binary station-opening variable. Capture at both ends assumes
independent access/uptake. Potential demand remains fixed if a city has no open
station. Reported unmet demand is potential minus served, including trips that
fail access or uptake; accessible-but-unserved trips are reported separately.

## 6. Corridors and railway construction costs

Select nearby city pairs with `neighbors_per_city` and `maximum_edge_km`, then
route their station-location combinations. Always evaluate a direct WGS84
geodesic, integrating terrain and density along it. Reject it when sampled
blocked terrain or the land mask forbids it. A weighted Theta* search uses H3
cells as waypoints but permits straight parent-to-neighbor segments across cells;
both ordinary steps and shortcuts are compared using integrated construction
cost. Search stops at `terrain_search_limit`. Keep a cheaper any-angle alternative
when found, alongside the direct candidate. This finite search is not a proof
of the cheapest continuous alignment.

Sampling spacing is the smaller of `routing_sample_km` and one third of the
population H3 resolution's average edge length. Costs use segment midpoints;
sample endpoints and midpoints check blocked terrain, and the land mask checks
entire sampled polylines. Features smaller than sampling or source resolution
can still be missed. Raster colors are not engineering obstacles or gradients.

Alternative alignments have distinct physical edge IDs and actual lengths,
construction costs, capacity and maintenance. Line patterns expand into up to
`maximum_alignment_variants_per_line` combinations (default 16), always retaining
fastest and cheapest combinations before bounded mixed combinations. Only one
frequency/alignment choice may operate per stop pattern. Different patterns may
share the same physical edge; coincident geometry alone is not shared track.
The network objective compares passenger time, fleet and operating costs, and
construction/lifecycle cost, so direct routes can win despite higher terrain
cost. No fixed preference forces a detour. The distance ceiling applies to each
actual routed alignment.

For density D, the conventional construction estimate is:

```text
C(D) = 10 + 0.005 max(0, D - 5000)          [million euros/km]
     = base_track_cost_per_km + urban_cost_coefficient * max(0, D - urban_density_threshold)  [euros/km]
```

Defaults are €10,000,000/km, a threshold of 5,000 people/km², and €5,000 per
unit of density above that threshold. There is no urban surcharge at or below
the threshold; it increases continuously and linearly above it. Terrain-class
baselines are supplied in `terrain_cost_ranges_million`:

| `terrain_class` | Low–high (M€/km) | Midpoint (M€/km) |
| --- | ---: | ---: |
| `bridge` | 20–50 | 35 |
| `hilly` | 15–25 | 20 |
| `moderate_mountains` | 25–40 | 32.5 |
| `difficult_mountains` | 40–70 | 55 |
| `very_difficult` | 70–120+ | 95 |

For a classified cell, with p = `terrain_cost_position` (default 0.5):

```text
terrain_baseline = 1,000,000 * (low + p * (high-low))
base_g = max(base_track_cost_per_km, terrain_baseline)
C_g = (base_g + urban_cost_coefficient * max(0, D_g - urban_density_threshold)) * multiplier_g
```

Unclassified cells use the conventional baseline. Multipliers default to 1 and
must be at least 1. Raise the configurable 120 upper value for more expensive
base tunnels. A terrain class replaces the baseline rather than adding a second
full railway cost; the urban surcharge is then added. Missing population is
priced at base cost, an optimistic unknown-density assumption.

`terrain_file` contains H3-keyed `terrain_class`, `multiplier` and/or `blocked`
attributes at the population H3 resolution. `land_file` contains allowed-land
WGS84 polygons. Neither classes nor bridge crossings are inferred automatically.
Route cost sums segment length times the cost of the segment midpoint's H3 cell.
Shared candidate track is charged once regardless of how many services use it;
geometric crossings do not create shared track or junctions.

```text
CAPEX = selected track cost + selected station cost + dedicated line-fleet cost
```

`generate_terrain.py` independently averages a supplied DEM into 3 km and 22 km
EPSG:3035 GeoTIFFs. These square elevation grids are not H3 cost surfaces and are
not consumed directly by the optimizer. A source DEM is not bundled. Grades,
curvature, tunnel design and exact coastlines are outside this model.

## 7. All-stop services, fleet and capacity

Candidate lines include every two-stop corridor and a bounded pool of simple
multi-stop paths with no repeated city. Extra paths favor low detour and are
limited by `maximum_line_stops` and `maximum_extra_lines`.
An additional `maximum_shared_track_lines` pool (default 40) reserves three-stop
hub services that generic ranking omitted. Their length/direct-distance ratio
must not exceed `maximum_shared_track_detour` (default 1.35). Ranking estimates
the avoided direct-edge capital minus the cheaper spoke's capital and one
station's capital, assuming the other spoke may already be shared. This is only
a candidate-ranking heuristic; the MILP pays all actual costs and chooses the
network jointly. Candidate hubs are existing station locations, not new free
junctions, and all stops incur dwell and station costs. Multiple services on the
same physical edge share its construction and maintenance costs and its
directional track throughput limit.

For line l and configured frequency f, binary z_lf selects that frequency option.
At most one frequency is selected per line. Defaults are 1, 2 and 4 departures
per hour in each direction. Every line stop must open and every required track
must be built; every open station must be served and built track must be used.
Selected lines stop at every listed station.

```text
directed segment passenger load_l,f,k <= train_seats * f * z_lf
sum(lines using track_e, selected frequency) <= track_trains_per_hour_per_direction
```

Through passengers consume seats on all segments traversed, without another wait
at intermediate stops. Dedicated fleet size uses the ceiling of round-trip cycle
time times frequency, including running time, intermediate dwell and turnarounds.
Fleet is not pooled across lines. Physical-track connectivity of every selected
station is enforced by auxiliary root-flow constraints when
`require_connected_network=true`.

## 8. Passenger flow and journey time

The implementation uses an OD multi-commodity flow model, not enumeration of
complete itineraries. For each OD, a ride arc can board and alight at any ordered
stop pair on any line/frequency option. Flow conservation at station locations
links those arcs to served endpoint flows. Ride flows consume directional segment
capacity, which also prevents riding an unselected service.

A ride arc's time coefficient is:

```text
ride_time = segment_km / train_speed_kmh
            + intermediate_stops * dwell_minutes / 60
            + 0.5 / frequency + transfer_minutes / 60
```

Endpoint flow contributes origin access plus destination egress, minus one
transfer penalty so the initial boarding is not counted as a transfer. Thus a
journey includes access, egress, initial and transfer half-headway waits, running,
intermediate dwell, and a penalty for each transfer after initial boarding.

For each OD, total passenger-hours must be at most
`maximum_journey_hours * served_OD`. This is an OD-average limit; it does not
bound each individual's itinerary. Hourly aggregate capacity assumes steady
loads and evenly spaced services, not queues or a minute-by-minute timetable.

## 9. Operating costs and ten-year payback

Annual operating hours are `(operating_end_hour - operating_start_hour)` times
`operating_days_per_year`: 5,840 by default. This is a 16-hour/day approximation,
not an inclusive count of departures at both endpoints of the service window.

```text
annual_departures_l = 2 * frequency_l * annual_hours
annual_train_km_l = annual_departures_l * line_km_l
O = train-km costs + departure costs + track maintenance + station annual costs
R = sum(OD served trips/hour * direct_OD_km * ticket_per_km * annual_hours)
```

Both directions incur operating costs. Track maintenance is
`maintenance_fraction * selected_track_capital`. Revenue uses served passengers
and fixed direct OD distance, so railway detours do not increase billed distance.
Construction is paid at time zero with no passenger construction-delay term.

For T = `appraisal_years` (30), r = `discount_rate` (4%), define
A = sum(t=1..T, (1+r)^(-t)). Each asset group has full-cost replacements at
multiples of its configured life strictly before T, and straight-line
remaining-life residual value at T. Default track/station/fleet lives are
40/50/25 years; these are editable scenario assumptions.

```text
PV_cost = CAPEX + A * O + PV_renewals - PV_residual
PV_revenue = A * R
NPV = PV_revenue - PV_cost
PV_revenue + shortfall >= (1 + financial_margin) * PV_cost + minimum_horizon_profit
```

The nonnegative shortfall is reported in present-value euros; internally its
annual equivalent is stored in thousands of euros. The financial margin defaults
to 5% and the profit floor to €1 in present value. A soft penalty allows a
loss-making network. Equivalent annual cost is PV_cost/A. Zero discount is
supported (A=T). Sensitivity at 2%, 4%, and 6% evaluates the selected network;
it does not rerun optimization. Cash flows are end-of-year, constant real prices.

The separate `accounting_years=10` report remains undiscounted:
`horizon_profit = Y*R - CAPEX - Y*O`, excluding renewals and residuals.
It no longer drives selection. A positive upfront budget still constrains
initial CAPEX only. Land and asset subcomponents are not yet separated.

## 10. Objective, constraints and solve

Let S be total served trips/hour, T total served passenger-hours/hour, and
Q_total the fixed potential demand. Define a fixed optimistic capture bound:

```text
B = max(1, sum(ij, Q_ij * max_capture_i * max_capture_j))
```

The implemented weighted objective is:

```text
minimize -served_weight * S / B
         - population_weight * sum(s, x_s * covered_population_s) / total_market_population
         + time_weight * T / B
         + access_weight / number_of_cities * sum(s, x_s * (access_multiplier_s - 1))
         + center_weight / number_of_cities
             * sum(s, x_s * center_offset_km_s / max(1, station_offset_km))
         + cost_weight * (PV_cost / A) / 100,000,000
         + profit_shortfall_weight * (shortfall / A) / 100,000,000
```

The served term is equivalent to minimizing unmet demand up to a fixed constant.
Population coverage counts residents in exclusive market zones within operating
stations' catchments, before uptake. It rewards absolute population, not density
alone. Defaults are population_weight=1, served_weight=10, time_weight=0.5,
access_weight=0.05, cost_weight=0.001 and profit_shortfall_weight=0.25.
All else equal, saving an hour on twenty existing passenger trips offsets the
ridership reward for one additional trip. Journey time includes access, waiting,
transfers and in-vehicle travel; this does not directly optimize track length.
The €100M annual-equivalent denominator is an implementation scaling constant.
Weights are configured in JSON; this is a single weighted objective, not a
lexicographic optimization or automatic Pareto sweep.

Hard constraints include the station/line/track consistency rules,
directed segment capacity, track throughput, optional connectivity,
OD-average journey limits, optional upfront budget, and:

```text
S >= minimum_served_fraction * Q_total
served_from_i >= minimum_origin_served_fraction * potential_from_i
sum(selected_city_i) >= ceil(minimum_city_served_fraction * number_of_cities)
served_from_i >= minimum_selected_origin_served_fraction * potential_from_i * selected_city_i
```

Only frequencies at or above `minimum_service_frequency` are generated. Each
selected station must have operating service. `maximum_annual_subsidy`, when
positive, caps lifecycle equivalent annual cost minus annual fare revenue;
unlike the soft profit target it cannot be relaxed by financial slack.

The global served fraction defaults to 0.001; the per-origin fraction defaults
to zero. Both refer to fixed potential demand, including failed access/uptake.
The positive global target excludes an empty profitable network.

HiGHS solves the MILP with a configurable time limit and relative gap.
`maximum_model_variables` guards against oversized models. Every exported
incumbent is checked for bounds, integrality, linear-constraint feasibility and
financial consistency. Optimality is relative to the finite generated candidates.

## 11. Progress, outputs and current scenario

The run writes a fixed set of latest-result files and appends rendered frames to
`progress.gif`. Initial and final frames are always saved. `snapshot_every`
counts native solver progress callbacks, not seconds, new incumbents or completed
branch-and-bound nodes. Rendering runs independently and can coalesce updates.
`progress.jsonl` stores compact history; `manifest.json` stores reproducibility
metadata. A checkpoint exists only when a feasible incumbent has been saved.
Resume is a compatible incumbent warm start, not restoration of the search tree.

The historical `runs/netherlands-belgium-3km/` scenario predates population-only
seeding and unified 25 km markets. It used eight city seeds, 3,621
population cells, 32 station options, 56 track candidates and 96 candidate lines.
It reached the 120-second solver limit without a feasible incumbent. This does
not prove infeasibility. Its map shows candidates, not a selected railway network.
It has no supplied terrain classes or land mask. Previous outputs are preserved
under `runs/archive_previous/`.

Future extensions include observed OD calibration, road-based station access,
explicit national masks, engineering alignments, express services, multiple train
types, shared fleets, timetable/queue models and scenario-based financial checks.
