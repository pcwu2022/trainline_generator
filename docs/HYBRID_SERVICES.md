# Hybrid train services in rail_gen

The `hybrid-services-v1` evaluator separates the newly constructed railway from
its train services. This supersedes the edge-as-train evaluator described in the
original v0.1 specification. The legacy optimizer in `train_optimizer` is unchanged.

## Infrastructure and rolling stock

The infrastructure genome has exactly three states: `0` unbuilt, `1` conventional
(CR) and `2` high-speed (HSR). Each built edge selects that infrastructure's cached
alignment, length, construction cost, speed limit and train-movement capacity.
**Every built section incurs full new-construction costs.** A conventional extension
is a new railway, not an existing line or a discounted/free HSR extension.
Shared raster cells and built stations are purchased once. As before, shared
cells are an approximation to common infrastructure; they do not create junctions
or passenger interchange unless routes share a station node.

`tracks.CR` and `tracks.HSR` contain only infrastructure parameters:
`speed_kmh`, `base_meur_km`, `grade_limit`, and `max_frequency`.
`rolling_stock.CR` and `rolling_stock.HSR` contain `speed_kmh`, `seats`,
`train_meur`, `opex_eur_trainkm`, and `yield_eur_pkm`.

Compatible gauge, electrification and signalling are assumed across the newly
built network. Both train types may use either infrastructure type. Section
speed is `min(infrastructure.speed_kmh, rolling_stock.speed_kmh)`. Defaults retain
the previous illustrative 140/230 km/h speed meanings, applied to both layers;
there is no new claim about acceleration, engineering design speed or signalling.
An HSR train on conventional track runs at 140 km/h but retains HSR seats,
purchase price, operating cost and fare yield. CR stock on HSR track remains
limited to its 140 km/h capability.

Track count is independent of the three-state genome. `infrastructure_layout`
sets the run-wide search layout (`single`, `double`, or `four`; default `double`).
Programmatic evaluation also accepts `track_counts=[...]` with 1/2/4 on built
edges and zero elsewhere. The short-lived 3/4/5 genome encoding is retired.
Searches optimize infrastructure type and services. Automatic repair can widen
overloaded segments toward four tracks when doing so carries more passengers;
explicit track counts remain fixed. This is bounded capacity repair, not an
exhaustive optimization over every track-count combination.

| Layout | Track construction cost | CR infrastructure trains/h/direction | HSR infrastructure trains/h/direction |
| --- | ---: | ---: | ---: |
| Single | 0.5 × | 1 | Prohibited |
| Double | 1 × | 6 | 12 |
| Four | 2 × | 12 | 24 |

`track_layouts` controls these costs and limits. The infrastructure's
`tracks.<type>.max_frequency` remains a double-track limit; lowering it scales
other layouts' limits proportionally, bounded by the layout limit. Capacity is
shared by **all** train types, not an independent allowance per train type.
The earlier single-track policy is retained: neither HSR infrastructure nor HSR
rolling stock is permitted on single track. Its low capacity is a passing-loop
approximation, not a simulated meet schedule.

## Services and actual passenger journeys

A `Service(route, train_type, frequency, formation)` is a symmetric, all-stop train
service with a continuous, simple station route. Frequency is departures/hour in
each direction. Formation is the number of coupled trainsets and stays constant
along the whole route. Every section must be built and compatible. Explicit
invalid routes, invalid formations and forbidden single-track operations raise
`ValueError`. A valid but overloaded or financially infeasible plan is evaluated
and returned with `feasible=False`.

```python
from rail_gen.evaluator import Evaluator
from rail_gen.services import Service

# graph contains A--B--C as its first two candidate edges.
result = Evaluator(graph, config).evaluate(
    [2, 1],
    services=[Service((0, 1, 2), 'HSR', frequency=2, formation=2)],
    details=True,
)
```

Passengers route over a station/service expanded graph. Riding across the HSR/CR
boundary on one service incurs no transfer and no second wait. Boarding costs
half that service's headway. Changing service incurs another boarding wait plus
`operations.transfer_hours`; the first boarding has no transfer penalty.
`transfer_angle_degrees` is accepted in old configs but is no longer used.
Each section retains the prior `dwell_hours` convention, including the terminal
section. Routes are bidirectional and demand remains symmetric: every unordered
market represents two one-way markets. There are no one-way or express services.

Routing minimizes travel, waits, transfers and an additive fare-time proxy based
on ridden section distance and stock yield. The final fare retains the existing
OD-distance and taper convention, with distance-weighted stock yield for mixed
journeys; a detour does not itself earn extra fare. Final generalized time,
legacy rail-share demand and revenue use that final fare. Reference-elasticity
demand uses fare-free journey hours and a fixed gravity baseline without logistic
uptake; fares affect routing but not its elasticity ratio. Because OD fares are not
additive, the proxy-selected journey need not be the exact minimum final-fare
journey. Access time remains population-weighted. In elasticity mode, capacity
fulfillment uses assigned potential demand, which changes with journey quality;
the fixed gravity denominator applies only in legacy rail-share mode.

Assignment is all-or-nothing to a best generalized journey. Parallel services
are not pooled into a combined boarding frequency; passengers choose one service
at each boarding. Each service section
has its own seats × formation × frequency capacity, with the existing annual
hours, peak factor and maximum load factor. An OD's carried demand is scaled by
its most constrained ridden service section. Loads are then aggregated onto
infrastructure; capacity is not independently granted again to each overlapping
service. There is no iterative overflow rerouting onto spare parallel services.

All service frequencies using an edge are summed. Shared raster tiles also sum
frequencies across services. A through service is counted once on a tile even if
adjacent abstract edges share it; this avoids double-counting a common endpoint
cell. The tile uses the largest constructed movement capacity among its owners.
The model does not resolve spatial revisits, conflicting junction movements,
mixed-speed headways, platforms, or signal blocks. Track overload is a separate
fifth constraint, independent of passenger spill. Explicit services retain their
requested frequencies and expose violations; generated services ration scarce
train paths before being accepted.

## Fleet and annual finance

For each service:

- One-way hours sum **all** section times at the effective stock/track speed.
- Cycle hours = `2 * one_way_hours * (1 + turnaround_overhead)`.
- Active formations = `ceil(cycle_hours * frequency)`; active units are this count
  times `formation`. Fleet units = `ceil(active_units * (1 + spare_ratio))`.
- Annual unit-km = `2 * frequency * route_km * hours_per_day * days_per_year * formation`.
- Annual train operating cost = `annual_train_km * opex_eur_trainkm *
  (1 + additional_unit_cost_fraction * (formation - 1))`, where annual train-km
  excludes formation size. `operations.additional_unit_cost_fraction` defaults
  to `0.6` and accepts values from 0 to 1. Each extra coupled trainset adds 60%
  of the single-set operating rate; a pair costs 1.6 times the single-set rate.
  The existing operating overhead applies after this calculation. Capacity,
  physical unit-km and fleet purchases still count every trainset in full.
- Purchase and operating costs use that service's rolling-stock parameters
  throughout, including conventional sections of an HSR through service.

Default trainset purchase prices (`rolling_stock.CR.train_meur` and
`rolling_stock.HSR.train_meur`) are EUR 10 million and EUR 30 million. Single-set
operating rates remain EUR 12/km and EUR 20/km, respectively. The purchase
assumptions use historical Italian procurement as a guide; the 60% marginal
operating fraction is a modeling assumption for sensitivity analysis.

Older configuration overrides that omit the coupling parameter inherit the new
default when loaded through `load_config`. Set the fraction to `1.0` to reproduce
the former linear operating-cost scaling, and explicitly set purchase prices
to 20/35 to reproduce the former purchase assumptions. Saved run artifacts keep
their original results; changing defaults does not recalculate past runs.

Fleet rounding occurs per complete service and rounds whole active formations before
multiplying by units, with no pooling between services. Spare units may be held
individually; operating formations always remain complete.
The existing capital recovery factors, asset lives, overhead, maintenance,
station costs, subsidy/budget/spill limits and objectives remain: net annual cost,
negative carried trips, negative population at built stations, and optional mean
generalized time. Population coverage continues to mean built station coverage;
passenger and need-satisfaction metrics reflect actual operated journeys.

Corridor allocations reconcile to network totals: fare revenue by ridden distance,
service fleet by section running time, operations by section train-km weighted
by each service's operating multiplier and rate, stations by
built degree, and shared construction by cell ownership at the highest required
construction cost. Track cost multipliers affect construction and maintenance,
not fleet or station prices. Explicitly built but unused infrastructure is still
charged. Automatic `repair=True` retains the search's optional unused-edge removal
and regenerates services after any removal; explicit supplied plans are never
silently rewritten or pruned.

## Bounded service optimization and search

For every infrastructure proposal, all service and passenger effects are
re-evaluated. No rolling-stock type is encoded as a third active edge state.
The same evaluator is used by incremental search, capacity packages, NSGA-II,
NSGA-III, benchmarking, reporting and analysis.

The deterministic generator computes infrastructure shortest paths and ranks
eligible OD pairs by fixed gravity demand (ties by station indices). It admits
at most `services.max_through_routes` simple through paths, each with at most
`services.max_route_edges` sections. Uncovered edges get feeder services. Up to
`services.plan_trials` plans are compared:

1. Through routes with HSR stock where a route includes HSR infrastructure,
   alongside CR feeder services where appropriate.
2. Section-by-section services, permitting different trains at a junction.
3. The through-route plan with conventional rolling stock.

Duplicate plans are skipped. No more than `max_through_routes + built_edges`
services occur in a plan. A route is admitted only if its minimum frequency fits
all shared resources, in deterministic order. Admission may leave a built section
unserved when configured capacity is insufficient; that section is still charged.
The default bounds are 16 through routes, 12 sections/route, and 3 plans.

Frequency sizing runs at most `operations.assignment_iterations` rounds (default
3). It requests discrete headways to target load factor, then proportionally
rations additional movements above minimum service frequencies against shared
track capacity. Resulting rationed frequencies may be fractional. Formations are
then sized to the maximum load anywhere on a service, within
`maximum_train_units`. `services.optimize_frequencies=false` disables the demand
updates but still fits initial frequency to shared capacity and sizes formations.
Explicit supplied service plans bypass automatic sizing.

Plans are ranked first by total positive normalized constraint violation, then by
net annual cost, then by carried trips. This bounded inner economic heuristic is
not an exhaustive service Pareto optimization; the outer search still retains its
infrastructure Pareto archive and growth/consolidation rules. Incremental search
can retain spill-only infeasible live networks for capacity expansion and
calibration, while the archive remains restricted to fully feasible alternatives.
Budget and other non-spill constraints remain hard. In
particular, it may choose CR stock even on HSR infrastructure if HSR operating
costs are not justified. HSR-only stock on an entirely conventional route can be
evaluated explicitly, but is not generated automatically. There is no randomness
inside service generation, so fixed-seed searches and threaded evaluation are
reproducible. No external solver or raster routing is used inside an evaluation.

Runtime increases relative to the old single infrastructure shortest-path
assignment: each plan requires an expanded service graph, up to three frequency
assignments plus final assignment, service capacity and shared tile accounting.
The graph has stations plus one node per service stop. Routing uses sparse
Dijkstra from OD origins; planning uses an infrastructure all-pairs shortest path.
More routes, route length, iterations, and plans increase time and memory. The
bounds deliberately trade service optimality for evaluation speed. Benchmark a
representative region before selecting large search populations.

## Configuration and saved-result migration

`load_config` automatically migrates old `tracks.<type>` rolling-stock fields
(`seats`, `train_meur`, `opex_eur_trainkm`, `yield_eur_pkm`) into `rolling_stock`.
A legacy `tracks.<type>.speed_kmh` initializes both speed limit and train
capability unless the same override explicitly specifies rolling-stock speed.
Explicit new rolling-stock keys win within that override. Migrations apply to
external defaults, selected profiles and CLI/programmatic overrides before the
normal strict merge/validation. New saved `run_config.json` uses only the split
schema. Old HSR frequency limits of 6 remain 6 if explicitly supplied; the new
default is 12. Existing scripts that directly mutate a loaded config must now use
`rolling_stock` for train parameters.

Old 0/1/2 genomes can be re-evaluated directly but **old objectives, fleets, spill
and passenger metrics are not numerically compatible**. For the retired layout
encoding, use an explicit conversion:

```python
from rail_gen.track_layout import migrate_legacy_genome
x, counts = migrate_legacy_genome(old_genome)
new_result = evaluator.evaluate(x, track_counts=counts)
```

States 3/4/5 otherwise raise an explanatory error. Search does not resume solver
or RNG state from historical results. Original result files should be retained;
start a new output directory for a new search.

Results carry `model_version=hybrid-services-v1`. Live snapshots and checkpoints
contain infrastructure, independent counts and the **evaluated** service plan.
Pareto genomes have a model-version field and companion `.services.json` files.
Replaying services uses `Service.from_dict` or `evaluate(..., services=records)`;
extra result fields in each service record are ignored when reading decisions.
Reports and cost logs list complete service routes, train types, frequencies,
formations and fleet cycles. GeoJSON preserves infrastructure `track_type` and
adds `infrastructure_type`, `track_count`, `service_ids`, `service_types` and a
top-level service list. Corridor `frequency` is aggregate movement frequency;
legacy `train_units` is the maximum formation on the corridor, not a new decision.
Map symbols denote infrastructure; violet dots show HSR trains on CR track and
orange dots show CR trains on HSR track. Analysis preserves the actual service plan.

Geometry cache schema remains version 3 because cached alignments are unchanged.
Rolling-stock/service settings do not enter geometry fingerprints or geometry
files. Splitting a legacy combined `tracks` config changes its fingerprint once;
new cache paths may therefore be created. Evaluations and historical objectives
are never reused as geometry. Standalone saved analysis refuses historical
snapshots unless `--reevaluate-legacy` is explicitly supplied; this recomputes
analysis under the new model rather than claiming old metrics are unchanged.

## Reproduce and validate

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_hybrid_services.py'
.venv/bin/python -m unittest discover -s tests -p 'test_capacity_repair.py'
.venv/bin/python -m unittest discover -s tests -p 'test_track_layouts.py'
.venv/bin/python -m unittest discover -s tests -p 'test_railgen*.py'
.venv/bin/python -m examples.hybrid_services --output runs/hybrid-example
```

The self-contained example constructs both sections from scratch, runs an HSR
service across a high-speed trunk and a conventional extension alongside a CR
service, and writes maps, GeoJSON, service records, reconciled analysis and costs.
It requires no downloaded input data. All values are illustrative, not calibrated.

Local validation measurements for this implementation: the four-city/five-edge
CLI benchmark took about 9 ms median (20 sampled designs, excluding cache load).
A dense 100-city/254-built-edge evaluation took about 94 ms median across three
runs, with 244 services in the selected plan. These are illustrative local
measurements, not runtime guarantees or a claim that the dense design is feasible.
The four-city/four-iteration incremental CLI example completed with all cities
connected and produced the live and Pareto artifacts.
