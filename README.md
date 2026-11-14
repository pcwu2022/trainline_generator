# Fast Pareto railway generator

The `rail_gen` pipeline uses cached geography and a service-based annual-demand/
finance evaluator. HSR services can continue over newly built conventional track;
infrastructure and rolling stock are independent. See the
[hybrid service model and migration guide](docs/HYBRID_SERVICES.md).
Its default incremental search starts empty and edits the
previous accepted network at every iteration, then reduces net annual cost while
retaining service. Each update ranks useful proposals and combines up to ten
with disjoint edge edits, then evaluates the combined network under the same
phase criteria. Failed combinations retry smaller batches and fall back to the
best individual proposal. Use `--batch-accept-limit 5` to cap batches at five,
or `--batch-accept-limit 1` for individual updates (JSON: `search.batch_accept_limit`).
Batch evaluations count toward reported evaluation totals and feasible results
enter the Pareto archive. Larger updates may reduce the iterations needed;
wall-clock speed depends on evaluation cost and proposal compatibility.
It also saves a Pareto archive of evaluated alternatives.

Spill guides capacity expansion in incremental search instead of excluding a
connection. Evaluation first sizes frequencies and train formations; overloaded
segments may then widen from two to four tracks, with the additional capital and
operating costs included. Explicit track counts remain fixed. During growth,
an overloaded incumbent favors edits that carry more passengers without losing
coverage or connectivity. Spill-only infeasible networks may be retained as live
diagnostic results; budget, subsidy, component-share and infrastructure-capacity
constraints remain hard. `feasible` still reports the full constraint result,
and only fully feasible alternatives enter the Pareto archive. Current layouts
support at most four tracks, so extreme demand can leave unresolved spill.
Iteration logs and saved metrics show spill, potential trips, carried trips and
lost trips for calibration. A run reaching its iteration limit with excessive
spill reports `capacity_shortfall`.

Growth also evaluates complete construction paths for the twelve largest
underserved city-pair markets each iteration. These demand-ranked packages are
considered deterministically alongside sampled edge edits, with no city-specific
rules. Expansion candidates inherit paid track capacity and retain the accepted
service plan as an alternative. Added services reserve incumbent frequencies and
formations. Proposal selection compares the normalized financial, ridership and population
objectives using satisfaction-dependent weights. Passenger assignment tries up to eight shortest
viable paths around saturated service segments, recalculating elastic demand for
each alternative journey before recording residual spill.

This changes evaluation semantics to `network-operating-cost-v4`. Historical snapshots
must be explicitly re-evaluated for comparison; the currently running process
continues using the code it loaded at startup.

```sh
.venv/bin/python -m pip install -r requirements-railgen.txt
.venv/bin/python optimize.py --solver pareto --profile france \
  --output runs/france-live --iterations 200 --snapshot-every 5 --gif-frame-ms 500
```

Incremental search uses the original 150-proposal neighborhood, complete demand
connection packages, capacity repair, integration and validated batching. Candidate
ranking now uses normalized `w1*(annual_cost-revenue) - w2*passengers - w3*population`.
Weights are `(0.05, 0.70, 0.25)` below 30% demand satisfaction,
`(0.20, 0.55, 0.25)` from 30% to below 60%, and `(0.70, 0.20, 0.10)`
from 60% onward. The original service and topology protections still apply.
The `--profit` and `--restarts` options are removed: each invocation grows one network,
with the original iteration/exhaustion stopping rules and no ten-failure cutoff.

Watch `latest.png` for the evolving network, `progress.png` for its metrics, and
`progress.gif` for the animation. Maps use the original population-density
background, green demand-satisfaction colors and frequency-scaled widths.
Infrastructure symbols distinguish CR and HSR track; dotted overlays identify
train types operating across infrastructure boundaries. `latest.json`, `latest.geojson` and
`cost_breakdown.log` describe that same accepted state. Comparison designs and
maps are grouped under `pareto/`; they are not successive iterations.

The existing `python -m rail_gen run ...` command uses the same incremental
default. Select `--algorithm nsga2` / `--algorithm nsga3`, or
`optimize.py --solver nsga2` / `--solver nsga3`, for population evolution.
Parameters for the fast pipeline are in `rail_gen/config.json`.

Construction keeps the double-track baselines of EUR 6M/km (CR) and EUR 20M/km
(HSR). Ground-slope anchors 0%, 1%, 2%, 5%, 10%, 20% use multipliers
1, 1.2, 1.5, 2, 3, 5, linearly interpolated and capped at 5. Land costs add
`base * 0.35 * max(0, log10((100 + density) / 5100))` in million EUR/km.
Water replaces the land rate: unclassified inland water defaults to EUR 40M/km;
sea defaults to EUR 120M/km. Sea routing is enabled by default.
Configured `alignment.water_crossings` accept an optional `water_type`:
`small_river` (20), `large_river` (40), `wide_river_bay` (60), `sea` (120), or
`difficult_sea` (180), in million EUR/km for either track technology. For example:

```json
{"id": "bay", "geometry": [[2.0, 46.0], [2.1, 46.0]], "water_type": "wide_river_bay"}
```

Crossing portals must be on land. Explicit crossing assets price water segments
by their length and land approaches at the local land rate, without exponential
water growth. Crossing geometry is split at raster boundaries; water extent still
comes from the input mask, so subcell river widths require finer water data.
Without explicit assets, routing can traverse water cells directly. With explicit
assets, water is traversed through those portal links.
Construction charges actual unsmoothed segment kilometres within each cell,
including station offsets. Shared cells/assets pay the highest selected charge
once; track multipliers remain 0.5/1/2. Alignment caches are rebuilt automatically
for this pricing version. Existing run snapshots and running processes retain
their previous costs until re-evaluated/restarted.


Railway operating costs use annual train-km times 19.50 (CR trains) or 31.00
(HSR trains). Shared physical track maintenance uses 105,000/145,000 per
track-km/year plus 1.55/2.90 per train-km for CR/HSR infrastructure, respectively.
Defaults use 365 operating days; daily departures equal hourly frequency times
operating hours. Train costs follow rolling stock; maintenance follows physical
infrastructure, including HSR trains running on CR tracks. Formation size does
not multiply train movement costs. Annualized capital and station operating
expenses remain part of the total financial objective. See the
[operating-cost definitions](docs/HYBRID_SERVICES.md#fleet-and-annual-finance).

Demand defaults to `demand.model = "reference_elasticity"` for **all** eligible city
pairs, including international journeys, with no country distinction. Routing
recomputes complete service journeys, including connecting services, after each
candidate edit and frequency update. Demand uses fare-free journey hours (access,
waiting, transfers and onboard time); fares still affect routing and revenue.
There is no additional network or induced-demand multiplier.

Each evaluator fixes a synthetic conventional-rail reference:
`G0 = access_a + access_b + distance_km * reference_detour_ratio / reference_speed_kmh + reference_wait_hours + reference_transfer_hours`.
Defaults assume 100 km/h, a 1.2 detour ratio, 0.5 hours waiting and zero additional
transfer hours. Reference two-way rail trips are `Q0 = 2 * gravity_market`, without
logistic uptake or a reference-fare penalty. The baseline scale is absorbed into
`demand.k`. `t0_hours`, `sigma_hours`, and `reference_yield_eur_pkm` do not affect
reference-elasticity demand; the first two remain available for legacy `rail_share`
mode. These are configurable prototype assumptions,
not observed existing-network trips. They never depend on the current candidate
or the sequence of accepted changes.

For `r = G / G0`, total potential trips are `Q0 * r ** total_elasticity` (default
−1.4); modal shift change is `Q0 * (r ** shift_elasticity - 1)` (default −0.9);
induced change is the remainder. Slower journeys produce signed reductions;
unreachable pairs have zero potential trips, a modal shift loss of `Q0`, and zero
induced change. Frequency optimization uses this potential demand, then capacity
limits determine carried trips and spill. To test total elasticities −1.0 and
−2.0, override `total_elasticity`, keeping `shift_elasticity` at −0.9.

`analysis.json` includes per-pair reference hours/trips, candidate journey hours,
potential/carried trips and signed demand changes. Metrics include raw and
capacity-scaled demand components. Their reference + modal shift + induced sums
reconcile to potential and carried trips respectively. Demand fulfillment replaces
fixed-market satisfaction: its denominator uses candidate potential on connected
pairs and reference rail trips on unreachable pairs. Legacy metric keys
`eligible_need_trips` and `need_satisfaction_ratio` remain aliases for this
measurement; corridor maps measure carried / assigned potential demand.
Set `demand.model = "rail_share"` to reproduce the previous demand calculation.
Old calibrated parameters should be recalibrated for the new model.

Use a new or empty output directory, or add `--overwrite` to delete its previous
contents and start a fresh run. Both fast-generator entry points accept this flag.
The supplied city and relief data support uncalibrated development runs;
sourced metro populations and corridor backtests are required before
interpreting the results. See [configuration, calibration, outputs and
limitations](docs/FAST_GENERATOR.md).

The earlier optimizer is documented in [the legacy guide](docs/LEGACY_OPTIMIZER.md)
and remains available with
`optimize.py --solver annealing`, `heuristic`, or `milp`.

## Documentation

- [Fast generator: configuration, formulas, search and outputs](docs/FAST_GENERATOR.md)
- [Explicit water crossings and Channel scenario](docs/WATER_CROSSINGS.md)
- [Hybrid services, capacity and financial accounting](docs/HYBRID_SERVICES.md)
- [Performance measurements](docs/RAILGEN_PERFORMANCE.md)
- [Historical capacity-repair experiment](docs/experiments/capacity-repair.md)
- [Legacy optimizer, terrain tools and earlier run notes](docs/LEGACY_OPTIMIZER.md)
- [Legacy algorithm](ALGORITHM.md) and [legacy formulation](docs/FORMULATION.md)
