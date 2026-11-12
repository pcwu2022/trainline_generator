# Fast Pareto railway generator

The `rail_gen` pipeline uses cached geography and a service-based annual-demand/
finance evaluator. HSR services can continue over newly built conventional track;
infrastructure and rolling stock are independent. See the
[hybrid service model and migration guide](docs/HYBRID_SERVICES.md).
Its default incremental search starts empty and edits the
previous accepted network at every iteration, then reduces net annual cost while
retaining service. It also saves a Pareto archive of evaluated alternatives.

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
formations. Growth compares passenger gains per additional annual cost; profit
mode compares annual net benefit. Passenger assignment tries up to eight shortest
viable paths around saturated service segments, recalculating elastic demand for
each alternative journey before recording residual spill.

This changes evaluation semantics to `network-capacity-v3`. Historical snapshots
must be explicitly re-evaluated for comparison; the currently running process
continues using the code it loaded at startup.

```sh
.venv/bin/python -m pip install -r requirements-railgen.txt
.venv/bin/python optimize.py --solver pareto --profile france \
  --output runs/france-live --iterations 200 --snapshot-every 5 --gif-frame-ms 500
```

Add `--profit` to incremental runs to switch after `ceil(iterations * 0.8)` updates,
reserving the final 20% of iterations (rounded down) for profit optimization.
The switch saves `before_profit.png`, `before_profit.json`, and
`before_profit.geojson`. Profit mode gives normalized annual revenue minus total
annual cost 100 times the weight of ridership and population rewards; it allows
service reductions and accepts only profit-improving moves. It continues until
the iteration limit or `profit_saturated` (the entire local neighborhood has
been checked without an acceptable improvement). Time limits and interrupts
still apply. Population algorithms do not support this flag.

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
