# Fast railway generator

The current pipeline lives in `rail_gen/`. It uses cached geography, candidate
railway alignments, explicit train services, annual passenger demand and lifecycle
costs. It is a strategic scenario model, not a calibrated traffic forecast.

## Run

Run commands from the repository root:

```sh
.venv/bin/python -m pip install -r requirements-railgen.txt
.venv/bin/python -m rail_gen run --profile italy --iterations 200 \
  --output runs/italy-live --snapshot-every 10
```

Add `--profit` to optimize profit during the final 20% of iterations. The equivalent
entry point is `optimize.py --solver pareto`. Optional population algorithms are
`--algorithm nsga2` and `--algorithm nsga3`; they do not support `--profit`.
Use an empty output directory. `--overwrite` deletes existing contents.

## Configuration

[rail_gen/config.json](../rail_gen/config.json) contains defaults and named profiles.
External `--config` overrides defaults, the selected profile applies next, and CLI
arguments apply last. `run_config.json` records effective settings for each run.
The root [config.json](../config.json) belongs to the separate legacy optimizer.

Input paths select city records, population cells, boundaries and terrain. `bbox`,
`countries`, `minimum_population` and `max_cities` control city selection.
Candidate edges combine nearest neighbors, Delaunay links and forced gravity-ranked
pairs; nearby intermediate cities can split candidate links. Geography and
alignment caches live under `runs/railgen-cache/` by default.

## Population and demand

H3 density is smoothed with weight 0.4 on the center and 0.1 on each adjacent cell;
missing neighbors contribute zero. Station radius in km is
`center_coefficient * sqrt(local_density)`. Larger retained cities suppress smaller
city centers inside their radius. Population cells are assigned exclusively to
the nearest retained city, provided they fall inside its radius.

Effective population is captured population times `prep.uptake`:

```text
E_i = P_i * kappa_i = captured_population_i * uptake
A_i = population_weighted_mean_access_distance_i / access_speed_kmh
M_ij = k * E_i^a * E_j^a / max(distance_ij, minimum_distance_km)^beta
```

OD distance is great-circle distance. Self-trips and pairs beyond
`maximum_distance_km` have zero demand. Catchment and routing distances use the
projected grid. `kappa` can exceed one because metropolitan catchments can contain
more people than the city-proper population.

The default `reference_elasticity` mode uses:

```text
T0 = A_i + A_j + distance * reference_detour_ratio / reference_speed_kmh
     + reference_wait_hours + reference_transfer_hours
Q0 = 2 * M_ij
T = A_i + A_j + onboard_time + waiting_time + transfer_count * transfer_hours
r = T / T0
potential_trips = Q0 * r^total_elasticity
modal_shift_change = Q0 * (r^shift_elasticity - 1)
induced_change = potential_trips - Q0 - modal_shift_change
```

Trips are annual and include both directions. The reference is synthetic and fixed
within an evaluator. It has no logistic uptake or reference-fare penalty.
Disconnected pairs have zero potential trips. Slower journeys give signed demand
reductions. The shift/induced breakdown is an assumed elasticity decomposition,
not explicit simulation of other modes or trip purposes.

`t0_hours`, `sigma_hours` and `reference_yield_eur_pkm` do not affect reference
elasticity demand. Legacy `rail_share` mode retains the logistic function
`1 / (1 + exp((generalized_hours - t0_hours) / sigma_hours))`.
Actual fares affect route choice and revenue; reference-elasticity demand responds
to fare-free journey time. Fare is:

```text
fare = distance * yield * (max(distance, minimum_distance_km) / fare_reference_km)^(-fare_taper)
```

Yield is weighted by ridden distance for mixed-stock journeys. Changing `a` changes
absolute demand dramatically because populations are measured in people; refit
`k` whenever population exponent, catchments, distance exponent or demand model
changes. Fit annual city-pair observations with `calibrate --backtest ... --fit k a`.
The backtest CSV schema is defined in `rail_gen/calibrate.py`; reserve independent
corridors for validation. Parameters without observations remain assumptions.

## Services and capacity

See [the hybrid services guide](HYBRID_SERVICES.md) for full routing, formation,
track layout and cost details. Each boarding adds half the service headway to
journey time. Segment time includes length divided by the minimum track/train
speed plus dwell time. Through services avoid intermediate boarding waits.

Frequency sizing targets `target_load_factor` over a bounded number of assignment
iterations. Annual directional service capacity is:

```text
capacity = seats * formation * frequency * hours_per_day * days_per_year
           * maximum_load_factor / peak_factor
```

Capacity is shared across passengers using each service segment. Carried OD trips
are scaled by the most constrained segment on their assigned journey. There is no
queue simulation or overflow rerouting. Spill is potential minus carried trips;
spill fraction is spill divided by potential trips, or zero for zero potential.

With automatic repair, overloaded segments can widen from the configured layout
toward four tracks when added capacity carries more passengers. Additional tracks,
trains and operations are charged. Explicit track counts and service plans remain
authoritative. Four tracks is the current upper bound, not unlimited capacity.

## Incremental search

The search starts empty and edits the last accepted network. Growth favors carried
ridership gains relative to annual net cost and can add coverage. When an incumbent
has excessive spill, capacity-expansion selection favors more carried passengers
without losing coverage or worsening component count. Integration prioritizes
joining components and serving missing candidate cities. Connected networks can
enter cost consolidation after `growth_fraction` of the iteration budget.

Spill-only infeasible candidates may become the live network, allowing diagnostic
outputs even under excessive demand. Budget, subsidy, component-share and shared
track-capacity constraints remain hard. `feasible` still includes the spill limit;
only fully feasible alternatives enter the Pareto archive. Reaching the iteration
limit with excessive spill gives `capacity_shortfall`.

With `--profit`, the transition happens after `ceil(0.8 * iterations)` updates:
for 200 iterations, profit starts at 161. It saves `before_profit` artifacts.
Profit selection requires a net annual improvement and permits service reductions.
The bounded search can stop at neighborhood exhaustion, incomplete coverage or
profit saturation. More iterations do not guarantee a connected or optimal network.

## Outputs and interpretation

- `latest.json`, `latest.geojson`, `latest.png`: the current accepted network.
- `history.jsonl`, `progress.png`, `progress.gif`: evolution over iterations.
- `analysis.json`, `analysis.png`: station/corridor counts and OD demand components.
- `latest_revenue_cost.png`, `cost_breakdown.log`, `report.md`: financial analysis.
- `run_config.json`, manifests and `checkpoint.npz`: settings and saved state.
- `pareto/`: evaluated alternatives, rather than successive live networks.

Corridor satisfaction is carried divided by assigned potential demand. It measures
capacity fulfillment, not frequency quality: poor service can suppress potential
demand while retaining 100% fulfillment. Network/station denominators use candidate
potential for connected pairs and reference trips for unreachable pairs. Station
endpoint counts count each trip at both ends. Corridor counts include transfer legs.
Map captions referring to a fixed market are legacy wording in elasticity mode.

The inner service-plan choice prioritizes constraint violation, then net annual
cost, then passengers. It can favor sparse edge shuttles over through services.
These limitations should be considered when interpreting green maps, tuning demand
or comparing designs. Profit allocations are not independent route-closure savings.

## Other documentation

- [README and quick start](../README.md)
- [Hybrid services and migration](HYBRID_SERVICES.md)
- [Performance measurements](RAILGEN_PERFORMANCE.md)
- [Historical capacity experiment](experiments/capacity-repair.md)
- [Legacy optimizer and terrain tools](LEGACY_OPTIMIZER.md)
