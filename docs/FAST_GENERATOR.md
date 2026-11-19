# Fast railway generator

The current pipeline lives in `rail_gen/`. It uses cached geography, candidate
railway alignments, explicit train services, annual passenger demand and lifecycle
costs. It is a strategic scenario model, not a calibrated traffic forecast.

## Run

Run commands from the repository root:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m rail_gen run --profile italy --iterations 200 \
  --output runs/italy-live --snapshot-every 10
```

Optional population
algorithms are `--algorithm nsga2` and `--algorithm nsga3`. Incremental search runs
one continuous search, with 150 sampled proposals per update by default
(`--population 150`). There is no `--profit` or `--restarts` flag.
Use an empty output directory. `--overwrite` deletes existing contents.

## Configuration

[config/config.json](../config/config.json) contains shared defaults. [profile_locations.json](../config/profile_locations.json) indexes the separate files in `config/profiles/`.
See [Country profiles](COUNTRY_PROFILES.md) for country names, scope, and data limitations.
The `netherlands` profile selects only the Netherlands; Belgium uses `belgium`.
External `--config` overrides defaults, the selected profile applies next, and CLI
arguments apply last. `run_config.json` records effective settings for each run.

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

Capacity is shared across passengers using each service segment. Each round
scales carried OD trips by the most constrained segment, then tries
residual market fractions on alternative journeys around full segments. Up to
eight rounds are evaluated, each with its own journey-time elasticity. There is
no queue simulation or equilibrium guarantee. Spill is potential minus carried trips;
spill fraction is spill divided by potential trips, or zero for zero potential.

With automatic repair, overloaded segments can widen from the configured layout
toward four tracks when added capacity carries more passengers. Additional tracks,
trains and operations are charged. Explicit track counts and service plans remain
authoritative. Four tracks is the current upper bound, not unlimited capacity.

## Incremental search

The search starts empty and edits the last accepted network. It also proposes
complete construction paths for up to twelve markets ranked by unmet reference
demand per iteration. All missing sections are evaluated together. Expansions
inherit purchased capacity and retain the accepted timetable as an alternative;
new services reserve incumbent frequencies and formations. Service-plan selection
retains its original internal service-plan rules; outer proposal selection uses
the weighted objective. Results use model version `network-capacity-v3`. Growth favors carried
ridership gains relative to added annual cost and can add coverage. When an incumbent
has excessive spill, capacity-expansion selection favors more carried passengers
without losing coverage or worsening component count. Integration prioritizes
joining components and serving missing candidate cities. Connected networks can
enter cost consolidation after `growth_fraction` of the iteration budget.

Spill-only infeasible candidates may become the live network, allowing diagnostic
outputs even under excessive demand. Budget, subsidy, component-share and shared
track-capacity constraints remain hard. `feasible` still includes the spill limit;
only fully feasible alternatives enter the Pareto archive. Reaching the iteration
limit with excessive spill gives `capacity_shortfall`.

### Adaptive objective weights

The original proposal generation, deterministic connection packages,
capacity expansion, integration, fixed service protections, caching, batching and
stopping rules are retained. Candidate ranking uses the weighted objective:

`F = w1*(annual_cost-revenue)/finance_scale_eur - w2*passengers/reference_trips - w3*population_served/total_population`.

| Demand satisfaction of incumbent | Cost | Passengers | Population |
| --- | ---: | ---: | ---: |
| Below 30% | 0.05 | 0.70 | 0.25 |
| 30% to below 60% | 0.20 | 0.55 | 0.25 |
| At least 60% | 0.70 | 0.20 | 0.10 |

Normalization scales are fixed for the run: `search.finance_scale_eur` defaults to
EUR 100 million; reference trips and total candidate population are floored at one.
Weights come from the incumbent and stay fixed while comparing all proposals and
batches in that iteration. Internal demand assignment and service-plan rules remain unchanged.
Growth requires an improving weighted score and retains its service protections.
Capacity expansion retains its passenger and topology requirements and ranks by F.
Integration still prioritizes connecting components and covering cities, then ranks
by F: it can accept a necessary connection without immediate objective improvement.
Consolidation requires a lower F while retaining its original fixed service baseline.
The satisfaction thresholds change weights, not the original structural search phases.

Edge sampling also follows the incumbent's satisfaction phase. Its demand-per-capital
weight is multiplied by `1 + coefficient * served_endpoints`, where
`search.served_station_bias` configures the coefficients:

| Phase | Satisfaction | Coefficient | Neither served | One served | Both served |
| --- | --- | ---: | ---: | ---: | ---: |
| expansion | Below 30% | 0.5 | 1 | 1.5 | 2 |
| development | 30% to below 60% | 2 | 1 | 3 | 5 |
| consolidation | At least 60% | 4 | 1 | 5 | 9 |

Coefficients must be finite and nonnegative; zero removes the station preference.
The same sampling schedule applies during capacity expansion and integration.
Deterministic demand packages and acceptance rules are unchanged, including the
capacity selector's restriction on introducing disconnected components.
Running processes retain their loaded configuration and code until restarted.

Each invocation starts one network. It stops at the iteration/time limit, interruption,
or the original neighborhood-exhaustion conditions. There is no ten-unsuccessful-iteration
cutoff, extra 60% acceptance floor, recovery controller or automatic seed restarts.
History records `weight_phase`, `objective_weights`, `objective_scales` and
`weighted_objective` alongside the original search phase and metrics.

## Outputs and interpretation

- `latest.json`, `latest.geojson`, `latest.png`: the current accepted network.
- `history.jsonl`, `progress.png`, `progress.mp4`: evolution over iterations.
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

- [README and quick start](../readme.md)
- [Hybrid services and migration](HYBRID_SERVICES.md)
- [Performance measurements](RAILGEN_PERFORMANCE.md)
- [Historical capacity experiment](experiments/capacity-repair.md)

### Optional simulated annealing

Enable `search.annealing` in a JSON configuration override:

```json
{"search": {"annealing": true, "annealing_initial_temperature": 0.1,
            "annealing_final_temperature": 0.001}}
```

For example, pass this file with `python -m rail_gen run --config annealing.json`.
Annealing is opt-in and supported only by incremental search. During the first
`ceil(generations * growth_fraction)` iterations it replaces greedy acceptance
with one Metropolis decision on a randomly assembled, freshly evaluated batch
of disjoint edits. Individual edits need not improve the network. Existing
sampled evaluations are reused; a novel batch adds at most one evaluation per
iteration. Rejected edits can be tried again on later iterations.

The energy to minimize is the normalized weighted objective above, using the
incumbent's demand-satisfaction weights for both states in each comparison.
The optional journey-time archive objective does not enter the energy. Temperature
cools geometrically between the configured positive endpoints. Improvements are
accepted; deterioration by delta is accepted with probability `exp(-delta / temperature)`.
This is a heuristic annealing search, not an exact Metropolis–Hastings sampler.

Budget, subsidy, component-share and infrastructure-capacity checks remain hard;
the existing spill exception still applies to live states. Only fully feasible
results enter the Pareto archive. History records temperature, current and
proposed energies, and acceptance under `annealing`. The live network may get
worse; consult the archive for preserved alternatives. Integration/consolidation
rules resume after exploration. Existing internal service
plan selection remains unchanged. Annealing cannot add edges outside the prepared
candidate graph, and does not guarantee better results within a fixed run budget.

## Continuing an incremental run

```bash
.venv/bin/python -m rail_gen run --profile europe \
  --resume runs/europe/checkpoint.npz --iterations 1000
```

Resume reads the checkpoint directory's `run_config.json` by default and writes
back to that directory, preserving history and appending animation frames. Use
`--output` with a new empty directory to branch the run. `--iterations` specifies
additional updates; the time limit applies to this invocation. Iteration numbers
continue from the checkpoint. Resume and overwrite are mutually exclusive.

The saved genome, independent track counts and service decisions are restored.
Model, geography and configuration must match; only iterations, time limit,
workers and reporting settings may change. New checkpoints also restore the RNG
and consolidation/profit baselines. Historical checkpoints without those fields
start a fresh random stream and phase schedule from the saved network. Each
invocation uses its own growth/annealing schedule and rebuilds proposal caches
and the Pareto archive, so continuation is not a bit-for-bit uninterrupted run.
Population algorithms currently reject resume.
