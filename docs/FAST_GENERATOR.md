# Fast multi-objective railway generator

`rail_gen/` implements the two-layer architecture in the supplied
[v0.1 specification](FAST_GENERATOR_SPEC.txt). Geography is computed once;
network search uses fixed abstract edges and never calls routing or HiGHS.
The default search now follows one persistent network from iteration to iteration.
NSGA-II/III remain selectable population alternatives. The earlier optimizer
remains available through `optimize.py --solver annealing`.

## Run

From the repository root, using Python 3.9 or newer:

```sh
.venv/bin/python -m pip install -r requirements-railgen.txt
.venv/bin/python -m rail_gen run --profile smoke --output runs/railgen-smoke
.venv/bin/python -m rail_gen run --profile france --output runs/railgen-france
.venv/bin/python -m rail_gen run --profile europe --output runs/railgen-europe
```

For the original entry point and live-output flags:

```sh
.venv/bin/python optimize.py --solver pareto --profile france \
  --output runs/france-live --iterations 200 --snapshot-every 5 --gif-frame-ms 500
```

Both commands run the same fast incremental algorithm. `--solver incremental`
is an alias for `--solver pareto`. `--algorithm nsga2` / `nsga3` on the module
command, or `--solver nsga2` / `nsga3` on `optimize.py`, select population search.
`optimize.py` without a fast solver flag retains its existing default.

Choose a new or empty output directory, or pass `--overwrite` to delete the
previous output and start a fresh run, as with the older optimizer. This flag
works with both entry points. It rejects symlinks, the working directory and
its ancestors.
`--iterations` and `--generations` are aliases for the number of network updates
(incremental) or population generations (NSGA). `--population` sets the maximum
neighbours evaluated per incremental update, or the NSGA population size.
`--seed` and `--workers` override search settings. `--time-limit SECONDS` caps
search time including live rendering, after geography preparation. Ctrl+C
finishes the current update and saves its accepted network and evaluated front.
A second Ctrl+C interrupts immediately.
Iteration logs and saved metrics include `need_satisfaction_ratio`: completed
two-way annual trips divided by fixed two-way annual gravity demand across all
candidate OD pairs, including unserved pairs. The denominator includes fixed
station capture and distance screening, before journey-time rail share and
capacity spill. This is distinct from the fraction of cities with stations.
The terminal summary prints the search status; `coverage_incomplete` can occur
before the iteration limit if all local coverage/connectivity improvements are
rejected, for example by the maximum spill constraint.
`--workers` uses a shared-memory thread pool for independent evaluations; SciPy
and NumPy kernels can release the GIL. One worker is the default: parallel
speedups depend on network size and hardware.

```sh
.venv/bin/python -m rail_gen precompute --profile france
.venv/bin/python -m rail_gen benchmark --profile france --evaluations 200
.venv/bin/python -m rail_gen run --profile smoke --sensitivity --output runs/railgen-sensitivity
.venv/bin/python -m unittest discover -s tests -p 'test_railgen*.py'
```

`rail_gen/config.json` is the single parameter source for this pipeline, separate
from the legacy root configuration. Named profiles cover France, Italy,
Netherlands/Belgium, Europe and a small smoke scenario. External `--config PATH`
can contain a partial override object, a complete `run_config.json`, or
`{"defaults": {...}, "profiles": {...}}`. Overrides merge recursively; precedence
is defaults, external defaults, selected profile, CLI. Unknown parameters,
non-finite numbers, invalid frequencies and invalid load factors are rejected.
Paths resolve relative to the working directory, as in the existing optimizer.

## Inputs and calibration status

The supplied `data/cities50000.json` contains city populations and includes
suburbs/administrative districts. **It is not a verified metropolitan table.**
It is usable to exercise the pipeline, but the resulting networks and finances
are uncalibrated scenarios. Replace `cities_file` with a curated CSV or JSON
array with columns `id` (optional), `name`, `lon`, `lat`, `population` (or `pop`),
and `country` (ISO alpha-2, required when country filtering is enabled). Keep one
non-overlapping metropolitan record per intended node, then set
`metro_populations_verified=true` after verifying the input yourself.

Population input is the existing GeoJSON of points/H3 cells. `population` and
`h3Index` provide population and area; other point/polygon cells must supply
`area_km2`. Polygons use centroids. Each sample belongs exclusively to its nearest
city, preventing overlapping catchments from counting the same residents twice.
Captured population within 15 km is capped at metro population and multiplied
by uptake. Access time is the population-weighted catchment distance divided
by access speed. These fixed quantities are the only density-dependent inputs
to demand. There is no city coverage floor.

The bundled RGB relief map supports approximate terrain multipliers and inland
water premiums, **not engineering grades**. Set `terrain_kind="dem"` and point
`terrain_file` to a single-band elevation GeoTIFF in metres for grade limits and
slope premiums. The existing `fetch_terrain.py` and `generate_terrain.py` produce
suitable metric DEMs. Coarse-grid gradients underestimate engineering grades;
this remains strategic routing. `terrain_kind="flat"` is an explicit synthetic
scenario. Real terrain needs `boundaries_file` or `land_class_file` to block sea.

Optional `land_class_file` is a single-band categorical raster: 0 land, 1 inland
water, 2 sea, 3 excluded/protected. Missing coverage is blocked. Optional
`exclusions_file` is a polygon GeoJSON. Country boundaries define land across
all countries; the `countries` filter selects city nodes independently. Small
waterways and protected areas absent from these layers are not inferred.

No ridership observations are fabricated or downloaded. Copy the CSV header in
[backtest_template.csv](../rail_gen/data/backtest_template.csv), then supply
consistent, sourced observations. `observed_annual_passengers` must count both
travel directions on the city-pair market, matching model reporting. Distinguish
corridor totals from city-pair totals when collecting data. Use explicit `train`
and `test` splits, about one third held out. `anchor` accepts `true`, `yes`, or
`1`; `class` accepts `strong` or `thin`; `track_type` is CR or HSR.

```sh
.venv/bin/python -m rail_gen calibrate --profile france \
  --backtest my-corridors.csv --output runs/france-backtest.json
.venv/bin/python -m rail_gen run --profile france \
  --backtest my-corridors.csv --output runs/france-calibrated
```

Calibration fits k against log ridership using **training observations only**.
Optional `--fit k a beta t0_hours` fits more parameters when enough training rows
exist. Fixed actual line length, speed and frequency are used, including access,
wait, fare and the same dwell convention as the network evaluator. The output
records fitted parameters, observation sources, input SHA-256 and every check.

At least 10 sourced corridors are required for a complete backtest.
Checks cover held-out Spearman ranking (at least three held-out corridors),
anchor magnitude, strong/thin ratio, frequency, fleet, train operating recovery,
and capital recovery arithmetic. A missing check is `not_assessed`, never a
pass. The calibration command returns 2 for a failed/incomplete report. A run
without observations is explicitly `not_calibrated`. Route loss diagnostics are
separate from demand backtesting. No results should be interpreted as validated
forecasts until the data and backtests are independently checked.

## Implementation

- `prep.py`: fixed city-centre nodes; exclusive capture and access; k-nearest
  neighbours + Delaunay + forced gravity pairs; near-collinear via pruning with
  replacement links retained. Forced pairs can exceed the ordinary distance cap.
- `alignment.py`: EPSG:3035 cost grid; distinct CR/HSR base prices and grade
  limits; relief/DEM, density, water and exclusions; one windowed MCP sweep per
  source/type resolves all its candidate destinations. Forced long pairs enlarge
  their source window. Station cells are fixed; blocked station cells cannot
  silently move across water. Douglas–Peucker/Chaikin display smoothing is
  accepted only inside the priced footprint.
- `demand.py`: annual **one-way** gravity market with logistic rail share and a
  fare/VOT term. Only upper-triangle markets are stored; evaluator totals multiply
  by two for symmetric two-way trips. Unconnected pairs carry zero passengers.
- `evaluator.py`: SciPy shortest paths, sparse path assignment, branch-turn
  transfer proxy, at most three load/frequency updates, a final demand calculation
  at the returned frequency, and one simultaneous bottleneck spill pass. Spill
  capacity includes peak factor and maximum load factor at the actual operating
  frequency. Fleet includes round-trip service time, turnaround, integer
  trainsets and spares. Fare uses great-circle OD distance and the path's
  length-weighted yield, so detours cannot inflate revenue per trip.
- `incremental.py`: the default persistent-network search. Start empty; make
  add/remove/CR–HSR edits or join components by a cheapest candidate path, always
  from the previous accepted genome. Growth prefers ridership gained per added
  net annual euro while preserving current ridership and captured population.
  It can also accept net cost savings that preserve service. When growth stalls,
  or after `growth_fraction` (default 70% of iterations), integration prioritizes
  serving missing candidate cities and joining components. Integration can accept
  higher costs or lower ridership to improve topology, subject to financial and
  operational constraints. Once every candidate city is served in one component
  and the growth fraction has elapsed, freeze the baseline and consolidate: lower
  net annual cost with at least `retained_ridership` (98%) of baseline passengers
  and captured population, and at most `journey_time_allowance` (5%) more mean
  generalized time. These are incumbent selection rules; the separate Pareto
  archive retains all feasible non-dominated evaluated alternatives. There is no
  additional coverage constraint on the Pareto archive or population algorithms.
  Consolidation preserves all candidate cities and connectivity. Rejected proposals
  retain the preceding state. Incomplete coverage at the iteration limit or after
  exhausting available moves is reported as `coverage_incomplete`, not successful
  completion; constraints, zero-load repair or disconnected candidate geography
  may prevent completion. Time limits and interruption retain their own statuses.
  Coverage refers to the selected city candidates, not every settlement in France.
  No random restart is performed.
- `search.py`: optional integer NSGA-II, or NSGA-III for four objectives;
  uniform crossover, flips, cheapest component joining, low-value removal,
  busy-edge upgrade and low-load downgrade. Greedy marginal-net/capital growth
  supplies a fraction of seeds at multiple stopping sizes; the rest are sparse
  random/demand-biased states. The empty design is included. Zero-load repair
  writes the genome back into the pymoo population. Feasible non-dominated
  designs are archived across generations, not just from the final population.
- `calibrate.py`: fixed-line prediction, log-error fitting and held-out checks.
- `live.py`: iteration-triggered current map, metrics, GeoJSON, cost log, six
  progress panels, and a streaming GIF using the original encoder. Images and
  JSON are atomically replaced; individual frame files are not accumulated.
- `report.py`: full-precision recomputation/deduplication, k-medoids selection
  with financial/subsidy/knee anchors, maps, finance reconciliation, loss flags,
  and optional six-run sensitivity analysis.

The objectives are net annual cost, negative annual carried trips and negative
captured population. Optional `include_time_objective` adds trip-weighted mean
generalized time. An empty network has an explicit worst time score. The knee
representative is the normalized point nearest the ideal objective vector.
Budgets, annual subsidy, spill and optional largest-component share use pymoo's
constraint interface rather than objective penalties. Connectivity is measured
among active nodes, so opening a disconnected singleton is not an escape.

Capital uses separate CR/HSR edge-by-tile sparse matrices. The union is paid
once; HSR overrides CR on a shared cell. Annual costs use capital recovery
factors, track maintenance, station operation and train/overhead costs. Fleet
capital is recovered separately. Diagnostic per-edge net allocations reconcile
to the network net, splitting shared tiles by active owners, station cost by
degree, fleet by train-time and fare revenue by ridden distance.

The tile model charges one `cell_km` unit per occupied raster cell. Thus union
accounting is exact within the chosen cell approximation; it is not a
sub-cell engineering cost integral. MCP uses geometric movement costs to choose
paths; recorded/priced capex comes from their fixed cell footprints. Smoothing
can adjust reported route length without changing tile ownership. Fixed-cell pricing has a direction-dependent discretization bias (a diagonal
step covers more distance than a cardinal step for the same tile charge).
Compare grid resolutions before interpreting costs.
Shared cells affect capital, not service topology: geometric crossings create
no passenger junction without a shared city node. Frequencies/capacity are
edge-based; overlapping corridors do not have a combined tile throughput limit.

## Cache and output

`runs/railgen-cache/<sha256>/` stores `edges.parquet`, `tiles_CR.npz`,
`tiles_HSR.npz`, compact tile cost vectors and a city/metadata manifest.
Fingerprints cover input **contents**, routing settings, track parameters,
city preparation, demand-dependent candidate ranking and cache schema. Search,
finance and constraint changes reuse geography. A temporary directory is
renamed only after a complete cache is written. Only cells used by some
candidate alignment are retained for evaluation speed.

The main output directory describes one evolving network:

- `latest.png`: the single map overwritten at snapshot cadence. The first frame
  contains the real region/city nodes and the empty network. The final accepted
  network is always rendered, even between scheduled snapshots.
- `progress.gif`: the map animation, updated atomically at the same cadence.
  `reporting.snapshot_every` (default 5) and `reporting.gif_frame_ms` (default
  500) have matching `--snapshot-every` and `--gif-frame-ms` CLI flags. GIF timing
  must be a multiple of 10 ms. Only compressed past frames are copied when
  appending; historical image buffers are not retained in memory.
- `progress.png`: stations/corridors, annual trips, captured population and
  connectivity, track/fleet, upfront capital, and annual finances versus iteration.
- `latest.json`, `latest.geojson`, `cost_breakdown.log`: metrics, geometry and
  reconciled accounting for the same current network shown in `latest.png`.
  Per-edge costs/revenue use the evaluator's allocation rules.
- `history.jsonl`: every iteration, including rejected proposals, with parent
  genome, accepted genome, changed edge IDs/states, metrics and stage. This is
  the actual incumbent lineage, not a sequence of independently selected front
  members. `search_history.json` also records search-side history.
- `checkpoint.npz`: current integer genome, objectives, iteration, graph
  fingerprint and effective configuration. It is a state export; solver/RNG
  resumption is not implemented.
- `report.md`: the current network, animation, progress charts, accounting and
  a link to Pareto comparisons. `manifest.json`, `run_config.json`,
  `alignment_manifest.json` and `backtest.json` record assumptions and provenance.

`pareto/` contains comparison outputs written after search: `pareto.csv`,
`genomes.npz`, `designs/*.geojson`, `pareto.png`, representative `design_*.png`
maps, per-edge tables in `report.md`, `loss_sanity.json`, and optional
`sensitivity.json`. The front CSV, genome array and loss diagnostics are also
available at the output root for compatibility. Every built edge with allocated
annual loss over EUR 200M is flagged. Sensitivity performs fresh searches for
demand k, capital and yield at ±25% on fixed alignments, reporting exact genome
persistence; it does not vary every fitted exponent jointly.

Maps share the original optimizer's visual styling in
`train_optimizer/map_style.py`: population density uses the `YlOrRd` log scale
(people/km²), white tile borders and 65% opacity; tracks run from pale green
(`#b2f2a5`) to bright green (`#00cc33`). Conventional track is dashed, HSR is
solid, and width is exactly `1.7 + 0.6 × trains/hour/direction`. White station
markers, dashed green catchment circles and grey city crosses match the original.
Both live/GIF frames and representative maps use this renderer.

For the fast model, track color measures two-way carried annual trips divided by
the fixed capturable gravity markets of the OD pairs using that edge, before rail
share and spill. Transfer ODs contribute to each leg, once per edge. The map
labels this **rail-market satisfaction**; it is distinct from occupancy and the
legacy model's separately computed rail-eligible-needs bound. Population is read
from the actual population cells and divided by cell area, rather than inferred
from city populations. H3 cells retain their hexagonal footprints; other polygon
cells retain their shapes, and non-H3 point cells use their declared area as a
square footprint. Country/bbox filtering follows the original display convention
when boundary data is available. The prepared population layer and viewport are
reused across snapshots, with fixed color scales and space reserved for both
colorbars so animation frames do not shift or resize.

In optional NSGA mode, parents and offspring evolve across generations. The live
view displays that generation's Pareto compromise, which can change lineage;
choose the default incremental mode for one continuous network trajectory.
The final map always follows the live accepted state and is not replaced by a
financial/coverage extreme from the alternative-design report.

## Performance and scope

On the development machine, a 100-city / 316-edge France graph evaluated 300
random designs at **5.62 ms median / 8.54 ms p95**, excluding cache loading.
Use the benchmark command to measure your own inputs and hardware. Earlier NSGA-II
150-member, 200-generation searches produced 449 France designs in 40.36 seconds
and 484 Europe designs in 41.78 seconds; these timings include seeding and exclude
geography loading/precomputation and report rendering. Development reports are
in `runs/railgen-france-v2/` and `runs/railgen-europe-v2/` (ignored by git).

The generator and live-output tests cover lineage, fixed-baseline consolidation,
rejected proposals, interruption/time limits, GIF cadence and final-state consistency.
All 36 generator/map/live-output tests pass. The full suite has 173 passing tests
and three pre-existing failures, reproduced on the original HEAD: profile time-limit
precedence in `test_settings` and two legacy HSR default assumptions in
`test_train_types`. No legacy model/configuration changes are part of this work.

The v0.1 pipeline, with the incremental search extension, supports all mandatory computational stages; empirical
validation is pending a curated metro table and sourced corridor observations.
Optional artificial hubs, alternative alignment genes, CO2 objectives, AMOSA,
full line pools and cross-sea hand-built links are not implemented. NSGA-III is
the available many-objective alternative; the installed pymoo MOEA/D does not
support the required constraint mechanism. There is no global optimality claim.
