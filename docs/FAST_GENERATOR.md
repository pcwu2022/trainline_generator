# Fast multi-objective railway generator

`rail_gen/` implements the two-layer architecture in the supplied
[v0.1 specification](FAST_GENERATOR_SPEC.txt). Geography is computed once;
network search uses fixed abstract edges and never calls routing or HiGHS.
The earlier optimizer remains available through `optimize.py`.

## Run

From the repository root, using Python 3.9 or newer:

```sh
.venv/bin/python -m pip install -r requirements-railgen.txt
.venv/bin/python -m rail_gen run --profile smoke --output runs/railgen-smoke
.venv/bin/python -m rail_gen run --profile france --output runs/railgen-france
.venv/bin/python -m rail_gen run --profile europe --output runs/railgen-europe
```

Choose a new or empty output directory. Existing runs are never deleted.
`--population`, `--generations`, `--seed`, and `--workers` override search settings.
`--workers` uses a shared-memory thread pool for independent evaluations; SciPy
and NumPy kernels can release the GIL. One worker is the default: parallel
speedups depend on network size and hardware.

```sh
.venv/bin/python -m rail_gen precompute --profile france
.venv/bin/python -m rail_gen benchmark --profile france --evaluations 200
.venv/bin/python -m rail_gen run --profile smoke --sensitivity --output runs/railgen-sensitivity
.venv/bin/python -m unittest discover -s tests -p 'test_railgen.py'
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
- `search.py`: integer NSGA-II (default), or NSGA-III for four objectives;
  uniform crossover, flips, cheapest component joining, low-value removal,
  busy-edge upgrade and low-load downgrade. Greedy marginal-net/capital growth
  supplies a fraction of seeds at multiple stopping sizes; the rest are sparse
  random/demand-biased states. The empty design is included. Zero-load repair
  writes the genome back into the pymoo population. Feasible non-dominated
  designs are archived across generations, not just from the final population.
- `calibrate.py`: fixed-line prediction, log-error fitting and held-out checks.
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

Each run writes:

- `pareto.csv`: all feasible non-dominated designs, objective values, finance,
  population, passengers, fleet, spill and topology metrics.
- `genomes.npz`: integer edge states aligned with the cached edge IDs, plus
  objective vectors for reproducibility.
- `designs/*.geojson`: every Pareto design's edges (track, length, standalone
  capex, directional load, frequency, allocated net) and all city nodes.
- `report.md`, `pareto.png`, and representative `design_*.png` maps with per-edge
  tables. The representative anchors include best net, maximum passengers
  within the requested subsidy ceiling when available, and the knee compromise.
- `run_config.json`, `alignment_manifest.json`, `search_history.json`,
  `backtest.json` and `loss_sanity.json`. Every built edge with an allocated
  annual loss above EUR 200M is flagged.
- `sensitivity.json` when requested: fresh searches for demand k, capital and
  yield at ±25%, using fixed candidate geometries and reporting exact genome
  persistence. It varies the demand scale, not every fitted exponent jointly.

## Performance and scope

On the development machine, a 100-city / 316-edge France graph evaluated 300
random designs at **5.62 ms median / 8.54 ms p95**, excluding cache loading.
Use the benchmark command to measure your own inputs and hardware. Full
150-member, 200-generation searches produced 449 France designs in 40.36 seconds
and 484 Europe designs in 41.78 seconds; these timings include seeding and exclude
geography loading/precomputation and report rendering. Development reports are
in `runs/railgen-france-v2/` and `runs/railgen-europe-v2/` (ignored by git).

The 25 new generator tests pass. The full suite has 162 passing tests and three
pre-existing failures, reproduced on the original HEAD: profile time-limit
precedence in `test_settings` and two legacy HSR default assumptions in
`test_train_types`. No legacy model/configuration changes are part of this work.

The v0.1 pipeline supports all mandatory computational stages; empirical
validation is pending a curated metro table and sourced corridor observations.
Optional artificial hubs, alternative alignment genes, CO2 objectives, AMOSA,
full line pools and cross-sea hand-built links are not implemented. NSGA-III is
the available many-objective alternative; the installed pymoo MOEA/D does not
support the required constraint mechanism. There is no global optimality claim.
