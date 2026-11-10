# Population-driven railway optimizer

## Annealing search (default)

The default `solver_method="annealing"` starts empty and changes the network one
proposal at a time. It routes corridors only when needed and caches them. Moves
add, extend, remove, join, split, reroute, or change the frequency of services;
they can also move station sites, close stations, and reuse track through hubs.
Early exploration seeds useful regional services in new areas. At 30% passenger-need satisfaction,
integration prioritizes bridges whose onward journeys connect existing regional
networks. At 75% satisfaction, a validated network and sufficient connectivity allow
cost consolidation. Final connected-network requirements remain enforced;
exploratory components can be separate. Set `annealing_coverage_metric` to
`needs` (the default), `markets`, `population`, or legacy `demand` to choose
the measure. Needs satisfaction is completed trips divided by fixed rail-eligible
demand, accounting for uptake, station access, and best-case journey limits.
Opening stations alone does not satisfy needs. The share of all potential trips
served is reported separately.

Default `annealing_budget_mode="goals"` continues until those stages finish or
Ctrl+C is pressed. `--time-limit 120` selects a bounded 120-second search;
`--budget-mode goals` explicitly selects goal-driven stopping. In goal mode,
`solver_time_limit` caps individual passenger LPs and the proposal cap is ignored.
After the target is reached, the search tests shared-track rerouting,
station relocations, intermediate station branches, through services, and lower
frequencies to cut lifecycle cost while preserving at least 98% of baseline
ridership globally and per origin, with at most 5% higher mean journey time.
Worse proposals can be accepted during growth. The live map shows labelled
exploratory networks before coverage targets are met; validated results remain
in `best_feasible.json` and `checkpoint.npz`. The final map shows the selected
validated result when available. `exploration.png` and `exploration.json`
preserve the broader developing network when its components are not yet joined.

```sh
.venv/bin/python optimize.py --config configs/netherlands.json --solver annealing --output runs/annealing
```

Corridor addition proposals are weighted by demand divided by an estimate of
construction capital. `annealing_proposal_cost_exponent=1` applies this cost
preference; zero restores demand-only sampling. The estimate uses sparse terrain
samples and never substitutes for actual routing or validation.

`annealing_iterations` defaults to 1000 proposals in bounded mode. `--time-limit` includes lazy
routing and passenger evaluation; population preparation is additional. See
[ALGORITHM.md](ALGORITHM.md) for the complete formula, settings, constraints,
checkpoint behavior, and limitations. Existing run outputs are not modified.

Settings are loaded in order: `Config` defaults in `train_optimizer/config.py`,
then `config.json`, then the selected regional JSON, then CLI overrides. The
standard Europe, France, Netherlands–Belgium, and UK profiles inherit the Python
time default (currently 3,600 seconds). An explicit regional override, such as
the Italy profile, still takes precedence. Startup logs and `manifest.json`'s
`run_settings` record the effective limit and whether it applies to the total
search or individual passenger LPs.

## Earlier heuristic solver

The optional `solver_method="heuristic"` chooses one station per discovered market
by capture fraction, breaking ties by access time. It adds demand-ranked corridors
to the nearest-neighbor graph and searches direct and through services. Guided
moves add shortcuts, replace shuttles with through trains, bypass intermediate
markets, prune services and adjust frequencies. Dense networks and trees provide
initial designs; the final network can contain cycles. Each design fixes every
discrete decision; HiGHS solves continuous passenger assignment and connectivity.
Acceptance uses the full lifecycle, ridership and passenger-time objective, with
all hard constraints checked. No optimality guarantee is reported. An unsuccessful
search does not prove infeasibility; the final status is `search_exhausted`.

```sh
.venv/bin/python optimize.py --config configs/netherlands.json --solver heuristic --output runs/heuristic
```

`heuristic_iterations` (default 100) limits proposed designs, and `--time-limit`
limits solver/search time after preparation and model construction. Increase
iterations for a longer search. `random_seed` controls reproducible proposals.
The heuristic fixes one station location per market and includes up to
`heuristic_through_lines` (default 12) three-stop through-service patterns, ranked
by endpoint demand and bounded by the configured line and detour limits.
`heuristic_demand_connections` (default 2 per market) adds candidate corridors
ranked by potential passenger-km. Construction and service costs are charged in
full; no small-city detour receives an artificial saving.

`maximum_journey_detour_ratio` limits OD-average travel time relative to direct
rail running time, plus access and one initial wait. The root config sets 2;
zero disables this cap. `maximum_journey_hours` remains an additional limit.
The objective's `time_weight` controls further travel-time preference. Profit is
a soft objective, so a validated network can still lose money.

The full MILP remains available with `--solver milp`. It searches station
alternatives and a larger through-service pool. Checkpoints require an identical
generated model; use a new output directory after these candidate-pool changes.

## UK profile

`configs/uk.json` discovers up to 20 UK markets, including Northern Ireland.
Separate rail components are allowed. The profile supplies a 120-second LP/bounded-search setting
and the default 75% rail-eligible need-satisfaction target.

```sh
.venv/bin/python optimize.py --config configs/uk.json --output runs/uk
```

## Regional coverage scenario

Shared-track services have a dedicated candidate budget:
`maximum_shared_track_lines=40` adds three-stop hub patterns beyond the generic
line pool, with `maximum_shared_track_detour=1.35` limiting routed length to
1.35 times direct endpoint distance. Candidates are ranked by potential capital
savings if a spoke is already shared. The solver decides whether opening the
intermediate station is worthwhile, charging its full cost and stopping time.
Physical track capital and maintenance are paid once per selected edge, while
each service pays its own fleet and operating costs and consumes shared capacity.
Set the shared-track budget to zero to disable this additional pool. Hubs must
be existing candidate station sites; geometric track crossings alone do not
create junctions or shared infrastructure.

The Netherlands/Belgium profile sets `maximum_extra_lines=120` (the generic
default remains 40). With the 40-pattern pool, the Jette–Delft–Amstelveen
through service was excluded even though its route is only about 2.8% longer
than the direct Jette–Amstelveen corridor. Increasing the pool admits this
shared-track option without changing fares, construction prices, objective
weights, or the detour limit. A larger pool also makes the MILP harder; a
time-limited fresh run is not guaranteed to discover the shared solution.

Objective defaults prioritize ridership (`served_weight=10`), population within
operating station catchments (`population_weight=1`), and shorter passenger
journeys (`time_weight=0.5`, including access, waiting and transfers).
`access_weight=0.05` favors convenient access; `profit_shortfall_weight=0.25`
retains a softer financial penalty. These are tunable preferences, not calibrated
economic values. Population coverage uses fixed exclusive zones to avoid double
counting; `latest.json` reports its population count and fraction.

`configs/netherlands.json` now requires service at 75% of candidate markets
(15 of 20), at least 10% of fixed regional potential demand, and at least
5% of each selected market's outbound demand. Every selected line operates at
least twice per hour in each direction. Station circles may overlap; fixed
nearest-market population zones prevent double counting. These thresholds are
scenario assumptions, not calibrated service standards.
The scenario uses five neighboring markets and a 250 km corridor limit to
connect the candidate graph; the previous three-neighbor/180 km graph split
into components of five and three markets, making six-market coverage impossible.
Country boundaries restrict population-based discovery to the configured countries.

`minimum_city_served_fraction`, `minimum_selected_origin_served_fraction`, and
`minimum_service_frequency` control these requirements. Set `upfront_budget`
and/or `maximum_annual_subsidy` to positive euro amounts to impose hard spending
limits; zero disables each ceiling. Subsidy uses lifecycle annual-equivalent
cost minus fare revenue, including capital, renewals and residual credit. No
budget is chosen automatically. The weighted objective and soft profitability
penalty remain active. `latest.json` reports city coverage, fare recovery,
annual-equivalent subsidy, and subsidy per passenger; `cost_breakdown.log`
provides readable accounting for each run.

Access remains restricted to each population cell's fixed market zone, and
journey limits remain OD averages. Cross-market station choice, per-route time
limits, and budget/coverage scenario sweeps are not implemented here.

An end-to-end **strategic network optimizer** for the supplied European population
and world city data. It jointly chooses station locations, catchment radii, track
corridors, multi-stop all-stop lines, frequencies and capacity-constrained passenger
flows. It penalizes shortfalls against a **ten-year payback target** and saves the best feasible network while
it searches. This is a research/scenario model, not an engineering alignment,
calibrated ridership forecast or minute-by-minute timetable.

## Install and run

Python 3.9+; run from this project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python optimize.py --config configs/netherlands.json --output runs/my-region --time-limit 3600 --snapshot-every 5
```

The command runs for up to one hour of **solver time**, with additional time for
preparation and final exports. It can finish early if optimality or infeasibility
is established. `--snapshot-every 5` requests a progress frame after each five
solver progress steps, even when no better network has been found. An iteration
here means one **native solver progress callback**, rather than seconds, a new
solution or a completed branch-and-bound node. HiGHS reports live LP iteration
counts as unavailable during MIP search, so counting those would not produce
intermediate frames. The callback count is explicitly labeled in the reports;
final LP iteration totals are included separately. Population data and map
geography are loaded before rendering the first image. No blank loading image
is written to `latest.png` or included in the GIF. The complete map and final
state are always saved once data is available. The
output directory must be new or
empty unless resuming or using `--overwrite` to delete its existing contents.

Open **`runs/my-region/index.html`** for a live dashboard, or watch
**`runs/my-region/latest.png`**. The dashboard works directly as a local HTML file;
it reloads the latest results every two seconds without external services. Open
**`runs/my-region/progress.gif`** to watch the accumulated progress animation.

If no feasible network exists yet, snapshots show population and city candidates
with an explicit status, rather than inventing a profitable solution.

### Example assumptions

`configs/netherlands.json` selects up to 20 population-derived markets. Its monetary and demand defaults are uncalibrated; it may be infeasible.
The bounding box defines the study area and can include neighboring countries.
`countries` filters population cells using the configured boundary dataset.
City-file records provide optional labels only. Verify source coverage before
interpreting national totals.

`configs/exploratory.json` supplies explicitly **optimistic illustrative costs**,
a higher trip rate and a higher fare, to exercise feasible network searches. A
profitable result with it is not evidence of real-world profitability:

```sh
.venv/bin/python optimize.py --config configs/exploratory.json --output runs/exploration --time-limit 120 --snapshot-every 5 --gif-frame-ms 400
```

Export the built-in defaults, edit the JSON, then use it with `--config`
(the export does not copy your edited `config.json`):

```sh
.venv/bin/python optimize.py --write-default-config my-config.json
.venv/bin/python optimize.py --config my-config.json --output runs/custom
```

Paths in JSON resolve relative to the current working directory. Unknown keys and
invalid parameters are rejected. The current cost assumptions use euros;
distances are km, densities people/km², demand directional trips/hour.

## Tune costs and station access

`config.json` contains all optimizer assumptions, including costs in **euros**,
objective weights, candidate limits, access speed, solver duration and animation
settings. The CLI loads it automatically; `--config configs/netherlands.json`
applies top-level regional overrides on top of it (nested values are replaced,
not recursively merged). CLI runtime options override JSON. The current defaults
use H3 resolution 6 population cells, `population_subdivision=0`, and a 120-second
solver limit.
Run the Netherlands–Belgium scenario with the supplied 3 km population H3 tiles:

```sh
.venv/bin/python optimize.py --output runs/my-netherlands-belgium-3km
```

Conventional track costs **€10M/km**. The urban surcharge applies only above
`urban_density_threshold` (default 5,000 people/km²), increasing linearly with
excess density D: **C(D) = 10 + 0.005 × max(0, D − 5000) M€/km**.
For explicitly classified terrain,
the terrain baseline replaces the conventional baseline when higher, then the same urban
surcharge is added. `terrain_cost_position` selects the low (0), midpoint (0.5,
default), or high (1) end of these configurable ranges:

| Terrain class | Range (M€/km) | Default (M€/km) |
| --- | ---: | ---: |
| `water` (mapped crossings) | continuous-crossing exponential | 120 × exp(L / 25) |
| `hilly` | 15–25 | 20 |
| `moderate_mountains` | 25–40 | 32.5 |
| `difficult_mountains` | 40–70 | 55 |
| `very_difficult` | 70–120+ | 95 |

The 120 upper value is editable for more expensive base tunnels. Assign classes
in `terrain_file`, e.g. `{"terrain_class": "hilly"}` for an H3 cell; optional
`multiplier` then scales the combined terrain/urban cost. Missing classes use
conventional costs. No water or mountain classes are inferred without data.
The supplied scenario uses `data/exportImage.tiff` for approximate terrain costs;
it has no separate land exclusion mask.

For each station option, covered population samples provide
`mean_access_km = sum(people_i * distance_i) / sum(people_i)`. At subdivision 0,
these are the supplied hexagon centers; membership uses center-in-circle and
fixed city origin zones. Let `ideal_access_km` be the minimum of the same weighted
mean over possible station positions for those **same covered residents** (their
weighted geometric median). The `access_multiplier` is
`max(1, mean_access_km / max(ideal_access_km, access_normalization_floor_km))`.
This is 1 at a symmetric center, and also at any weighted geometric median.
The tiny configurable floor handles coincident population centers.

The objective adds `access_weight * (access_multiplier - 1) / number_of_cities`
for each selected station. This avoids charging a constant just for opening a
station. `mean_access_km / access_speed_kmh` also enters passenger journey times
at both ends. Set `access_weight` to tune the location penalty and `time_weight`
to tune passenger time. `center_weight` defaults to zero; station placement no
longer defaults to a density peak. Candidate locations are finite, so the network
optimizer chooses among generated sites rather than every coordinate.

Previous generated runs are preserved in `runs/archive_previous/`. New results
are isolated in their own output directory; `index.html`, `latest.png`,
`latest.json`, and `progress.gif` are the main files to inspect.

## Saved Netherlands–Belgium scenario

The historical run in `runs/netherlands-belgium-3km/` predates population-only
seeding and unified 25 km markets. It used the 3 km population data with
eight city seeds: Brussels, Rotterdam, Amsterdam, Antwerp, The Hague, Utrecht,
Ghent and Groningen. It contains 3,621 population cells, 32 station options,
56 track candidates and 96 candidate lines. The solver reached its 120-second
time limit without a feasible incumbent; **infeasibility was not proven**.
The saved map shows population and candidate cities rather than selected tracks.
Open `runs/netherlands-belgium-3km/index.html` or inspect `latest.json` for status.

## Live output and resume

Each run keeps a fixed set of output files, updated in place:

- `latest.png`: the latest population density, selected stations, catchment circles and lines
  labeled with departures/hour/direction.
- `progress.gif`: one looping animation containing all rendered progress frames.
  `--gif-frame-ms` controls playback speed (default 500 ms/frame).
- `latest.json`: selected infrastructure/services, ordered stops, directional
  segment loads, served/unmet OD demand, passenger ride flows, journey time,
  capital/operating costs, ten-year revenue/profit, validation, solver gap/status
  and assumptions.
- `latest.geojson`: separate station, physical-track and service-line features.
- `latest_stations.csv` and `latest_lines.csv`: readable station and ordered line summaries.

No numbered snapshot folders or individual historical images/JSON/CSV files are
created. `latest.png` and the latest data files are atomically replaced on every
update. `progress.gif` is also replaced atomically, so it stays readable while
the optimizer runs. GIF encoding holds only the new frame and small copy buffers
in memory; it copies the existing compressed animation without loading its old
images. The animation grows as frames are appended.

The map uses a fixed viewport fitted to the full population cells, candidate
corridors and station catchments. Plot and colorbar areas are reserved separately,
so changing solutions does not move or crop the map inside the animation canvas.

Rendering runs on a worker thread. If the solver advances faster than a frame can
be rendered, pending requests are combined into the next frame using the latest
available state, rather than blocking the search or creating an unbounded queue.
Counters are observed at solver callbacks, so frames can occur just after an
iteration threshold, or skip intermediate thresholds if rendering falls behind.
Each image and JSON report shows its actual progress-step count. No wall-clock timer
creates frames, and a long presolve with no progress callbacks waits for the next
callback. A final frame is still saved if the run ends before reaching a threshold.

Reports distinguish potential trips outside rail access/uptake from **accessible
but unserved** rail trips. Station boarding counts include transfers; served-trip
counts count each passenger journey once.

`progress.jsonl` records compact update history in one file; `manifest.json` records
configuration, input hashes, dimensions, solver version and model fingerprint.

`checkpoint.npz` stores the latest feasible incumbent. Ctrl+C or SIGTERM requests
a stop at the next solver callback and saves a final snapshot. The snapshot
thread never calls the solver. Native
[HiGHS incumbent callbacks](https://ergo-code.github.io/HiGHS/dev/callbacks/)
copy improving solutions and request an update at iteration thresholds. Rendering
and GIF writing happen off the solver thread.

Start fresh in an existing output directory (permanently deletes its contents):

```sh
.venv/bin/python optimize.py --config configs/netherlands.json --output runs/netherlands-belgium --overwrite
```

`--overwrite` and `--resume` cannot be used together.

Resume with the **same model configuration and input data**:

```sh
.venv/bin/python optimize.py --config configs/exploratory.json --output runs/exploration --resume runs/exploration/checkpoint.npz --time-limit 3600 --snapshot-every 10
```

This is an incumbent **warm start**, not restoration of the previous solver's
branch-and-bound tree. A model fingerprint and feasibility check reject
incompatible checkpoints. Abrupt power loss/SIGKILL can lose progress since the
last completed update; the last published image, GIF and checkpoint remain
readable. Resume appends frames to the same GIF and continues update numbers;
solver iteration and elapsed-time counters restart for the new solve invocation.
Resuming an older run without a GIF starts its animation at the resume point.
Existing snapshot folders from the previous output format are not deleted, and
no new ones are created.

Exit codes: `0` feasible result (inspect optimality gap), `2` proven infeasible,
`3` stopped without a feasible incumbent, `130` interrupted, `1` error.

## What the optimizer implements

`market_radius_km: 25` replaces `center_search_km`, `service_radii_km`,
`minimum_city_spacing_km` and `seed_search_radius_km`. Custom configurations
must remove those old keys. `max_cities` and `minimum_city_population` retain
legacy names for the population-derived market count and target population.


1. **Population and centers.** Filter population cells by `bbox` and country
   polygons from `country_boundaries_file`. Empty `countries` means bbox only.
   Discover markets directly from eligible population cell centers, with no
   predefined city anchors. Rank by distance-weighted population within `market_radius_km`
   using a Gaussian weight with scale `market_radius_km / 3`, favoring compact
   urban centers over positions between towns;
   after each pick, rescore using only population outside previously selected
   scoring neighborhoods. Require at least `minimum_city_population` new population
   per pick. Enforce `minimum_seed_spacing_km` (zero uses `market_radius_km`), up to
   `max_cities`. Set radius to `center_coefficient * sqrt(seed density)` using
   density in people/km², with no floor or cap. The default coefficient 0.16136
   gives Brussels approximately 20 km. Use this radius for center calculation
   and catchments.
   Seeds remain fixed while the solver chooses markets and connections.
2. **Fixed origin zones.** Assign eligible samples to their nearest population
   seed, capped at that seed’s adaptive radius. Overlaps have one owner. Samples
   below `density_threshold` or outside their owner’s radius contribute no demand.
   Exclusive ownership can reduce final market population below the target.
   Station catchments must also fall within their owning market. Optional city
   records label finalized centers using the nearest city within
   30 km; repeated labels receive a market-number suffix. Missing labels use
   `Market N`. City records have no effect on seed selection. Country filtering
   uses cell/sample centers; tracks can still cross political boundaries.
3. **Catchments.** Select at most one station/location/radius option per city.
   Eligible residents of that origin zone within the radius supply coverage.
   Uptake is `p_max*(1-exp(-uptake_a*R**uptake_alpha))`, with `R` equal to
   `uptake_reference_radius_km` when positive, otherwise the catchment radius.
   Regional scenarios use a 25 km reference to decouple uptake from adaptive
   geometry. Capture is covered
   population / zone population times uptake. Selected circles may overlap;
   fixed exclusive origin zones and one station per market prevent double counting.
   Cross-zone access to another market's station is not yet modeled.
4. **Gravity demand.** Default `production` mode generates
   `trips_per_person_hour * zone_population` outbound trips and distributes them
   by destination population / distance^`gravity_beta` (default 2).
   Optional `gravity` mode uses exactly
   `gravity_scale * population_i * population_j / max(distance,distance_floor)^beta`.
   Both require calibration. Maximum OD distance and a distance floor are tunable.
   For selected endpoint options a,b, rail demand is bounded by
   `fixed_potential_OD * capture_a * capture_b`, assuming independent endpoint uptake.
5. **Track and terrain.** Generate nearby-city corridor alternatives for station
   locations. Evaluate a direct geodesic and a cheaper terrain-avoiding any-angle
   alternative when found. Integrate terrain baseline plus
   `5,000*max(0, density-5,000)` euros/km along each alignment.
   Optional H3 terrain classes, multipliers, blocked cells and a land mask enable coarse
   least-cost routing. Shared candidate track is constructed and charged **once**,
   regardless of the number of services using it. Geometric crossings alone do
   not form junctions or shared track.
6. **All-stop lines.** Include every two-stop corridor and a bounded set of
   multi-stop patterns (`maximum_line_stops`, default 3; `maximum_extra_lines`,
   default 40). Multi-stop candidates favor low-detour paths. Every intermediate
   stop must be open. Trains can carry passengers through without another wait
   or transfer. Each line has a discrete frequency option (default 1, 2 or 4/hour
   **in each direction**), train capacity, dwell time and required fleet. Physical
   track also has a configurable train throughput limit.
7. **Passenger assignment.** Multi-commodity flows conserve each directed OD's
   passengers. A ride can board/alight at any stop pair on a selected line. All
   overlapping rides consume capacity on every directed segment they traverse.
   Initial and transfer waits use half the headway. Journey time includes access,
   egress, waiting, travel, intermediate dwell and a transfer penalty. The journey
   limit is an **OD-average** limit, not a maximum for each individual itinerary.
8. **Service hours and economics.** Departures operate 06:00–22:00 in an aggregate
   16-hour/day model, with 365 days/year by default. Every selected line offers at
   least one departure/hour/direction. Revenue counts served passengers only,
   using fixed direct OD fare distance to prevent profitable artificial detours.
   Operating costs include trains in both directions, per-train-km and departure
   costs, track maintenance and stations. Upfront capital includes track, stations
   and the dedicated line fleets. Construction duration adds no passenger delay.
9. **Hard constraints and objective.** Require a nonempty served-demand target,
   optional per-origin service targets, capacity, station/track consistency,
   one connected selected network (`require_connected_network`, default true),
   optional upfront budget. Thirty-year lifecycle profitability is a soft target. Minimize a weighted,
   normalized combination of unmet demand, passenger-hours, population-weighted access penalties
   and annual-equivalent costs. Weights and service targets are configurable.
   Rider and time terms are normalized by a fixed optimistic bound on capturable
   trips, so inaccessible potential demand does not drown out their importance.
   Every exported incumbent is checked against variable bounds, integrality,
   every linear constraint and the financial calculation.

The soft financial target is:

```text
pv_revenue + profit_shortfall >= (1 + financial_margin) * pv_cost + minimum_horizon_profit
```

`profit_shortfall` is nonnegative and measured in lifecycle present-value euros.
The objective adds `profit_shortfall_weight * (profit_shortfall / appraisal_annuity_factor) / 100000000`.
The weight defaults to 1; increase it to favor meeting the profit target, or set
it to 0 to remove the financial penalty. Loss-making networks can be feasible.
Reports retain `validation.financial_ok` as a target check and export the actual
shortfall, modeled slack and penalty in `metrics`.

Defaults are a 30-year discounted appraisal, a separate ten-year cash-flow
report, a 5% margin and a positive one-unit present-value profit floor. `upfront_budget=0` means no separate capital ceiling. The construction cost
is paid at time zero, rather than spread over an unrelated asset lifetime.

`minimum_served_fraction` is measured against **all fixed eligible potential
demand**, including unmet access. With uptake at both endpoints its achievable
value can be far below 100%. Choose it as a policy/scenario input. The positive
service target excludes an empty network.

## Generate European elevation rasters

To search and download Copernicus DEM tiles first, use `fetch_terrain.py`.
Install the optional download dependencies in a Python environment supported by
EODAG 4 (the existing Python 3.9 environment may need replacing):

```sh
python -m pip install -r requirements-terrain.txt
python fetch_terrain.py --bbox -10 35 25 65 --resolutions-km 3 22
```

The default collection is `COP_DEM_GLO30_DGED` (GeoTIFF), using `creodias_s3`
as in the [official EODAG Copernicus DEM tutorial](https://eodag.readthedocs.io/en/stable/notebooks/tutos/tuto_cop_dem.html).
Configure credentials for your selected provider in EODAG before running;
a Copernicus Data Space login alone does not configure every provider.
See [EODAG configuration](https://eodag.readthedocs.io/en/stable/getting_started_guide/configure.html).
Use `--provider` to select another provider supporting the collection, or
`--collection COP_DEM_GLO90_DGED` for 90 m input. DTED collections are excluded
because they do not supply the GeoTIFF files this workflow consumes.

The command searches all result pages and retains downloaded tiles in
`data/copernicus_dem/` for retries. Europe-wide downloads can be large; start
with a small bounding box to verify credentials. It creates a temporary virtual
mosaic and streams it through the resampler below, without allocating a
Europe-wide high-resolution array or intermediate raster. Outputs cover the
complete intersecting tiles, including tile edges beyond the requested bounds.
Incomplete downloads stop processing. Source nodata and uncovered areas remain
excluded from averages. Downloaded tiles and default generated grids are gitignored.


Place a georeferenced, single-band European DEM at `data/europe_dem_90m.tif`,
or pass its path explicitly. The source DEM is not included in this repository;
its geographic coverage and elevation units are preserved as supplied. Install
`requirements.txt` to include Rasterio, then run:

```sh
.venv/bin/python generate_terrain.py data/europe_dem_90m.tif
```

This produces `data/europe_terrain_3km.tif` and
`data/europe_terrain_22km.tif`. Both are compressed float32 GeoTIFFs in the
European metric projection **EPSG:3035**, with exactly 3,000 m and 22,000 m
pixel spacing respectively. Average resampling excludes declared source nodata;
uncovered pixels use NaN nodata. Each output is generated directly from the
source with bounded warp memory. Geographic (degree-based) DEMs are reprojected
before averaging; dividing a degree resolution into 3,000 would be incorrect.
The grid covers the source extent; supply a Europe-only source for Europe-only
outputs. Existing outputs are never overwritten.

To generate just one resolution or use another directory:

```sh
.venv/bin/python generate_terrain.py /path/to/europe_dem.tif --resolutions-km 22 --output-dir data/terrain
```

These rasters contain **mean elevation**, not construction cost multipliers or
rail gradients. The optimizer does not classify these single-band elevation
GeoTIFFs directly: `terrain_file` accepts H3 JSON or the RGB relief map described
below. A 22 km raster is a square grid
and does not align with the population dataset's H3 cells.

## Terrain input

The default `config.json` sets `terrain_file` to `data/exportImage.tiff`.
This file is a three-band, 8-bit **RGB relief image**, covering 11°W–60°E,
34°N–72°N at 1 arc-minute pixels. Its values are colors, not elevations.
`TerrainCostMap` maps WGS84 **longitude, latitude** to construction cost per km:

```python
from train_optimizer.terrain import TerrainCostMap

rail_cost = TerrainCostMap("data/exportImage.tiff")
rail_cost(4.9, 52.37)                  # Amsterdam: 10_000_000 currency/km
rail_cost(3.0, 54.0)                  # North Sea: 120_000_000 currency/km (local baseline only)
rail_cost(8.0, 46.5)                  # Alps: 32_500_000 currency/km
rail_cost(4.9, 52.37, density=100)     # Includes 500_000/km urban surcharge
rail_cost.terrain_class(3.0, 54.0)     # "water"
```

Pass `config=your_config` to use custom costs. With the supplied euro assumptions,
normal land uses €10M/km and blue water uses the separate
`water_track_cost_per_km` short-crossing baseline (€120M/km by default),
and brown mountain shading uses `moderate_mountains` (€32.5M/km midpoint).
The classifier uses blue dominance (`B > 1.12R` and `B > 1.04G`) for water,
then brown dominance (`R > 1.05G` and `R > 1.15B`) for mountains; other valid
pixels use normal costs. These are approximate visual rules for this map.
Hillshade, coastlines and mixed pixels can be misclassified. Smooth high
plateaus may be brown; green mountains may be missed. The image cannot provide
slope, water depth, crossing feasibility or reliable mountain severity, so no
more expensive mountain categories are inferred. Water crossings use a length-dependent scenario function, not a bridge/tunnel design.

For each continuous sampled water run of length **L km**, the router multiplies
its integrated local cost (including urban surcharges and JSON multipliers) by
**exp(L / water_cost_growth_km)**. Defaults give **C(L) = €120M × L × exp(L/25)**
without surcharges. Water runs continue across polyline vertices and reset on
sampled land. Separate rivers are priced separately. Local point-cost maps and
cheap proposal estimates show the baseline; full alignments receive the premium.
The exponential is capped at exp(100) to keep arithmetic finite.

Reference projects inform rounded scenario targets, rather than a currency- and
inflation-adjusted statistical fit:

- [Walton Bridge](https://www.gov.uk/government/news/go-ahead-for-new-bridge-over-river-thames-in-surrey):
  the UK government approved £32.3M for this Thames crossing in 2010. A representative
  0.25 km river crossing costs €30.3M in this model. This is an illustrative width,
  not a claim about Walton's exact water span or a conversion of its project cost.
- [Sydney Harbour Bridge](https://www.transport.nsw.gov.au/system/files/media/documents/2023/harbour-bridge-history.pdf):
  1.149 km including approaches, historical cost £6.25M. Its scale motivates a
  roughly €144M model allowance at 1.149 km; historical pounds are not current euros,
  and approach length is not the same as mapped water width.
- [Channel Tunnel (Bechtel)](https://www.bechtel.com/projects/channel-tunnel-chunnel/):
  about 51 km overall and $14.7B reported project cost. A representative 38 km
  underwater run gives €20.8B. These scopes differ; the model intentionally assigns
  a conservative megaproject-scale allowance to the water section alone.

The €120M/km baseline puts a 0.25 km river at about €30M; the 25 km e-folding
length raises a 38 km crossing to about €21B. A 100 km crossing costs €655B;
500 km costs €2.91e19. Small rivers therefore have modest absolute cost while
long sea crossings become prohibitive. The old explicit `bridge` terrain class
is removed; use `water` for water crossings.

Sampling uses the smaller of `routing_sample_km`, `water_routing_sample_km`
(default 0.25 km), and the H3-derived spacing. Rivers narrower than the sampling
interval or absent from the relief map can still be missed. Water depth and
geology are unavailable. The finite any-angle search uses local segment costs
as a heuristic; the completed alignment is repriced as continuous crossings,
so search is not guaranteed to find the globally cheapest nonlinear route.

The router samples raster costs at points on each alignment, using H3 cells
for the population surcharge. This avoids pricing a coastal water sample as
land merely because its hexagon center lies inland. Coordinates outside raster coverage raise `ValueError` in the mapping
function and are excluded by the router. No separate terrain generation or
download is needed. Existing saved runs retain their previous cost assumptions;
rerun the optimizer to apply the configured map. An explicit `land_file` can
still exclude water entirely when that is desired.

To visualize construction costs on H3 tiles for the configured region:

```sh
.venv/bin/python plot_terrain_costs.py
```

This writes `runs/terrain_cost_tiles.png`, comparing the terrain baseline with
terrain plus the population-density surcharge on a shared M€/km scale.
Use `--bbox WEST SOUTH EAST NORTH`, `--config path.json`, or `--output path.png`
to change the extent, assumptions, or output location. Water and unpopulated
tiles are included, with zero surcharge where population is missing.

H3 JSON remains supported for explicit terrain classes and blocked cells:

Without terrain and a land mask, alignments are **geometric placeholders** and may
cross water. Missing population is unknown; construction pricing there assumes
base cost, an explicitly optimistic assumption. No terrain is invented.

To add a cost/exclusion surface, use `terrain_file` with the same H3 resolution as
the population data:

```json
{
  "h3_resolution": 6,
  "cells": {
    "861969687ffffff": {"terrain_class": "hilly"},
    "86196968fffffff": {"terrain_class": "water", "multiplier": 1.2},
    "861969697ffffff": {"blocked": true}
  }
}
```

Unspecified cells have multiplier 1. Multipliers must be >=1. They multiply the
combined terrain/base and urban per-km construction cost. Use resolution 4
for the older 22 km population file and matching resolution-4 cell IDs. These example IDs are syntax examples, not
a real terrain dataset. Use `land_file` for a WGS84 polygon FeatureCollection of
allowed land. Reconstructing population hexagons is not a land/water mask.

Routing always evaluates the direct geodesic and searches for a cheaper
alternative with weighted Theta*: H3 cells provide waypoints, but shortcuts
cross cells at arbitrary angles. Both alternatives enter the network optimizer
with their actual construction costs and travel lengths. Search has a configurable
limit; failure to find a detour does not discard a valid direct candidate.
`maximum_alignment_variants_per_line` (default 16, minimum 2) bounds combinations
for through services, retaining the fastest and cheapest first. Selected track
records contain `alignment_kind`; line records contain `alignment_kinds`.
Coarse terrain and sampled segments cannot establish exact coastlines or bridge spans.
Gradient, curvature, tunnels, real bridge costs, permits and land acquisition
require additional engineering inputs/constraints. A starting elevation source
is [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).

## Interpretation and scaling

These are MILP solutions **within a finite candidate set**, to the displayed
solver gap, rather than a global optimum over all alignments on Earth. A time
limit may leave an incumbent and a substantial gap. The two example configurations
are scenarios, not economic recommendations. Verify the meaning/year/coverage of
`population` before relying on results; all stored populations exceed 1,000.

The operating model assumes steady hourly demand and evenly spaced services.
One train/hour gives a <=60-minute **scheduled gap**, not a guarantee about
individual waits under queues, disruptions or real transfer schedules. The
06:00–22:00 window constrains departures; late arrivals can occur after 22:00.
Timetable boundary effects, exact first/last services, peak/off-peak variation,
fleet interlining, timetable conflict resolution and depot trips are future work.

Start with 5–15 cities. Variables scale with OD pairs × line patterns × frequency
options × ride segments. `maximum_model_variables` prevents accidental oversized
models; tune the city/candidate counts before raising it. Use a larger
`--snapshot-every` for multi-day runs to keep the GIF small. Updating the GIF
requires copying its compressed bytes, so very long animations also increase I/O.

Future express services and train types can extend the line/service options with
stopping patterns, speeds, seat counts and costs. Mixed-speed infrastructure then
needs overtaking and timetable constraints. The data, line generation, MILP and
output modules are separated to support this extension.

## Code map

- `optimize.py`: command-line entry point.
- `train_optimizer/config.py`: assumptions and validation.
- `train_optimizer/geography.py`: population, city/station candidates and routing.
- `train_optimizer/lines.py`: all-stop candidate patterns.
- `train_optimizer/model.py`: joint infrastructure/service/passenger optimization.
- `train_optimizer/outputs.py`: maps, live dashboard, snapshots and checkpoints.
- `train_optimizer/runner.py`: callbacks, stopping and resume.
- `rail_model.py`: shared modeling helpers and the dependency-free single-line evaluator.
- `generate_terrain.py`: DEM resampling for 3 km and 22 km elevation grids.
- [docs/FORMULATION.md](docs/FORMULATION.md): the implemented algorithm and mathematical formulation.

## Lifecycle appraisal

The optimizer now uses 30-year discounted lifecycle costs for its cost objective
and soft profitability target. `accounting_years: 10` preserves the separate
undiscounted cash-flow report. Set `appraisal_years`, `discount_rate`,
`discount_rates`, and `asset_lives_years` in `config.json`.

Default real discount rates are 4% for optimization and 2%/4%/6% for sensitivity.
Track/station/fleet lives are provisional 40/50/25-year assumptions. Full-cost
like-for-like renewals occur at end of each asset life before appraisal end;
straight-line remaining service life determines terminal residual value. Costs
and revenue stay constant in real prices. No terminal-year renewal is purchased.
These simplified asset groups should eventually be split into structures, land,
signaling and other components. Maintenance remains separate from capital renewal;
ensure calibrated maintenance costs exclude the renewals modeled here.

JSON/dashboard metrics include `lifecycle` with NPV, discounted cost components,
asset renewal schedules, residual values and annual cash flows;
`discount_rate_sensitivity` repeats the appraisal at each requested rate. Changing
the sensitivity rates does not reoptimize the network at those rates. The default
rate alone determines selection. Existing checkpoints require a fresh run.

Network map colors use the `viridis` colormap (0–1) for passenger throughput
divided by full demand for the OD pairs using each service segment, including
transfer legs. Width continues to show train frequency per direction.
`demand.png` is generated before solving, using the same population background
and fixed viewport. It combines both directions of each candidate city pair;
width is proportional to demand in trips/hour. Set `demand_map_top_percent`
(default 10) and `demand_map_min_trips_per_hour` (default 0) to filter connections.
At least one pair is omitted to avoid displaying the complete graph. The dashboard
links to this map. If evaluated demand changes, the demand map is regenerated.
