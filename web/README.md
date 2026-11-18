# Browser network lab

This is a buildless GitHub Pages app. It runs the repository's unchanged
`rail_gen/*.py` sources in Pyodide 0.27.7 (CPython on WebAssembly). JavaScript
loads the runtime, transports updates, and draws the map; it does not implement
the railway algorithm. Python executes in a worker, keeping the interface responsive.

## Parameter help

Question-mark links beside Demand multiplier (k) and Distance decay (beta) open
[`guide.html`](guide.html) in a separate tab at the parameter’s explanation.
The guide covers the incremental search, population catchments, gravity demand,
journey-time elasticities, fares, capacity and finance formulas. It works without
JavaScript or external math-rendering packages and leaves the running lab open.

## Profiles and prepared geography

From the repository root, using the normal Python environment and input data:

```sh
python -m tools.export_web --profile france --city-counts 4 8 12 20 50 100 --output web/scenarios/france
```

To rebuild all bundled profiles:

```sh
python -m tools.export_web --profiles france taiwan usa uk japan germany spain italy poland benelux switzerland india scandinavia usa_northeast --city-counts 4 8 12 20 50 100
```

To add any other existing Python profile, for example Portugal:

```sh
python -m tools.export_web --profile portugal --city-counts 4 8 12 20 50 100
```

The exporter writes to `web/scenarios/<profile>/` by default and automatically
updates `web/scenarios/index.json`. The website reads this catalog, so newly
prepared profiles appear in the selector without changing JavaScript. Only
profiles with prepared bundles are offered. `--output` selects a profile folder
for `--profile`, or a parent folder for `--profiles`.

The profile selector offers France, Taiwan, United States (`usa` in the Python
configs), United Kingdom, Japan, Germany, Spain, Italy, Poland, Benelux,
Switzerland, India, Scandinavia and the northeastern United States. Each offers
city limits of 4, 8, 12, 20, 50 and 100, with 12 km routing cells and the actual ETOPO
raster. These are upper limits: population filtering and metropolitan catchment
screening can retain fewer cities. The selector shows the available count when
it is below the requested limit; it never duplicates cities to fill a quota.
For example, the current Switzerland input retains 10 eligible cities at a limit of 100.

Benelux combines Belgium, Netherlands and Luxembourg. Scandinavia combines
Denmark, Norway and Sweden. India uses `data/asia.geojson`. `usa_northeast` uses
the Census Bureau's nine Northeast states: CT, ME, MA, NH, RI, VT, NJ, NY and PA.
See [`data/usa_northeast/README.md`](../data/usa_northeast/README.md) for boundary
provenance, clipping rules and regeneration. Each count has a separate graph with its own population catchments and candidate routes.
Common search, finance, demand, operating hours, load factor and fare settings have
form controls; edits stay synchronized with the full JSON configuration. Switching
city counts preserves these settings and clears results from the previous scenario.
Switching profiles loads the selected profile’s defaults and clears previous results.
Only the selected region and city-count bundle are downloaded. Use `--max-cities`, `--cell-km`, `--config`, and `--cache-dir` to prepare
another resolution or input configuration. Unchanged bundles are reused only when
the configuration and input fingerprint match; use `--force` to re-render them. The exporter uses the existing
`cached_graph` pipeline, including population catchments, terrain routing,
water crossings and construction costs. It writes an `index.json` scenario catalog plus a folder per city count containing:

- `background.png`: population density and terrain difficulty, rendered offline.
- `graph.zip`: cities, alignments, sparse infrastructure matrices and tile costs.
- `scenario.json`: default config and projected map display coordinates.

Only these compact artifacts are downloaded by the browser. Source population
GeoJSON and the 464 MB ETOPO raster are not requested. The map image is for display;
route geometry and terrain costs in the prepared graph drive the optimization.

Demand, operations, finance, constraints, rolling stock, services and search can
be edited without rebuilding. The city-count selector switches between prepared graphs. Other geography, catchment preparation, track construction
parameters and alignments are fixed in a scenario; the Python adapter rejects edits
to these settings. Regenerate the scenario for such changes. Browser execution
currently supports incremental search with one worker; calibration and sensitivity
runs remain available in the CLI. The Stop button terminates the worker and keeps
the last displayed result; it is not a resumable checkpoint.

## Preview and publish

```sh
python -m http.server 8000
```

Open `http://localhost:8000/web/`. Opening HTML through `file://` does not support
worker and source loading. No npm build or server backend is required.

In GitHub repository Settings → Pages, publish the repository root from your
chosen branch. The root `index.html` links to the repository and `./web/`.
All asset paths are relative, including the Python sources in `../rail_gen/`,
so GitHub Pages project subpaths work. Deploy the **whole repository root**,
not just `web/`. No elevation files or `runs/` caches need to be deployed.

If Python files are added or removed, refresh the source manifest:

```sh
python -c "import json; from pathlib import Path; Path('web/python-files.json').write_text(json.dumps(sorted(str(p) for p in Path('rail_gen').glob('*.py')), indent=2)+'\\n')"
```

The first execution downloads the pinned Pyodide runtime and scientific packages
from jsDelivr, and pure Python dependencies from PyPI. An internet connection is
required; browsers may cache downloads. Runs and edited configs stay in memory;
download results/configs to retain them. Larger networks can take substantial
CPU time and memory in a browser.

## Third-party Python source

`vendor/pymoo-0.6.1.3.zip` contains unmodified upstream Python files and the Apache
2.0 license from the official PyPI source distribution. Native extensions and
non-Python assets are omitted. The incremental optimizer uses pymoo's existing
pure Python nondominated sorting implementation. Dependency versions are pinned
in `worker.js`. Upstream: https://pypi.org/project/pymoo/0.6.1.3/.
