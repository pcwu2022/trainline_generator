# Optimizer performance

The evaluator shares candidate-local infrastructure data across its service-plan
trials. This includes selected tiles, edge/tile movement capacities, allocated
track capital, station degrees, connectivity, and layout multipliers. A repaired
genome gets a fresh context. Contexts are not retained in a global cache, so
parallel candidate evaluation cannot mix resources from different designs.

Service-plan generation returns its admitted service networks to the evaluator.
Identical proposed plans are skipped before resource construction. When admission
rejects services, the network is rebuilt for the admitted subset so segment and
service indices remain correct. The ordinary `generate_plans` API still returns
service tuples.

Each service network builds its passenger-routing CSR graph and arc lookup once.
Frequency changes update boarding weights; zero-cost alighting arcs remain
explicit. OD source indexing, demand ranking, and edge lookups are prepared once
per evaluator. Segment-to-edge maps and resource COO indices are reused. A
converged frequency iteration supplies the final assignment directly. Losing
trial evaluations are released while selecting the best plan.

Live updates already contain complete allocations. The live maps and dashboard
reuse those results rather than independently assigning passengers again.
Standalone financial maps and analysis retain reevaluation by default;
`reevaluate=False` is reserved for an existing result from the same graph and
configuration. Rendering remains synchronous and snapshot cadence is unchanged.

## Measurements

Local measurements on the cached graph selected by
`runs/europe-live/run_config.json`: 100 cities, 270 candidate edges, 18,124 tiles,
and 2,333 eligible OD pairs. Both versions ran in the same Python environment.
The reference was a copy of the working code immediately before these changes,
including the preexisting uncommitted service-model changes.

For each density, eight fixed-seed random designs were evaluated with
`repair=True`. Timings are medians over three passes (24 samples). Alignment
loading, geography preparation, and reporting are excluded.

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 5% selected edges | 16.58 ms | 6.83 ms | 2.43x |
| 25% selected edges | 59.49 ms | 35.01 ms | 1.70x |
| 67% selected edges | 138.42 ms | 105.79 ms | 1.31x |
| Saved 57-edge incumbent | 25.13 ms | 13.97 ms | 1.80x |
| Ten incremental iterations, 150 proposals per iteration | 18.99 s | 6.99 s | 2.72x |

The search benchmark used one worker, seed 42, and no reporter. Both versions
performed 1,480 evaluations and produced the same accepted trajectory, Pareto
front, service plans, and metrics. The shortened ten-iteration configuration
also shortens the growth phase relative to the normal 500-iteration run.

All evaluation fields were compared across the 32 benchmark inputs, including
sparse incidence, constraints, trips, financial allocations, service details,
and repaired genomes. Numeric comparisons used relative tolerance `1e-13` and
absolute tolerance `1e-8`; discrete results matched exactly. These are local
measurements, not runtime guarantees for other regions or larger graphs.

To measure the current evaluator through the existing CLI:

```sh
.venv/bin/python -m rail_gen benchmark \
  --config runs/europe-live/run_config.json --profile europe \
  --cache-dir runs/railgen-cache --evaluations 200
```

The CLI uses its own random-design workload and reports median/p95; its numbers
are not directly comparable to the density groups above.

## Validation and remaining costs

`tests/test_railgen_performance.py` covers frequency-dependent path switching,
repeated assignments, rejected services, shared tile capacities, different track
counts, concurrent evaluations, empty demand, and analysis result reuse. Existing
service-model, search, map, CLI, and reporting tests also cover these paths.

Passenger shortest paths and predecessor reconstruction still scale with the
number of OD markets and expanded service nodes. Floyd–Warshall, full-genome
neighborhood copies, detailed evaluation caching, and synchronous image rendering
remain opportunities for larger workloads. This change preserves the search
policy, routing tie behavior, public evaluation result fields, and worker settings.
