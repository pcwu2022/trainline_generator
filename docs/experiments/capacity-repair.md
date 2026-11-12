# Connect-and-repair capacity experiment

Historical experiment from `experiment/connect-and-repair-capacity`.
These results predate reference elasticity, automatic track widening, and retention
of spill-only infeasible live networks. They are not current defaults or a
calibrated forecast. Refer to [the current guide](../FAST_GENERATOR.md).

The experiment sizes formations up to two coupled train units, charging full
per-unit fleet and operating costs. When ordinary integration moves exhaust,
it tests paired approaches to unserved cities and bounded sequences of extra
routes/upgrades near saturated corridors. Infeasible intermediate networks are
never published as accepted states. Ordinary defaults retain single-unit trains
and disable the package search; `configs/capacity-repair.json` enables both.

## Europe trial

```sh
.venv/bin/python optimize.py --solver pareto --profile europe \
  --config configs/capacity-repair.json --output runs/europe-capacity-repair \
  --iterations 500 --snapshot-every 50 --gif-frame-ms 500
```

Compared with the saved `runs/europe-live` run, inputs, geography cache, seed,
iteration limit and existing model assumptions were unchanged. Snapshot cadence
changed from 5 to 50 to reduce rendering overhead. The existing run was preserved.

| Metric | Saved baseline | Capacity experiment |
| --- | ---: | ---: |
| London served | No | Yes |
| Served candidate cities | 102 / 120 | 108 / 120 |
| Active components | 5 | 7 |
| Annual completed trips | 1,191.92 million | 1,750.95 million |
| Need satisfaction | 5.66% | 8.31% |
| Capacity spill | 14.92% | 14.77% |
| Annual net | −€85.98 billion | −€88.55 billion |
| Last iteration / limit | 361 / 500 | 353 / 500 |
| Final status | `coverage_incomplete` | `coverage_incomplete` |

The new network uses two-unit formations on 43 corridors. At exhaustion, the
package stage performed 719 additional evaluations, found no feasible complete
package, and recorded a minimum spill of 15.28% among topology-improving evaluated
proposals. No package was accepted in this particular full run. Serving London
and the additional ridership therefore demonstrate the paid-formation change;
they do not establish that the package search improved this full-run result.

A controlled check on the saved baseline network also confirmed that adding a
conventional London–Bristol connection changes from 18.90% spill with single-unit
operation to 4.62% with demand-sized formations of up to two units across the
network. This includes resizing existing congested corridors, not just the new
London service.

This is a mixed result: more cities and trips, but higher annual loss and more
disconnected components. The bounded package search is not a proof of national
infeasibility. It does not exhaust every combination of candidate routes.
Two-unit operation assumes suitable platforms/coupling; platform extensions and
new parallel-track capacity are not modeled. Demand and finance remain uncalibrated.

## Validation

All 42 existing fast-generator tests and three new capacity tests passed.
The new tests verify paid fleet/operating capacity, unchanged frequency limits,
formation export, validated settings, and a triangular network where neither
single connection is feasible but the two-connection package meets the spill
limit. Budget and interruption checks are also covered.

Trial outputs: `runs/europe-capacity-repair/latest.png`, `latest.geojson`,
`latest.json`, `report.md`, and `search_history.json`. The history includes
`capacity_repair` evaluation and feasibility diagnostics.
