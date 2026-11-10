# Railway network algorithm (simple version)

The program searches for a railway network within a finite set of candidate
stations, tracks and services. The default heuristic chooses one station per
market and searches demand-ranked direct and through services, with shortcuts,
bypasses, pruning and frequency changes. Trees are only initial designs. It fixes all discrete choices and uses HiGHS for continuous passenger
assignment, retaining hard feasibility checks without guaranteeing optimality.
`--solver milp` runs the original full mixed-integer search. The steps below
cover the full model; heuristic mode uses a bounded three-stop service pool and
keeps its chosen station location fixed. `heuristic_iterations` limits candidate designs.

1. **Read settings and geography.** Load `config.json`, apply the selected regional
   configuration and command-line options, and read population cells and cities.
   Filter population by the requested bbox and country polygons. City records provide labels only.
2. **Discover markets and station choices.** Use eligible population tile centers
   as candidates. Greedily select up to `max_cities` seeds by newly represented
   population within `market_radius_km`, weighted by a Gaussian with scale
   `market_radius_km / 3` to favor dense urban centers. Rescore after each pick. Enforce
   independent `minimum_seed_spacing_km` (zero uses the maximum market radius).
   Set radius to `center_coefficient * sqrt(seed density in people/km²)`,
   with no floor or cap. The default coefficient 0.16136 gives Brussels about
   20 km. Use this radius for center evaluation
   and catchments. Assign residents to their nearest seed within its radius. Final
   assigned populations can fall below the target. Attach optional city labels
   afterward. Seeds and demand stay fixed while the solver chooses connections.
3. **Estimate travel demand.** Generate directional trips between cities using
   population and distance. Station capture at both endpoints limits how much
   of that demand can use rail. Regional configs use `uptake_reference_radius_km`
   to keep uptake independent of adaptive market geometry.
4. **Create track and service choices.** Connect nearby candidate locations,
   estimate construction costs from population density and available terrain,
   retain a direct geodesic and a cheaper any-angle terrain detour when found,
   and generate two-stop and selected multi-stop all-stop lines. Each line has
   discrete frequency choices and a calculated train fleet.
   The solver compares alignment variants using actual cost and journey length;
   through lines can mix direct and detouring segments and share physical tracks.
5. **Choose a network and assign passengers together.** The solver selects
   stations, tracks, lines and frequencies, and routes passengers through them.
   Passengers may transfer; each train segment has a capacity limit.
6. **Enforce operating requirements.** Allow at most one station option per city,
   allow overlapping catchments with exclusive population zones, require stations and tracks for each
   service, enforce passenger flow balance, track throughput, average journey
   limits and minimum service targets. Require a connected network when enabled.
   An optional upfront capital budget remains a hard limit.
   The Netherlands scenario requires 75% of markets, 10% of regional demand,
   5% of each selected origin's demand, and at least two trains/hour/direction
   on every selected line. An optional annual-equivalent subsidy ceiling is hard.
7. **Balance benefits, costs and profitability.** Minimize a weighted sum of
   unserved demand (equivalently, reward served trips), passenger journey time,
   station access penalties, annual-equivalent costs and profit-target shortfall.
   The default center-offset penalty is zero.
8. **Save progress and the final result.** Validate improving solutions and export
   the map, dashboard, animation, JSON, GeoJSON, CSV summaries and checkpoint.
   Stop at the time limit, interruption, or a solver conclusion. A time limit
   without a solution does not prove infeasibility.

## Lifecycle appraisal and soft profitability target

The default appraisal runs for `appraisal_years = 30` at a real discount rate
of 4%, with 2%, 4% and 6% sensitivity reports. All prices and annual revenue and
operating costs remain constant in real terms. Initial investment occurs at year
0; operations, replacements and residual values occur at year end.

Track, station and fleet lives default to 40, 50 and 25 years respectively.
These are editable scenario assumptions, not verified asset estimates. Each asset
is replaced at full original real cost at multiples of its life strictly before
the terminal year. Residual value is replacement cost times its remaining-life
fraction at the terminal year. There is no terminal-year replacement.

```text
NPV = -initial_capital + sum((revenue - operating - renewals_t)/(1+r)^t)
      + residual_T/(1+r)^T
A = sum(1/(1+r)^t, t=1..T)
required_pv_revenue = (1 + financial_margin) * pv_cost + minimum_horizon_profit
shortfall = max(0, required_pv_revenue - pv_revenue)
objective penalty = profit_shortfall_weight * (shortfall/A) / 100,000,000
```

The MILP uses equivalent annual lifecycle costs, preserving linearity. Its slack
is thousands of equivalent annual euros; reports convert it to present-value
euros. A finite penalty does not guarantee profitability. The cost objective
also uses equivalent annual lifecycle costs. The upfront capital budget stays hard.

`accounting_years = 10` still produces the previous undiscounted cash-flow report:
revenue times years minus initial capital and operating cost times years. These
legacy horizon metrics exclude renewals and residuals and no longer determine
the soft financial target. NPV, asset schedules, annual lifecycle cash flows and
discount sensitivity are exported under `metrics.lifecycle` and
`metrics.discount_rate_sensitivity`.

This is a strategic scenario model with uncalibrated assumptions. Its result is
optimal only within generated candidates and the reported solver gap.
