# Railway network algorithm (simple version)

The program searches for a railway network within a finite set of candidate
stations, tracks and services. It uses a mixed-integer linear program (MILP),
solved by HiGHS.

1. **Read settings and geography.** Load `config.json`, apply the selected regional
   configuration and command-line options, and read population cells and cities.
   Keep the requested bounding box. City records supply labels only; country codes do not filter population.
2. **Create station choices.** Rank eligible population tiles by population within 25 km,
   select the strongest seeds at least 25 km apart, and retain up to `max_cities`.
   Find population-weighted central locations and nearby alternatives. Use the
   unified `market_radius_km = 25` for scoring, center evaluation, market extent
   and station catchments. Assign overlap residents to their nearest seed;
   residents beyond 25 km of all seeds do not contribute demand. Attach nearby
   city names afterward for station labels and map reference.
3. **Estimate travel demand.** Generate directional trips between cities using
   population and distance. Station capture at both endpoints limits how much
   of that demand can use rail.
4. **Create track and service choices.** Connect nearby candidate locations,
   estimate construction costs from population density and available terrain,
   and generate two-stop and selected multi-stop all-stop lines. Each line has
   discrete frequency choices and a calculated train fleet.
5. **Choose a network and assign passengers together.** The solver selects
   stations, tracks, lines and frequencies, and routes passengers through them.
   Passengers may transfer; each train segment has a capacity limit.
6. **Enforce operating requirements.** Allow at most one station option per city,
   prevent overlapping station catchments, require stations and tracks for each
   service, enforce passenger flow balance, track throughput, average journey
   limits and minimum service targets. Require a connected network when enabled.
   An optional upfront capital budget remains a hard limit.
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
