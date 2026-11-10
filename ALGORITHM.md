# Railway network search

The default search starts with no railway. It first discovers useful regional
services, then joins regional networks, then cuts costs. It proposes one change
at a time, assigns passengers, and decides whether to keep the change. Early rounds favor passenger service. Later rounds minimize lifecycle cost while
protecting the ridership and journey times of the best validated growth network.

This is simulated annealing: a worse change can sometimes be accepted, especially
early on. This helps the search leave a local optimum. It does not guarantee the
best possible railway.

## 1. Prepare markets, without routing every pair

Read population and terrain settings. Select population-based market seeds and
candidate station sites. Assign residents to fixed market catchments. Estimate
fixed directional demand between markets, using the configured production or
gravity demand model. Station capture fractions limit the usable demand at both
ends of a trip.

Markets, demand, and station choices stay fixed during the search. Station moves
choose among these prepared sites; they do not move to arbitrary coordinates.
Unlike the older search, this stage does not run any terrain routes or enumerate
a pool of service lines.

## 2. Represent the railway

A service contains an ordered list of station sites, its physical track segments,
and departures per hour in each direction. The active services determine which
stations and tracks are open. Unused infrastructure is removed automatically.
Each market can use at most one station site.

Different services can share the same cached track segment. Track construction
and maintenance are charged once. Each service has its own operating costs and
fleet, and all services consume the shared segment's train capacity. Merely
crossing or overlapping on a map does not create shared track or a junction.

All services stop at their listed stations. Required trainsets are

$$
 n_l=\left\lceil 2 f_l\left(\frac{d_l}{v}
       +\frac{(s_l-2)t_{dwell}}{60}+\frac{t_{turn}}{60}\right)\right\rceil,
 \qquad n_l\geq 1.
$$

Here, frequency `f` is departures/hour/direction, `d` is line length in km,
`v` is km/hour, `s` is number of stops, and dwell/turnaround times are minutes.
Adding trains means changing frequency and calculating the corresponding fleet.

## 3. Propose a small change

Each round samples an action; it does not enumerate every possible next network.

| Action | Change |
|---|---|
| Add a corridor | Choose a market pair using demand and estimated construction cost; add stations, track, and an initial service together |
| Extend | Add a new market to the end of a service |
| Remove a service | Close a line and discard infrastructure no other service uses |
| Close a station | Remove its incident services and unused infrastructure |
| Change frequency | Choose another configured departure frequency |
| Through service | Join two services meeting at an endpoint, replacing both with one train service |
| Split | Replace a through service with two services meeting at an intermediate stop |
| Move a station | Choose another site in the same market and update every incident service |
| Change alignment | Switch a segment between its cached route alternatives |
| Bypass | Remove an intermediate stop and connect its neighbors directly |
| Route through a hub | Replace one segment by two segments through an existing station, allowing track reuse |

Coverage, rather than iteration count, selects the growth behavior:

| Phase | Default entry condition | Preference |
|---|---|---|
| Exploration | Less than 30% passenger-need satisfaction | Seed useful services between unserved markets in new regions |
| Integration | At least 30% passenger-need satisfaction | Join regional networks and unlock onward passenger journeys |
| Consolidation | At least 75% passenger-need satisfaction and sufficient connectivity, with a validated baseline | Cut lifecycle cost while protecting service |

Exploration usually chooses an unserved–unserved pair, with a geographic novelty
factor that reduces preference for another corridor beside the existing network.
It can also attach an unserved market to an existing component. Integration
usually samples bridges between components, then frontier expansion. The bridge
weight uses all capturable OD demand across the two components, not just demand
between its endpoints. Bridge endpoints can have zero direct OD demand.
Existing two-stop service pairs are excluded from regional additions. A smaller
share of proposals adjusts frequency or creates through trains. Selected origins
that miss required service shares still receive targeted repairs.

Regional seeds must actually serve passengers in the LP. Separate exploratory
components are allowed even when the final network must be connected. Final
feasibility checks keep the original connected-network requirement. A connected
component can be saved as a feasible incumbent while the broader exploratory
walk continues; this does not discard the other regions from exploration.

An inapplicable action is a no-op and still counts toward the proposal limit.
Lines cannot repeat a market or exceed `maximum_line_stops`.

The later search uses a ranked, bounded cost-saving neighborhood instead of
random growth actions. It also switches all users of a segment to a cheaper
cached alignment when that improves the protected railway. It tests reuse of existing multi-segment paths, insertion
of an existing hub, relocation to another prepared station site, joining train
services, frequency reductions, removals, and compound branch consolidation.
The last action inserts a prepared intermediate station into two branches at
once, creating one shared trunk and two outgoing segments. A move that requires
new terrain routing is only materialized when its turn arrives.

## 4. Route only connections that are needed

Reject endpoint pairs whose straight-line distance exceeds `maximum_edge_km`.
When a proposal first needs another pair of station sites:

1. Evaluate a direct geodesic alignment.
2. Search the H3 terrain grid with an any-angle Theta* search for a cheaper
   construction alignment.
3. Retain the valid direct alignment and a cheaper terrain alternative, if found.
4. Cache the results, including failures, for the remainder of the run.

Costs include the configured terrain and urban-density assumptions. Blocked land
and the terrain search limit still apply. This is an approximate route search.
Cached station coordinates and routing settings are fixed within the run.
Moving a station uses another endpoint key. Through-service and frequency changes
usually need no new terrain search.

## 5. Assign passengers and check the proposal

Build a model containing the proposed tracks and services, with only its active
station choices and the fixed demand. Passenger normalization still uses the
full set of prepared station choices. Only the chosen frequency of each service enters this model. Fix every discrete decision. HiGHS then solves only
the continuous passenger and connectivity flows. This retains seat capacity,
track throughput, access, waiting, transfers, journey limits, and financial
accounting from the existing model.

During exploration, minimum passenger and market coverage requirements are
replaced by the penalty below, and final connectivity is temporarily relaxed.
One site per market, physical/passenger capacity, journey limits, upfront budget,
and any configured subsidy ceiling remain hard requirements. The empty board is
a special exploratory state with no passengers or infrastructure.

The passenger LP minimizes the scheduled service-and-financial objective; coverage
shortfalls are measured afterward for the annealing decision. If the exploratory
LP is infeasible or fails validation, reject the proposal.

Each nonempty proposal that survives this evaluation is also checked against
the original coverage requirements and final objective. When final connectivity
is required, its individual connected components are evaluated as incumbent
candidates; repeated unchanged component designs reuse the previous check. Publish it only if this
second evaluation passes full validation and improves the best final score.
The live map and animation show accepted exploratory states while growing.
They are marked `exploratory=true` and `feasible=false`, because final coverage
requirements may still be unmet. The first one-, two-, and three-service frames
are queued explicitly so rendering does not skip directly to a full network.
`best_feasible.json` and `checkpoint.npz` contain only the validated incumbent.
At termination, the main map switches to that incumbent, if one exists; otherwise
it retains a clearly labelled partial network and reports search exhaustion.
`exploration.json` and `exploration.png` preserve the latest broader exploratory
network, including unfinished connections. They remain available when the final
map selects a smaller fully feasible connected component.

## 6. Complete scoring formula

Smaller scores are better. All objective weights below are the corresponding
configuration values, such as `served_weight` or `cost_weight`.

Define these fixed quantities:

- `M`: number of markets.
- `D`: total potential directional trips/hour; use `D0 = max(1, D)` in ratios.
- `D_i`: potential trips/hour originating in market `i`.
- `N = max(1, sum_od demand_od * best_capture_origin * best_capture_destination)`:
  a fixed normalization for passenger benefits.
- `P = max(1, sum_i market_population_i)`.
- `A = sum_(y=1..Y) (1+r)^(-y)`: the appraisal annuity factor.

For a proposed railway and its passenger allocation, define:

- `q`: completed trips/hour, counted once per trip, including trips with transfers.
- `H`: passenger-hours/hour, including access, running, intermediate dwell,
  half-headway waiting at each boarding, and the transfer penalty after the first
  boarding. Remaining aboard a through train adds no new wait or transfer.
- `x_s`: 1 if station choice `s` is open, otherwise 0.
- `p_s`: population within that station's catchment.
- `o_s`: station offset from its market center, in km.
- `a_s`: access multiplier, the mean access distance divided by the best possible
  mean distance for the same residents, with the configured normalization floor.
- `R`: annual fare revenue. Each completed OD trip pays OD distance times
  `ticket_per_km`, multiplied by annual operating hours.

The service part of the score is

$$
 U=\frac{-w_{served}q+w_{time}H}{N}
   -w_{population}\frac{\sum_s p_s x_s}{P}
   +\frac{1}{M}\sum_s x_s\left[
      w_{center}\frac{o_s}{\max(1,\text{station\_offset\_km})}
      +w_{access}(a_s-1)\right].
$$

For each selected asset `a`, let `K_a` be its initial capital, `O_a` its annual
operating/maintenance cost, and `L_a` its lifetime. Track, stations, and fleet use
their configured lifetimes. For appraisal length `Y` and discount rate `r`, let

$$
 J_a=\{L_a,2L_a,\ldots\}\cap[1,Y),\qquad
 t_a=\max(\{0\}\cup J_a),\qquad
 z_a=\max\left(0,\frac{L_a-(Y-t_a)}{L_a}\right),
$$

$$
 F_a=1+\sum_{y\in J_a}(1+r)^{-y}-z_a(1+r)^{-Y},\qquad
 C=\sum_a\left(\frac{K_a F_a}{A}+O_a\right).
$$

`C` is equivalent annual lifecycle cost, including replacements and terminal
residual value. Let `m = financial_margin` and `Pi = minimum_horizon_profit`,
which is a present-value target despite its historical setting name. Then

$$
 B=\max\left(0,(1+m)C+\frac{\Pi}{A}-R\right),\qquad
 G=w_{cost}\frac{C}{10^8}+w_{shortfall}\frac{B}{10^8}.
$$

This financial term rewards lower costs and smaller financial shortfalls.
It does not reward additional profit once the shortfall is zero. A positive
profit is not guaranteed: the profit target is soft. The upfront budget and
optional annual subsidy ceiling are hard constraints.

For coverage, let `q_i` be completed trips/hour from market `i`, and `x_i` indicate
an open station in that market. Let `g`, `c`, `o`, and `s` be
`minimum_served_fraction`, `minimum_city_served_fraction`,
`minimum_origin_served_fraction`, and `minimum_selected_origin_served_fraction`.
Here `D_0` and `D_i` are fixed rail-eligible needs regionally and by origin,
after endpoint capture, uptake, and best-case journey filtering.
Using `[z]+ = max(0,z)`, define

$$
 V=\left[g-\frac{q}{D_0}\right]_+
   +\left[c-\frac{\sum_i x_i}{M}\right]_+
   +\frac{1}{M}\sum_{i:D_i>0}\left(
       \left[o-\frac{q_i}{D_i}\right]_+
       +\left[sx_i-\frac{q_i}{D_i}\right]_+\right).
$$

The phase controller uses achieved passenger-need satisfaction `h` and target
`tau`, not elapsed time, proposal count, or station count. The default metric
is `needs`; the other measures remain optional compatibility settings:

$$
 h_{needs}=\frac{q}{N_{needs}},\qquad
 h_{markets}=\frac{\sum_i x_i}{M},\qquad
 h_{population}=\frac{\sum_s p_s x_s}{P},\qquad
 h_{demand}=\frac{q}{N}.
$$

`N_needs` is a fixed rail-eligible need bound computed before selecting a
network. For each directional OD market, consider every prepared endpoint station
pair. Let `gamma_a` and `gamma_b` be their capture fractions and let `A_a`, `A_b`
be access hours. A conservative best-case journey bound is

$$
 \underline t_{ab}=A_a+A_b+\frac{0.99d_{ab}^{geo}}{v}+\frac{1}{2f_{max}}.
$$

`f_max` is the highest configured frequency within the track-throughput limit.
The small distance margin avoids excluding trips because spherical and
ellipsoidal distances differ. Let `T_od,ab` be `maximum_journey_hours`, additionally
capped by the existing direct-time detour limit when enabled. Define

$$
 N_{needs}=\sum_{od}D_{od}\max_{a,b}
 \left(\gamma_a\gamma_b\,\mathbf 1[\underline t_{ab}\leq T_{od,ab}]\right).
$$

The denominator includes eligible demand involving unserved markets. Station
closure, route removal, and relocation never shrink it. Actual completed trips
must still satisfy access, capacity, transfers, service frequency, and journey
limits; opening a station alone satisfies no passenger need. Transfers count as
one completed trip. If the denominator is zero, report `no_eligible_needs` and
stop without claiming 100% satisfaction.

This evaluates rail-eligible needs after uptake and capture assumptions, not all
travel needs in the region. Total-demand served shares are no longer reported
or used for service floors. `N_needs` is optimistic: infrastructure,
shared capacity, and mutually exclusive station choices can make a high target
unattainable even after best-case journey filtering.

The market measure counts markets with an operating station/service, not just
stations discovered during preparation. Population measures residents within
selected catchments. Demand measures completed trips relative to the fixed
optimistic capturable-demand bound; it does not mean all regional trips.
All denominators are fixed, so closing stations cannot shrink the denominator.
Use zero coverage for the empty network and clamp reported ratios to `[0, 1]`.

With `tau = annealing_consolidation_coverage`, use

$$
 p=\min(1,h/\tau),\qquad
 \lambda(p)=\lambda_0+(1-\lambda_0)p,
$$

$$
 T(p)=T_0\left(\frac{T_{end}}{T_0}\right)^p,\qquad
 \boxed{E_p=U+\lambda(p)G+w_{coverage}(V+[\tau-h]_+)
 +w_{connect}\eta(h)[L_* - L]_+
 -w_{network}\eta(h)w_{served}\frac{q}{N}}.
$$

For covered-population connectivity, define

$$
 L=\max_{component\ A}\frac{\sum_{s\in A}p_s}{\sum_{s\ selected}p_s},\qquad
 \eta(h)=\operatorname{clip}\left(\frac{h-h_{integration}}{\tau-h_{integration}},0,1\right).
$$

Use the configured integration threshold capped at `tau`; if both thresholds
coincide, integration pressure steps from zero to one at `tau`. `L_*` is
`annealing_connectivity_target` (default 0.90), `w_connect` is
`annealing_connectivity_penalty` (default 10), and `w_network` is
`annealing_network_effect_weight` (default 1). When selected covered population
is zero, use station counts for `L`; the empty railway has `L = 0`.

The extra `[tau-h]+` term explicitly rewards progress toward the controller's
coverage target even if the hard minimum market requirement is zero. As coverage
grows, the extra completed-trip reward and covered-population connectivity penalty
encourage integration. Completed trips are counted once, including journeys
using multiple services. There is no flat reward for adding a transfer or just
reducing the number of components. The passenger LP includes the additional
completed-trip coefficient; the coverage and connectivity terms are measured
for acceptance after passenger assignment. Both sides of each acceptance decision
use the same current integration pressure and financial weight.

Low coverage keeps the growth score warmer and financial pressure lower even
late in the run. Reaching the target applies full financial pressure. Neither
iteration count nor elapsed time forces consolidation.
Both the current state's stored allocation and the proposal are scored using
the same current `lambda`. An action has no fixed reward of its own:

$$
 \Delta=E_p(\text{proposal})-E_p(\text{current}),\qquad
 \boxed{P(\text{accept})=
   \begin{cases}1,&\Delta\leq0\\
   \exp(-\Delta/T(p)),&\Delta>0.
   \end{cases}}
$$

During growth, first prefer valid networks with greater coverage up to `tau`;
when that capped coverage is equal, prefer the lower fixed score `U + G`.
This prevents a narrow but cheap incumbent from blocking the coverage transition.
Candidate selection is independent of whether the exploratory walk accepts it.
Exploratory frames are displayed separately and cannot become valid checkpoints.

### Cost consolidation and refinement

By default, grow until a fully validated network satisfies 75% of fixed
rail-eligible passenger needs and at least 90% of covered population belongs to
the largest component.
When `require_connected_network=true`, all selected sites must additionally be
connected before consolidation. Both the exploratory walk and the selected
validated baseline must pass these gates; a narrow connected incumbent cannot
trigger consolidation while the broader walk is still fragmented. Continue
exploration/integration and origin repairs while any gate remains unmet.
The old iteration-based `annealing_growth_fraction` and `annealing_refine_fraction`
settings have been replaced by the coverage settings below.
Let `Q0`, `q_i0`, and `t0` be this baseline's total trips/hour, per-origin trips,
and mean journey hours. These values remain fixed throughout consolidation.

Let `rho = annealing_retained_ridership` and
`epsilon = annealing_journey_time_allowance`. Subsequent designs solve passenger
assignment under the original hard requirements and these additional constraints:

\[
 Q \geq \rho Q_0,\qquad
 q_i \geq \rho q_{i0}\quad\text{for every baseline origin},\qquad
 H \leq (1+\epsilon)t_0 Q,\qquad h \geq \tau.
\]

The cost-stage design objective is

\[
 \boxed{\min C}
\]

where `C` is equivalent annual lifecycle cost defined above. Passenger allocation
within each fixed design still uses the original weighted objective, subject to
the service-protection constraints. Accept a design only if it passes those
constraints, retains the configured connectivity threshold, and reduces `C`. Financial-shortfall penalties do not replace the
original hard budget or subsidy limits. The soft financial target stays soft.
A lower-cost protected result can have a worse original `U + G` score; it is
selected deliberately by the cost-stage criterion.

Candidates are ranked by estimated avoided unique-track capital minus estimated
new track and station capital. Already selected physical segments have zero
incremental construction cost, but consume their actual combined train capacity.
After any accepted saving, rebuild the bounded neighborhood around the new design.
Move-family quotas keep track sharing from crowding out station moves and train
frequency changes. Candidate estimates only rank proposals: final decisions use
actual routed costs and full passenger validation.

Consolidation includes the final strict improvement pass. Its coverage constraint
prevents savings from reducing coverage below the transition target.
If the bounded neighborhood is exhausted, stop and report that fact. In bounded mode, a time or proposal limit can arrive first; report that limit
and the number of pending candidates.
Neighborhood exhaustion is not a proof of global optimality.

Intermediate stations and relocations currently use prepared market station
sites. Services stop at those intermediate stations. Arbitrary geographic
junctions without stations, continuous station relocation, and express trains
passing an intermediate station without stopping remain outside this model.

## 7. Settings and stopping

| Setting | Default | Meaning |
|---|---:|---|
| `solver_method` | `annealing` | Select the new search |
| `annealing_growth_probability` | 0 | Legacy single-network proposal override |
| `annealing_integration_coverage` | 0.30 | Coverage at which component integration becomes the main growth action |
| `annealing_connectivity_target` | 0.90 | Minimum covered-population share in the largest component before consolidation |
| `annealing_connectivity_penalty` | 10 | Integration penalty for missing that connectivity share |
| `annealing_network_effect_weight` | 1 | Extra completed-trip reward at full integration pressure |
| `annealing_exploration_radius_km` | 250 | Geographic novelty scale for new regional seeds |
| `annealing_budget_mode` | `goals` | Finish stages; `bounded` enforces total run budgets |
| `annealing_coverage_metric` | `needs` | Passenger-need satisfaction; optional legacy `markets`, `population`, or `demand` |
| `annealing_consolidation_coverage` | 0.75 | Satisfaction required before cost cutting |
| `annealing_retained_ridership` | 0.98 | Preserve this fraction of baseline trips globally and per origin |
| `annealing_journey_time_allowance` | 0.05 | Maximum relative increase in mean journey hours |
| `annealing_consolidation_candidates` | 80 | Bounded cost candidates per accepted topology |
| `annealing_iterations` | 1000 | Maximum proposals in bounded mode only |
| `annealing_start_temperature` | 1 | Initial willingness to accept worse scores |
| `annealing_end_temperature` | 0.01 | Final willingness |
| `annealing_initial_financial_weight` | 0.1 | Initial fraction of final financial pressure |
| `annealing_coverage_penalty` | 20 | Penalty per unit of coverage deficit |
| `random_seed` | 42 | Controls sampled actions and acceptance |

Normal annealing runs use `annealing_budget_mode="goals"`. They continue past
`annealing_iterations` and the regional `solver_time_limit` until coverage,
integration, and the bounded consolidation neighborhood are completed, or the
user interrupts. `solver_time_limit` bounds each passenger LP in this mode; it is
not a total run deadline. There is no automatic total-runtime cap, and reaching
the goals is not guaranteed when constraints or geography prevent them.
Ctrl+C preserves the feasible incumbent and latest exploration view.

`--time-limit SECONDS` selects bounded mode unless `--budget-mode goals` is also
explicitly supplied. In bounded mode, the total search deadline includes routing,
model construction, and passenger LPs; preparation and input hashing come first.
Routing checks interruption while sampling/searching. Model construction and
individual native geometry operations can finish after a cooperative deadline.
`--budget-mode bounded` uses the configured total time and proposal budgets.

The same seed gives repeatable proposal draws. Cooling depends on coverage;
wall-clock deadlines and machine speed can still affect the result. No optimum or infeasibility proof is
claimed. If no fully valid network is found, report search exhaustion.

Checkpoints contain the selected design, cached alignments available when it was
saved, the passenger vector, any fixed service-protection constraints, and
input/model fingerprints. Resume validates them
and starts a fresh cooling schedule from that best design. It is a warm start,
not an exact continuation of the random-number stream or exploratory state.

## 8. Scope and comparison

The new mode avoids upfront terrain routing of every candidate market pair and
avoids enumerating the full service pool. It does not remove every quadratic
cost: demand preparation still examines market pairs, and passenger assignment
can become expensive for a large active network. This version rebuilds small
fixed-design LPs and normally evaluates both exploratory and final allocations;
LP reuse and incremental passenger updates are possible later improvements.

`--solver heuristic` keeps the older fixed-site, precomputed-corridor search.
`--solver milp` keeps the full candidate mixed-integer model. Those modes still
route their candidate corridors in preparation and use their existing settings.
Compare algorithms using total runtime, time to first valid network, final score,
ridership, journey time, lifecycle finances, and multiple random seeds. A faster
or better regional result must be measured; it is not guaranteed by annealing.

## Europe passenger-growth profile

The `europe` profile in `config.json` emphasizes service with `served_weight=50`,
`population_weight=2`, and `profit_shortfall_weight=0.1`. New services start at
2 trains/hour/direction instead of 1. The extra growth probability is 0.65,
which makes additions approximately 71% of nonempty-state action draws. Cooling
uses temperatures 0.15 to 0.01 and initial financial weight 0.05.
The staged regional proposal policy replaces the earlier addition override.

The terrain search limit is 500 visited cells rather than 15,000. This gives more
network proposals time to run, at the cost of less thorough terrain alternatives;
the direct alignment still receives the normal terrain and feasibility checks.
The final connected-network requirement remains in place. The configured
120 seconds is a per-LP cap in goal mode, or a total search cap in bounded mode.

A single-seed check using the original run and the tuned profile, each with a
120-second search budget, gave:

| Measure | Original Europe run | Tuned Europe run |
|---|---:|---:|
| Proposals | 39 | 295 |
| Completed trips/hour | 1,000 | 27,716 |
| Markets served | 2 | 10 |
| Services | 1 | 12 |
| Mean journey hours | 4.45 | 3.33 |
| Upfront capital | EUR 5.31 billion | EUR 83.09 billion |

Both passed hard feasibility validation. Neither met the soft financial target.
These are scenario outputs, not forecasts. The tuned score deliberately favors
passengers more, so objective values across these profiles are not comparable.
The check demonstrates more useful search progress in this case, not a guarantee
across random seeds. The existing run directory was preserved.

## Consolidation validation

A 40-second Netherlands–Belgium check with the staged search showed the first
three animation frames containing one, two, and three services. Growth reached a
valid baseline; consolidation accepted 16 savings and reduced equivalent annual
lifecycle cost from approximately EUR 4.61 billion to EUR 2.35 billion. The final
network served 15 markets and about 8,484 trips/hour, retaining approximately 99%
of baseline trips. Its mean journey time remained within the configured 5%
allowance. It stopped on the time limit with 52 candidates still pending.
This is one seeded scenario check, not a guarantee for other runs. The soft
financial target was still unmet.

Regression checks separately cover the Dordrecht–Utrecht–Groningen style trunk
replacement, an existing multi-segment shared path, opening one intermediate
station to consolidate two branches, protected ridership and journey time,
checkpoint restoration with fixed protection constraints, and early exploratory
frames that cannot overwrite the validated checkpoint.

## Coverage-driven phase validation

The phase controller now reports `coverage_metric`, `coverage_fraction`,
`coverage_target`, and (at termination) `coverage_target_met`. If a budget ends
before the target is reached, bounded mode stops in exploration/integration and reports unmet coverage;
it does not force a cost-cutting phase. An incumbent that meets the original
requirements may still be saved even when this additional search target is unmet.
Some population or demand thresholds may be unattainable under catchments,
journey limits, and budgets; the program does not lower the target automatically.

A 20-second Netherlands–Belgium check switched at 75% market coverage after
about 7 seconds and finished consolidation at 80%. Tests separately verify
low-coverage growth through the last proposal, immediate consolidation at high
coverage, all three fixed-denominator metrics, and rejection of cost savings
that would cross below the coverage threshold.

## Water crossing costs and proposal selection

The RGB map's blue-dominant pixels are classed as `water`. The explicit bridge
category is removed. `water_track_cost_per_km` defaults to EUR 120 million/km.
Terrain JSON can also mark cells as water. This is the short-crossing baseline;
README.md documents the reference projects and scenario calibration.

For a sampled segment of length `d` km classified as water, its contribution is

$$
 C_{water}=d\left[\max(c_{base},c_{water})
      +c_{urban}\max(0,\text{density}-\text{density_threshold})\right].
$$

For each continuous water run of L km, sum these contributions and multiply
by exp(min(100, L / water_cost_growth_km)), with a default growth scale of 25 km.
Runs continue across polyline vertices and reset on sampled land. Sampling uses
at most water_routing_sample_km (default 0.25 km). Search segment costs and cheap
proposal estimates are local approximations; final alignments receive the full
continuous-run premium. Independent physical corridor edges are priced separately.

Explicit H3 multipliers apply to the whole bracket. Raster classification and
cost lookup use points on the alignment, while population density uses the
containing H3 cell. This prevents coastline errors caused by pricing the whole
hexagon from its center. Sampling and RGB classification remain approximate;
small waterways may be missed. Without terrain data the model cannot infer water.

Water cost is part of ordinary track capital, not an additional duplicate charge.
It therefore also affects maintenance and lifecycle accounting, and is charged
once when services share a physical track segment. Routes and selected tracks
report `water_km` and `water_capital`; network metrics expose
`total_water_track_km` and `water_track_capital`. The cost log shows the water
portion as included in track capital.

Previously the first route proposal used demand alone, so changing terrain costs
could not change the first pair sampled with a fixed seed. Proposal weights now
use a cheap construction estimate:

$$
 w_{ij}=\frac{D_{ij}}{\max(1,\widehat C_{ij}/C_{ref})^{\gamma}},\qquad
 C_{ref}=c_{base}\,\text{distance_floor_km}.
$$

Here `gamma = annealing_proposal_cost_exponent`, default 1. Zero restores raw
demand weighting. Sixteen equally spaced midpoint terrain samples along the
market-center geodesic estimate `C_hat`; urban surcharges and detailed alignment
feasibility are left to full routing. Estimates are cached once per run and
symmetric pair, and perform no terrain path searches. Connectivity filtering
still applies before sampling. A pricey crossing remains a candidate; it simply
has a lower probability than an equally busy, equally long cheap land corridor.

For a sampled direct alignment between London and Paris, with population
surcharges excluded, changing the water rate from EUR 35M/km to EUR 200M/km
increased track capital from about EUR 5.77 billion to EUR 21.28 billion, with
about 94 km classified as water by this raster. This is an illustration of the
model's sampled geometry and costs, not the length or cost of an actual crossing.

## Regional-network and stopping regression checks

Tests cover a separate regional seed under a final connectivity requirement;
a bridge with no endpoint demand unlocking onward passengers; actual passenger
counting without transfer duplication; geographic novelty; rejection of duplicate
service pairs; connectivity gates before consolidation; and saved connected
components remaining separate from the broader exploratory map.

Goal-mode tests advance a simulated clock beyond the old total deadline and run
past the old proposal cap. The search continues until interruption. A separate
test checks normal stopping after consolidation, and bounded-mode tests retain
the explicit deadline behavior. CLI tests verify that `--time-limit` selects
bounded mode and an explicit `--budget-mode goals` takes precedence.

## Passenger-need satisfaction outputs

The default config now uses `annealing_coverage_metric="needs"`. Integration
begins at 30% need satisfaction and consolidation requires 75%, plus connectivity
and original hard service constraints. Explicit regional minimum market-service
requirements remain hard constraints; they are independent of the phase metric.
Cost-stage passenger flows must retain `q >= tau * N_needs`, in addition to the
fixed baseline ridership and journey-time protections.

New metrics are `rail_eligible_needs_per_hour`, `satisfied_needs_per_hour`,
`unmet_rail_needs_per_hour`, and `needs_satisfaction_fraction`. Maps, console logs,
and cost logs label the score as passenger needs satisfied. Legacy market counts
remain descriptive outputs. A bounded Netherlands check opened all 20 candidate
markets but satisfied only 58.7% of 11,692 eligible trips/hour, so it correctly
stayed in integration. Its 6,865 completed trips/hour count completed journeys, without transfer duplication.

## Effective settings and time-limit inheritance

All values live in `config.json`: `defaults`, then optional external JSON, then
the named `--profile`, then CLI overrides. Python only defines the schema and validation. The repeated 120-second overrides have been removed from the
base JSON and standard Europe, France, Netherlands–Belgium, and UK profiles, so
editing `defaults.solver_time_limit` in `config.json` controls their inherited setting. Explicit
custom profile overrides remain effective. Startup messages and manifest
`run_settings` expose actual CLI-adjusted limits. In goal mode this is a per-evaluation
limit with no total search cap; bounded mode uses a total search deadline and
its configured proposal cap.
