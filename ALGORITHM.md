# Railway network optimization

The default solver (`solver_method="annealing"`) searches over station sites,
physical alignments, all-stop service patterns, and departure frequencies. Each
proposed design is evaluated by HiGHS, which chooses train technology and assigns
continuous passenger flows. Growth uses simulated annealing; after coverage and
connectivity goals are met, a cost-reduction search protects a fixed service
baseline. This is a heuristic, with no global optimality guarantee.

This document describes the implementation in `train_optimizer/`. Distances are
km, times are hours unless stated otherwise, demand is directional trips per
representative operating hour, and monetary values are euros in constant prices.
Configuration values come from `config.json`; named profiles and CLI overrides
can change the defaults.

## 1. Fixed markets, station candidates, and demand

### Population and station capture

Population cells are filtered by the bounding box and country polygons. Eligible
seed cells meet `density_threshold`. Seeds are chosen greedily by previously
unrepresented population within `market_radius_km`, weighted by

$$
\exp\!\left[-\frac12\left(\frac{d}{R/3}\right)^2\right],
\qquad R=\texttt{market\_radius\_km}.
$$

Each selection must meet the minimum marginal population and seed-spacing
requirements; at most `max_cities` seeds are retained. City records supply labels,
not seed positions. A seed with density $\delta_i$ has catchment radius
$r_i=\texttt{center\_coefficient}\sqrt{\delta_i}$.

Population cells may be subdivided into area-weighted samples. Each sample belongs
to its nearest seed, provided it meets the density threshold and lies within that
seed's radius. These disjoint market zones are fixed before optimization. Let
$P_i$ be the population assigned to market $i$.

Prepared station sites include a center selected by `center_method` and nearby
population sample positions, subject to the site-count and separation rules. A
station $s$ in market $i$ captures only residents of that market lying within its
station radius. Write their population as $p_s$. Its uptake and capture fraction are

$$
u_s=p_{\max}\left(1-e^{-a r_u^\alpha}\right),
\qquad \gamma_s=\frac{p_s}{P_i}u_s,
$$

where $r_u$ is `uptake_reference_radius_km` when positive, otherwise the catchment
radius. Mean population-weighted access distance divided by `access_speed_kmh`
gives access time $A_s$. The access multiplier $a_s$ compares this distance with
the best achievable mean distance for the same residents, using the configured
normalization floor. A design may open at most one site per market. Market
populations, sites, access quantities, and capture fractions stay fixed.

### Directional demand

For distinct markets $i,j$ within `maximum_trip_distance_km`, define

$$
w_{ij}=\frac{P_j}{\max(d_{ij},d_{\min})^\beta}.
$$

Here $d_{ij}$ is the distance between seed coordinates. The two demand modes are

$$
D_{ij}=\begin{cases}
\texttt{trips\_per\_person\_hour}\,P_i\,
 w_{ij}/\sum_k w_{ik},&\text{production},\\
\texttt{gravity\_scale}\,P_iw_{ij},&\text{gravity}.
\end{cases}
$$

The production sum includes eligible destinations of $i$. Demand is fixed;
station capture subsequently bounds usable trips by $D_{ij}\gamma_s\gamma_t$.

## 2. Fixed rail-eligible needs

Coverage uses a network-independent optimistic need bound, including markets
that have no station in the current design. Let $f_{\max}$ be the highest allowed
frequency satisfying the minimum-service and track-throughput limits. Let
$v_{\max}$ be CR speed, multiplied by `hsr_speed_ratio` if HSR is available.
For every prepared endpoint pair $s\in i,t\in j$, calculate

$$
\underline T_{st}=A_s+A_t+
 \frac{0.99d^{\mathrm{geo}}_{st}}{v_{\max}}+\frac{1}{2f_{\max}}.
$$

The 0.99 distance margin avoids excluding trips solely because spherical and
ellipsoidal distances differ. Define the endpoint-specific journey limit

$$
T_{ij,st}=\min\left\{
T_{\max},\ A_s+A_t+\frac{1}{2f_{\min}}+
 \kappa\frac{d_{ij}}{v_{\mathrm{CR}}}\right\},
$$

omitting the second term when `maximum_journey_detour_ratio` ($\kappa$) is zero.
Here $f_{\min}$ is the smallest configured frequency meeting the minimum service
frequency. Then

$$
n_{ij}=D_{ij}\max_{s\in i,t\in j}
 \left\{\gamma_s\gamma_t\,
 \mathbf1[\underline T_{st}\le T_{ij,st}]\right\},\qquad
N=\sum_{ij}n_{ij},\qquad N_i=\sum_j n_{ij}.
$$

An empty maximum is zero. If there is no feasible frequency, all needs are zero.
The objective normalization is $\bar N=\max(1,N)$. These quantities are computed
using all prepared station candidates and never shrink when a station closes or
moves. They incorporate uptake, capture, and best-case journey screening, but
ignore actual infrastructure, shared capacity, and incompatible station choices.
They are optimistic planning bounds, not forecasts or all regional travel demand.

With the default `needs` metric, zero $N$ stops the search with
`no_eligible_needs`, without claiming complete satisfaction.

## 3. Design and physical routing

A design is a collection of ordered station lists, physical edge IDs between
successive stops, and frequencies $f_l$ (departures/hour/direction). Lines cannot
repeat a market or exceed `maximum_line_stops`. Stations and edges not used by
services are removed. Services share infrastructure only by using the same
physical edge ID; a geometric crossing or overlap creates no junction or reuse.

Routing is deferred until an endpoint pair is needed. Both straight-line endpoint
distance and retained route length must satisfy `maximum_edge_km`. The router:

1. Integrates cost along the direct WGS84 geodesic.
2. Searches the H3 grid with an any-angle Theta* variant, comparing neighbor and
   visible-parent connections using construction cost.
3. Keeps the valid direct route and a terrain alternative if the latter is
   cheaper by more than one euro, or the direct route is invalid.
4. Caches alternatives and failed pairs for the run.

Blocked cells, optional land geometry, raster coverage, and
`terrain_search_limit` restrict routing. This finite search is approximate.

### Terrain and water cost

For local density $\delta$, the unit construction rate is

$$
c(x)=b(x)+c_u[\delta-\delta_u]_+,\qquad [z]_+=\max(0,z).
$$

The baseline $b$ is at least `base_track_cost_per_km`. Water uses at least
`water_track_cost_per_km`; other named terrain uses the configured interpolated
low/high rate in millions of euros/km. Explicit H3 terrain multipliers multiply
the whole rate. RGB relief-map classification is a scenario approximation, not
an engineering slope model. Raster rates are sampled on the alignment; H3 cells
supply population density. Missing population implies the configured optimistic
base-cost assumption, not evidence of empty land.

Ordinary samples contribute length times local rate. For each uninterrupted
sampled water run of length $L$, its summed local contribution $B$ becomes

$$
K_{\mathrm{water\ run}}=B\exp\!\left(\min\{100,L/g_w\}\right),
\qquad g_w=\texttt{water\_cost\_growth\_km}.
$$

Runs continue across polyline vertices and reset on land; separate physical
edges are priced separately. Sampling spacing is bounded by both routing sample
settings and one third of an average H3 edge. Small waterways can be missed.
Water is included once in track capital, and therefore in maintenance and lifecycle
cost; it is not an additional duplicate asset charge.

## 4. Evaluate a proposed design

The outer search fixes station sites, physical edges, stop patterns, and
frequencies. It **does not fix train technology**: HiGHS chooses one available
CR/HSR variant per proposed stop pattern and any required HSR track upgrades.
Passenger and connectivity flows are continuous. Thus evaluation is generally a
small MILP; with technology choices eliminated it reduces to an LP.

### Services, fleet, and infrastructure

For technology $t$, let speed be $v_t$ and seats be $S_t$. HSR has the configured
speed multiplier, twice the CR seats, and the configured fare and operating-cost
multipliers. For line length $d_l$, stop count $m_l$, dwell time $t_d$, and
turnaround time $t_r$ in hours,

$$
R_{lt}=\frac{d_l}{v_t}+(m_l-2)t_d,\qquad
F_{lt}=\max\{1,\lceil2f_l(R_{lt}+t_r)\rceil\}.
$$

Fleet capital is $F_{lt}$ times `train_capital_cost` for either technology. Annual
operating hours are $h_y=(\text{end hour}-\text{start hour})\times\text{days/year}$.
Annual line operating cost is

$$
O_{lt}=2f_lh_y\left(d_l c_{\mathrm{train\ km}}\mu_t+
 c_{\mathrm{departure}}\right),
$$

where $\mu_t$ is 1 for CR and `hsr_opex_ratio` for HSR. The multiplier applies to
the kilometre cost, not the departure charge.

Each edge pays ordinary construction capital once. Any HSR service on it requires
one shared upgrade costing $(\texttt{hsr\_capex\_ratio}-1)$ times ordinary edge
capital. CR can use upgraded track. Both ordinary and upgrade capital incur the
maintenance fraction. Stations have fixed capital and annual operating cost.
For each edge $e$, shared train throughput requires

$$
\sum_{l:e\in l}f_l\le\texttt{track\_trains\_per\_hour\_per\_direction}.
$$

### Passenger assignment

For OD commodity $ij$, let $q_{ij,st}$ be completed trips between chosen endpoint
stations $s,t$. Let $g_{ij,a}$ be passengers using ride arc $a$: an ordered pair
of stops on a selected line/technology/frequency. A ride may pass intermediate
stops without another boarding. Flows obey

$$
0\le q_{ij,st}\le D_{ij}\gamma_s\gamma_t x_s,\qquad
q_{ij,st}\le D_{ij}\gamma_s\gamma_t x_t,\qquad
\sum_{st}q_{ij,st}\le D_{ij},
$$

where $x_s$ indicates an open station. At each station, outgoing minus incoming
ride flow equals endpoint injections minus withdrawals. Across every line
segment, separately in each direction, total commodity flow is bounded by
$S_tf_l$ for its selected technology.

A ride spanning $k$ successive segments and distance $d_a$ has time

$$
t_a=\frac{d_a}{v_t}+(k-1)t_d+\frac{1}{2f_l}+t_{\mathrm{transfer}}.
$$

Define completed trips $Q$ and total passenger-hours $H$ by

$$
Q=\sum_{ij,st}q_{ij,st},\qquad
H=\sum_{ij,st}(A_s+A_t-t_{\mathrm{transfer}})q_{ij,st}
 +\sum_{ij,a}t_a g_{ij,a}.
$$

Subtracting one transfer penalty per completed trip makes the first boarding
transfer-free. Every boarding incurs half-headway waiting; staying aboard a
through service incurs neither another wait nor another transfer. Transfers do
not count as additional completed trips.

The journey constraint is **an OD-average linear constraint**, not a separate
maximum for each individual passenger path:

$$
\sum_{st}(A_s+A_t-t_{\mathrm{transfer}}-T_{ij,st})q_{ij,st}
 +\sum_a t_a g_{ij,a}\le0\qquad\text{for each }ij.
$$

Revenue is charged on actual ridden passenger-kilometres, including transfers
and route detours, with the technology's fare multiplier $\phi_t$:

$$
R=h_y\,\texttt{ticket\_per\_km}\sum_{ij,a}d_a\phi_t g_{ij,a}.
$$

## 5. Lifecycle finance and the fixed objective

For appraisal horizon $Y$, discount rate $r$, and asset lifetime $L_b$, define

$$
A=\sum_{y=1}^Y(1+r)^{-y},\qquad
J_b=\{kL_b:k\ge1,\ kL_b<Y\},\qquad
\ell_b=\max(\{0\}\cup J_b),
$$

$$
z_b=\max\left\{0,\frac{L_b-(Y-\ell_b)}{L_b}\right\},\qquad
\psi_b=1+\sum_{y\in J_b}(1+r)^{-y}-z_b(1+r)^{-Y}.
$$

For selected capital $K_b$ and annual recurring cost $O_b$, equivalent annual
lifecycle cost is

$$
C=\sum_b\left(\frac{K_b\psi_b}{A}+O_b\right).
$$

Assets use track, station, or fleet lifetimes. Replacements occur strictly before
the terminal year, with straight-line residual value discounted at year $Y$.
`accounting_years` affects the undiscounted cash-flow report, not this objective.

Let $m$ be `financial_margin` and $\Pi$ be `minimum_horizon_profit` (a present-value
target despite the historical name). Annual-equivalent financial shortfall is
represented by a nonnegative slack $B$ satisfying

$$
B\ge(1+m)C+\Pi/A-R.
$$

With positive shortfall weight its optimal value is the positive part of the
right-hand side. The financial score is

$$
G=w_{\mathrm{cost}}\frac{C}{10^8}
 +w_{\mathrm{shortfall}}\frac{B}{10^8}.
$$

Profitability is soft. When enabled, upfront capital $\sum_bK_b\le\texttt{upfront\_budget}$
and lifecycle subsidy $C-R\le\texttt{maximum\_annual\_subsidy}$ remain hard constraints.
Zero disables either ceiling.

Let $M$ be the number of markets, $\bar P=\max(1,\sum_iP_i)$, and $o_s$ be station
offset from its prepared center. The service score is

$$
U=\frac{-w_{\mathrm{served}}Q+w_{\mathrm{time}}H}{\bar N}
 -w_{\mathrm{population}}\frac{\sum_s p_sx_s}{\bar P}
 +\frac1M\sum_sx_s\left[
 w_{\mathrm{center}}\frac{o_s}{\max(1,\texttt{station\_offset\_km})}
 +w_{\mathrm{access}}(a_s-1)\right].
$$

Weight subscripts correspond to configuration names (`served_weight`, etc.;
$w_{\mathrm{shortfall}}$ is `profit_shortfall_weight`). The final fixed objective
is $J=U+G$, minimized. Passenger benefits use $\bar N$, not raw demand or the
unscreened capturable-demand bound.

## 6. Growth: coverage, proposals, and acceptance

### Hard requirements and exploratory relaxation

Let $Q_i$ be completed trips originating at market $i$, and $x_i=\sum_{s\in i}x_s$.
Final evaluation enforces

$$
Q\ge gN,\qquad \sum_i x_i\ge\lceil cM\rceil,\qquad
Q_i\ge oN_i,\qquad Q_i\ge sx_iN_i.
$$

Here $g,c,o,s$ are the global, market, origin, and selected-origin minimum service
fractions respectively. Full connectivity is also enforced when configured.

Exploratory evaluation relaxes these four service floors and final connectivity.
Station exclusivity, service/infrastructure consistency, capacities, journey
limits, budget, and subsidy constraints remain hard. The empty railway is a
special initial state. After assignment, coverage shortfall is measured as

$$
V=\left[g-\frac{Q}{\bar N}\right]_+
 +\left[c-\frac{\sum_ix_i}{M}\right]_+
 +\frac1M\sum_{i:N_i>0}\left(
 [o-Q_i/N_i]_+ +[sx_i-Q_i/N_i]_+\right).
$$

### Coverage controller

The default phase metric is $h=\operatorname{clip}(Q/N,0,1)$. Alternatives are
selected-market share, selected covered-population share, and legacy `demand`
coverage $Q/\max(1,N_{\mathrm{capture}})$, where

$$
N_{\mathrm{capture}}=\sum_{ij}D_{ij}
 (\max_{s\in i}\gamma_s)(\max_{t\in j}\gamma_t).
$$

All denominators remain fixed and reported coverage is clamped to $[0,1]$.
Opening a station alone satisfies no passenger need.

Let $\tau$ be `annealing_consolidation_coverage` and
$h_I=\min(\texttt{annealing\_integration\_coverage},\tau)$. Default thresholds are
30% for integration and 75% for consolidation. At the current walk's coverage,

$$
p=\min(1,h/\tau),\quad
\lambda=\lambda_0+(1-\lambda_0)p,\quad
T=T_0(T_{\mathrm{end}}/T_0)^p,
$$

$$
\eta=\operatorname{clip}\left(\frac{h-h_I}{\tau-h_I},0,1\right).
$$

If $h_I=\tau$, $\eta$ steps from zero to one at $\tau$. Temperature and financial
pressure depend on coverage, not elapsed time or iteration count.

For a design $X$, let $L(X)$ be the largest track component's share of selected
covered population. Use station counts when selected population is zero, and
$L=0$ for the empty railway. With current $\lambda,\eta$ held fixed, energy is

$$
E(X)=U(X)+\lambda G(X)
 +w_{\mathrm{coverage}}\{V(X)+[\tau-h(X)]_+\}
 +w_{\mathrm{connect}}\eta[L_*-L(X)]_+
 -w_{\mathrm{network}}\eta w_{\mathrm{served}}Q(X)/\bar N.
$$

$L_*$ is `annealing_connectivity_target` (default 0.90); the other weights are
`annealing_coverage_penalty`, `annealing_connectivity_penalty`, and
`annealing_network_effect_weight`. Evaluation optimizes $U+\lambda G$ plus the
extra completed-trip reward; coverage and connectivity penalties are computed
afterward. The current state's stored allocation is not reoptimized on every
comparison, but both energies use the same current $\lambda,\eta$.

For $\Delta=E(X')-E(X)$, accept with probability

$$
\Pr(\mathrm{accept})=\begin{cases}
1,&\Delta\le0,\\
e^{-\Delta/T},&\Delta>0.
\end{cases}
$$

### Proposal selection

Before random growth, the search attempts targeted repairs for selected origins
below their required service share. Otherwise it samples eligible move families,
using `annealing_action_fraction` and phase-dependent priorities:

| Family | Proposed change |
|---|---|
| Regional seed | Add a two-stop service between unserved markets |
| Frontier | Attach an unserved market to an active region |
| Bridge | Connect different existing components |
| Extend | Add a market at a service endpoint |
| Frequency | Change a service's configured departure frequency |
| Through | Replace endpoint-connected services with one all-stop service |
| Move | Relocate a station and all its incident services to another prepared site |

Exploration favors seeds and frontier growth; integration favors bridges and
through services. Geographic novelty reduces seed preference near existing
infrastructure. Bridge weighting considers capturable demand across whole
components, so bridge endpoints need not have direct OD demand. Duplicate
existing two-stop market pairs are excluded. A new regional seed must actually
carry passengers between its newly added markets in the assignment model.
Inapplicable proposals are no-ops and still count toward bounded proposal limits.

Regional candidate scans use `annealing_candidate_pair_limit`; zero enables full
scanning. A cheap terrain-screened base weight is

$$
W_{ij}=\frac{D_{ij}}
 {\max\{1,\widehat K_{ij}/\max(1,c_{\mathrm{base}}d_{\min})\}^{\xi}},
\qquad \xi=\texttt{annealing\_proposal\_cost\_exponent}.
$$

Capital estimates use market-center distance and, when enabled, 16 geodesic
midpoint terrain samples. They avoid full routing and omit detailed feasibility
and continuous-water-run premiums. Estimates are cached on demand. Exponent zero
restores raw demand weighting. Regional novelty and component-demand factors
further modify selection.

The generic proposal helper also implements removal, station closure, split,
alignment change, bypass, and hub insertion. These are not all sampled by the
default exploration/integration controller; consolidation has its own neighborhood.

### Validated incumbent versus exploratory walk

Every surviving nonempty proposal is separately evaluated with the original hard
requirements and $J=U+G$. If full connectivity is required, individual connected
components are tested as incumbent candidates while the broader walk is retained.
Repeated final designs reuse the prior check. Candidate selection occurs even
if annealing subsequently rejects the exploratory proposal.

Incumbents are compared lexicographically: maximize $\min(h,\tau)$ first, then
minimize $J$ when capped coverage is equal. Only validated incumbents are published
as feasible checkpoints. This prevents a cheap, narrow network from blocking
progress toward the coverage target.

## 7. Protected cost consolidation

Consolidation begins only when **both** the current walk and validated incumbent
reach $\tau$, meet largest-component share $L_*$, and, when required, have exactly
one component. The walk resets to that incumbent. Baseline total trips $Q_0$,
origin trips $Q_{i0}$, and mean journey time $\bar t_0=H_0/Q_0$ remain fixed for
the entire cost stage.

With $\rho=\texttt{annealing\_retained\_ridership}$ and
$\epsilon=\texttt{annealing\_journey\_time\_allowance}$, every candidate must satisfy
all original hard requirements and

$$
Q\ge\rho Q_0,\qquad Q_i\ge\rho Q_{i0},\qquad
H\le(1+\epsilon)\bar t_0Q,\qquad h\ge\tau.
$$

The coverage floor is encoded linearly using the selected metric's fixed
denominator. Connectivity gates must also remain satisfied.

The outer objective is now to reduce equivalent annual lifecycle cost $C$.
Within each proposed design, technology and passenger assignment still minimize
the original weighted objective $J$, subject to the protections; evaluation does
not directly solve $\min C$ over technologies. Accept only validated candidates
whose resulting $C$ decreases by more than $\max(0.01,10^{-9}|C_{\mathrm{current}}|)$.
There is no annealing acceptance of cost increases in this stage.

A bounded neighborhood includes cheaper cached alignments (switching every user),
reuse of existing multi-edge paths, insertion of existing hubs, station relocation,
through services, lower frequency, removal, and compound branch consolidation.
The compound move reroutes two branches through one prepared station to create a
shared trunk, potentially opening that station.

Candidates are ranked by cheap saving estimates with family quotas and capped by
`annealing_consolidation_candidates`. Selection samples move families and favors
higher-ranked candidates within each family. Untested candidates stay queued;
terrain routes are materialized only when selected. Every accepted improvement
rebuilds the neighborhood. Exhaustion proves only that this bounded neighborhood
found no further improvement.

## 8. Optional gravity initialization and faster evaluation

`gravity_heuristic=true` starts a fresh run with market pairs ranked by
$P_iP_j/\max(d_{ij},d_{\min})^\beta$, independently of the demand model. It adds
the cheapest cached alignment at minimum frequency, retaining active station
sites and otherwise preferring maximum capture. During this stage HSR is disabled
and valid growth proposals bypass the annealing acceptance test. Initialization
ends when completed trips reach `gravity_heuristic_demand_threshold` times $N$,
or the ranked pairs run out. Resume skips this initialization.

`speed_up=true` limits exploratory solves to two seconds with a relative gap of
at least 5%, and supplies matching prior technology choices as a warm start.
Timeouts without a validated solution leave proposals unevaluated. Final and
protected evaluations retain the configured accuracy and evaluation budget.

## 9. Stopping, outputs, and other solvers

In `annealing_budget_mode="goals"`, growth has no total runtime or proposal cap.
`solver_time_limit` is the per-evaluation limit. The search continues until the
coverage/connectivity gates permit consolidation and its bounded neighborhood is
exhausted, or until interruption. Unreachable goals can therefore run indefinitely.

In `bounded` mode, the total search deadline and `annealing_iterations` cap apply;
individual solves receive the remaining time. `--time-limit` selects bounded
mode unless `--budget-mode goals` is explicitly supplied. Inputs are prepared
before the annealing search timer starts.

The live map and animation show accepted growth states marked exploratory and
not finally feasible. `exploration.json` and `exploration.png` preserve the broader
exploratory network. `best_feasible.json` and `checkpoint.npz` contain validated
incumbents only. At termination the main map selects the incumbent if available;
otherwise it retains a labeled partial result and reports search exhaustion.
Resume checks input/configuration signatures, model fingerprints, and feasibility;
it warm-starts from the saved incumbent, rather than restoring the full random
walk and random-generator state.

`solver_method="milp"` instead prepares a finite corridor and line pool and lets
HiGHS jointly select stations, tracks, technology, frequency, and flows under the
original hard requirements and objective $J$. Its optimality claims apply only to
that generated candidate model. `solver_method="heuristic"` uses prepared
candidates and a separate heuristic design search. Neither is the default
on-demand growth/consolidation algorithm described above. Failure of either
heuristic search to find a valid railway does not prove infeasibility.

Implementation references: `seeds.py` and `geography.py` prepare markets, demand,
and routes; `needs.py` fixes eligible needs; `model.py` builds assignment and
technology constraints; `finance.py` supplies lifecycle factors; `annealing.py`,
`network_growth.py`, `sampling.py`, and `consolidation.py` implement the search;
`runner.py` and `outputs.py` manage execution and publication.
