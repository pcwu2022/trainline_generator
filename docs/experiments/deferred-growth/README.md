# Deferred growth experiments

Status: archived and inactive. The active algorithm is restored to commit
`a773cff`, retaining the uncommitted batch-accept enhancement and its CLI/config
controls and tests. The archived experiments are not imported by the active code.

## Saved implementation

`implementation.patch` preserves the full working-tree experiment before rollback,
including revenue-plus-service growth scoring, early connectivity targets, private
bounded investment lookahead, early profit transition, related tests/documentation,
and batching. `manifest.json` records the exact base commit and patch SHA-256.

To revisit it, use a **clean separate checkout of the recorded base commit**, then
run `git apply --check /absolute/path/to/implementation.patch` and
`git apply /absolute/path/to/implementation.patch`. The patch includes batching,
so applying it over the active checkout with batching already installed will
conflict. No experimental implementation was committed or activated by this archive.
Saved run outputs have not been changed.

## Deferred idea: exploration, integration, profit

The user identified this as resembling the discarded `train_optimizer` approach;
it is recorded for possible later consideration, not approved for implementation.

- Maintain a current exploratory network separately from best-score, best-profit
  and best-connectivity checkpoints.
- During exploration, use the profit-plus-service score as a guideline rather
  than a hard positive-improvement gate. Consider normalized simulated-annealing
  acceptance of temporary score/ridership losses, cooling over an evaluation budget.
- Include corridor construction, component connections, track upgrades, timetable
  reallocations, service replacements and removals in proposal generation.
- Keep unfinished/unused prerequisite links provisional for a bounded number of
  evaluated proposals; reassess them instead of deleting them immediately.
- Integration emphasizes connecting regional networks and completing useful
  corridors, with bounded setbacks and explicit financial allowances if configured.
- Diversify or restart from alternative checkpoints when proposals are exhausted;
  transition based on actual evaluations/progress rather than publishing idle frames.
- Profit refinement starts from multiple checkpoints, prioritizes annual net profit,
  and preserves a fixed declared service baseline for each alternative.
- Physical capacity and explicitly configured construction/subsidy budgets remain
  hard. Growth-stage profitability and strictly increasing ridership need not be.

## Findings that motivated the experiments

The France network saturated Paris–Lille–Lyon approaches at the four-track ceiling.
The EUR5/trip hard score blocked further growth. Three-edit bounded lookahead
examined 104 drafts without a qualifying package; capacity repair and profit
refinement also failed to expand it. Forcing coverage could reduce both total
ridership and profit. A better search acceptance rule alone cannot represent
missing independently costed junction/approach capacity or service alternatives.

The restored commit intentionally also restores its original score, stage timing,
repair and exhaustion behavior. Only batching is retained from the uncommitted code.
