# Country profiles

Both `rail_gen/config.json` and `train_optimizer/config.json` provide the same 51 single-country profiles. Use `--profile NAME` with either optimizer. The former Netherlands/Belgium `netherlands` profile has been removed; `netherlands` now selects only `NL`, and `belgium` selects only `BE`.

Coverage includes the existing Europe profile country set, plus Armenia, Azerbaijan, Georgia, and Kazakhstan. Names use lowercase with underscores; the United Kingdom retains `uk`. Regional and diagnostic profiles (`europe`, `exploratory`, `smoke`) remain available in their respective optimizers.

Bounds are approximate geographic envelopes in `[west, south, east, north]` order. Country codes filter markets within each envelope. France includes Corsica; Denmark excludes Greenland and the Faroe Islands; Norway covers the mainland. Spain uses the mainland European map envelope (`[-9.4, 36.0, 3.4, 43.9]`), excluding the Canary Islands and North African enclaves; a rectangular envelope can still include nearby Balearic islands. Russia uses a non-wrapping envelope through 180° E, excluding its territory across the antimeridian. Other transcontinental country envelopes include their Asian territory. These profiles do not download or extend the bundled population, terrain, or boundary datasets. Countries outside those datasets, islands, and microstates may have too few candidate markets to run with the default population threshold and grid resolution.

City limits use the bundled `data/cities50000.json` catalog as a sizing proxy. Count cities matching each profile's country and bounds, then multiply by `100 / 129` (France has 129 catalog cities and retains its 100-market limit). Round values below 10 to the nearest integer and larger values to the nearest multiple of five, with a minimum of one. These are planning limits for population-derived markets, not guarantees of available markets. Microstates with no catalog cities receive a limit of one. The optimizer requires a minimum limit of two; this does not imply that microstates contain two viable markets. Regional and diagnostic profile limits are unchanged.

City limits use the bundled `data/cities50000.json` catalog as a sizing proxy. Count cities matching each profile's country and bounds, then multiply by `100 / 129` (France has 129 catalog cities and retains its 100-market limit). Round values below 10 to the nearest integer and larger values to the nearest multiple of five, with a minimum of two. These are planning limits for population-derived markets, not guarantees of available markets. Microstates with no catalog cities receive a limit of two. The optimizer requires a minimum limit of two; this does not imply that microstates contain two viable markets. Regional and diagnostic profile limits are unchanged.

| Profile | Country code | Catalog cities | City limit |
| --- | --- | ---: | ---: |
| `albania` | `AL` | 10 | 8 |
| `andorra` | `AD` | 0 | 2 |
| `armenia` | `AM` | 3 | 2 |
| `austria` | `AT` | 7 | 5 |
| `azerbaijan` | `AZ` | 14 | 10 |
| `belarus` | `BY` | 23 | 20 |
| `belgium` | `BE` | 32 | 25 |
| `bosnia_and_herzegovina` | `BA` | 7 | 5 |
| `bulgaria` | `BG` | 20 | 15 |
| `croatia` | `HR` | 7 | 5 |
| `cyprus` | `CY` | 4 | 3 |
| `czechia` | `CZ` | 22 | 15 |
| `denmark` | `DK` | 15 | 10 |
| `estonia` | `EE` | 3 | 2 |
| `finland` | `FI` | 20 | 15 |
| `france` | `FR` | 129 | 100 |
| `georgia` | `GE` | 5 | 4 |
| `germany` | `DE` | 191 | 150 |
| `greece` | `GR` | 42 | 35 |
| `hungary` | `HU` | 20 | 15 |
| `iceland` | `IS` | 1 | 2 |
| `ireland` | `IE` | 6 | 5 |
| `italy` | `IT` | 130 | 100 |
| `kazakhstan` | `KZ` | 30 | 25 |
| `kosovo` | `XK` | 9 | 7 |
| `latvia` | `LV` | 5 | 4 |
| `liechtenstein` | `LI` | 0 | 2 |
| `lithuania` | `LT` | 5 | 4 |
| `luxembourg` | `LU` | 1 | 2 |
| `malta` | `MT` | 0 | 2 |
| `moldova` | `MD` | 5 | 4 |
| `monaco` | `MC` | 0 | 2 |
| `montenegro` | `ME` | 2 | 2 |
| `netherlands` | `NL` | 57 | 45 |
| `north_macedonia` | `MK` | 9 | 7 |
| `norway` | `NO` | 15 | 10 |
| `poland` | `PL` | 101 | 80 |
| `portugal` | `PT` | 21 | 15 |
| `romania` | `RO` | 41 | 30 |
| `russia` | `RU` | 342 | 265 |
| `san_marino` | `SM` | 0 | 2 |
| `serbia` | `RS` | 18 | 15 |
| `slovakia` | `SK` | 10 | 8 |
| `slovenia` | `SI` | 2 | 2 |
| `spain` | `ES` | 150 | 115 |
| `sweden` | `SE` | 28 | 20 |
| `switzerland` | `CH` | 10 | 8 |
| `turkey` | `TR` | 177 | 135 |
| `uk` | `GB` | 237 | 185 |
| `ukraine` | `UA` | 89 | 70 |
| `vatican_city` | `VA` | 0 | 2 |
