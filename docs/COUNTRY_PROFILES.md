# Country profiles

Shared defaults live in [`config/config.json`](../config/config.json). Each profile
has its own JSON file in `config/profiles/`. The
[`profile_locations.json`](../config/profile_locations.json) index maps profile
names to paths relative to `config/`. It also includes `europe` and `smoke`.
Japan and Taiwan are available directly through `--profile`.

```sh
python -m rail_gen run --profile france --output runs/france
```

Use `--config config/overrides/slope-fast.json` for scenario overrides. Defaults
from a custom document apply first, followed by the selected profile, then CLI
overrides. Custom documents may still define inline `profiles` and `defaults`.
To add a country, create a profile file and add its location to the index.


### Display currency

Every profile specifies `reporting.currency_code`, `reporting.currency_symbol`,
and `reporting.eur_to_local` (local currency units per EUR). Map corner stats
convert annual net profit using that fixed rate; optimization, JSON metrics,
and financial analysis continue to use EUR. Change these settings to use your
preferred planning rate. Regional Europe, Benelux and Scandinavia profiles use
EUR so multi-currency networks have a single display unit.

The bundled rates are rounded display assumptions based on the provider snapshot
retrieved November 18, 2026 (provider timestamp November 19, 00:02 UTC), and are
not refreshed during runs. [Rates by ExchangeRate-API](https://www.exchangerate-api.com).
Bulgaria uses EUR following its January 2026 euro adoption.

The corner stats show annual carried trips, eligible need satisfaction, served
candidate cities / all candidate cities, and local-currency annual net profit.
Demand fulfillment on routes uses a light-to-dark blue scale; green dashed
circles continue to mark station catchments.
