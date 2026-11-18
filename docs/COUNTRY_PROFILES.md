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
