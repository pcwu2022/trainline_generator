# California

The `california` profile selects US cities covered by the Census California state
polygon and population cells whose centroids are covered by that polygon.
Entire population cells are retained; border cells are assigned by centroid.
State membership filters cities and population; routes may cross state borders.

Boundaries use the [2024 Census cartographic boundary archive](https://www2.census.gov/geo/tiger/GENZ2024/kml/cb_2024_us_state_500k.zip).
`source.json` records the archive checksum and source inputs.

Regenerate both state profiles from the repository root:

```sh
python -m tools.prepare_usa_states
```

Pass `--states-kml /path/to/cb_2024_us_state_500k.zip` to reuse a local archive.
