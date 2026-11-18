# United States Northeast

The `usa_northeast` profile uses the US Census Bureau Northeast region:
Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont,
New Jersey, New York and Pennsylvania. It excludes neighboring states and Canada.

Definition: [Census geographic regions](https://www.census.gov/data/data-tools/survey-explorer/geo.html).
State geometry: [2024 Census cartographic boundary files](https://www.census.gov/geographies/mapping-files/2024/geo/carto-boundary-file.html),
1:500,000 KML, from the [official download](https://www2.census.gov/geo/tiger/GENZ2024/kml/cb_2024_us_state_500k.zip).

`states.geojson` preserves the nine state polygons in WGS84. `cities.json` selects
existing `data/cities50000.json` US records whose coordinates are covered by those
polygons. `population.geojson` selects existing `data/north-america.geojson` cells
whose centroids fall inside the region. Entire population cells are retained, so
cells straddling a border are assigned by centroid rather than split. The cities
and population are not selected by a rectangular bounding box alone.

The profile retains the global land boundaries for terrain routing. A candidate
route may travel outside a state boundary; state membership controls the cities
and population inputs, not a restriction on every metre of track.

`source.json` records the input paths, state list and downloaded archive checksum.
To regenerate from the repository root:

```sh
python -m tools.prepare_usa_northeast
```

Or use an already downloaded official archive:

```sh
python -m tools.prepare_usa_northeast --states-kml /path/to/cb_2024_us_state_500k.zip
```
