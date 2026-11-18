# Japan railway inputs

The `japan` profile includes only Hokkaido, Honshu, Shikoku and Kyushu.
Map bounds are 129.4–146 E, 30.8–45.7 N, with a small margin around the islands.
Okinawa, the smaller offshore islands and neighboring countries are excluded.

The `japan_main_islands_boundaries.geojson` file contains the four largest
Japan polygons from `data/country_boundaries.geojson` (Natural Earth 1:50m;
see `data/COUNTRY_BOUNDARIES.md`). The matching city and population files
are derived from `data/cities50000.json` and `data/asia.geojson` by retaining
points covered by those four polygons. City source indices are preserved as IDs.
These generalized coastlines can exclude some coastal points.

Terrain uses the global ETOPO source and a regional projection at 3 km.
The profile retains up to 100 cities, batches of 5 updates, snapshots each
iteration and 100 ms GIF frames. Population and demand remain uncalibrated.

```sh
.venv/bin/python -m rail_gen run --profile japan --output runs/japan-main-islands
```

Previous clipped central-Honshu inputs and existing runs are preserved but
are not used by this profile. Use a fresh output directory for the new geography.
