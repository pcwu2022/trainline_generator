# South Korea mainland

The `south_korea` profile uses the largest South Korea polygon in the bundled
Natural Earth 1:50m country boundaries. This is the mainland peninsula; Jeju,
Geoje, Ulleungdo and other separate offshore islands are excluded. North Korea
is outside the boundary. See [boundary provenance](../COUNTRY_BOUNDARIES.md).

`boundaries.geojson` contains only that mainland polygon and is used for the
routing land mask and map boundary. `cities.json` selects KR records from
`data/cities50000.json` covered by the polygon. `population.geojson` selects
`data/asia.geojson` features whose centroids are covered by it. Source records
are preserved; population cells are assigned by centroid, not split at coasts.
Generalized coastlines can exclude some coastal city points. The bundled Busan
record is located on Yeongdo island and is excluded by this strict mainland filter.

Regenerate from the repository root:

```sh
python -m tools.prepare_south_korea
python -m tools.export_web --profile south_korea --city-counts 4 8 12 20 50 100
```

Terrain uses the existing global ETOPO raster. Browser scenarios use 12 km
routing cells, following the other prepared profiles. Reporting uses KRW with
a fixed modeling conversion of 1 EUR = 1,500 KRW, not a live exchange rate.
