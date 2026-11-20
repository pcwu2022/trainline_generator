# New Zealand profile inputs

The bounds include North, South and Stewart islands and nearby islands. Chatham, Kermadec and subantarctic islands are outside the bounds.

The boundary is the bundled Natural Earth country geometry (`NZ`), clipped
to the profile bounding box. See [boundary provenance](../COUNTRY_BOUNDARIES.md).
Cities are `NZ` records from `data/cities50000.json` covered by that geometry;
IDs preserve the source record indices. Population features come from
`data/oceania.geojson`, selected by centroid within the same geometry.
Cells are retained whole. Generalized coastlines can exclude coastal city points.
The country boundary also supplies the routing land mask and map outline.

Regenerate with `python -m tools.prepare_country_profiles --profiles new_zealand`.
Export browser scenarios with
`python -m tools.export_web --profile new_zealand --city-counts 4 8 12 20 50 100`.

Currency reporting uses a fixed modeling assumption of 1 EUR = 1.8 NZD;
this is not a live exchange rate. Terrain uses the existing global ETOPO raster.
