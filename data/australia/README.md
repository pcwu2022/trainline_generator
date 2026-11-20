# Australia profile inputs

The bounds include mainland Australia and Tasmania; remote external islands are excluded.

The boundary is the bundled Natural Earth country geometry (`AU`), clipped
to the profile bounding box. See [boundary provenance](../COUNTRY_BOUNDARIES.md).
Cities are `AU` records from `data/cities50000.json` covered by that geometry;
IDs preserve the source record indices. Population features come from
`data/oceania.geojson`, selected by centroid within the same geometry.
Cells are retained whole. Generalized coastlines can exclude coastal city points.
The country boundary also supplies the routing land mask and map outline.

Regenerate with `python -m tools.prepare_country_profiles --profiles australia`.
Export browser scenarios with
`python -m tools.export_web --profile australia --city-counts 4 8 12 20 50 100`.

Currency reporting uses a fixed modeling assumption of 1 EUR = 1.65 AUD;
this is not a live exchange rate. Terrain uses the existing global ETOPO raster.
