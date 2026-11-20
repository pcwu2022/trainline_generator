# Indonesia profile inputs

All country polygon components within the profile bounds are included, including islands.

The boundary is the bundled Natural Earth country geometry (`ID`), clipped
to the profile bounding box. See [boundary provenance](../COUNTRY_BOUNDARIES.md).
Cities are `ID` records from `data/cities50000.json` covered by that geometry;
IDs preserve the source record indices. Population features come from
`data/asia.geojson` and `data/oceania.geojson` (deduplicated by H3 index), selected by centroid within the same geometry.
Cells are retained whole. Generalized coastlines can exclude coastal city points.
The country boundary also supplies the routing land mask and map outline.

Regenerate with `python -m tools.prepare_country_profiles --profiles indonesia`.
Export browser scenarios with
`python -m tools.export_web --profile indonesia --city-counts 4 8 12 20 50 100`.

Currency reporting uses a fixed modeling assumption of 1 EUR = 17,500 IDR;
this is not a live exchange rate. Terrain uses the existing global ETOPO raster.

The bundled Indonesia population sources have sparse coverage and are not a
complete national population census. Demand estimates inherit that limitation.
