# Taiwan railway inputs

The active profile now uses the shared [ETOPO elevation source](../../docs/ETOPO.md).
The local relief and DEM files described below are historical inputs.

Run from the repository root:

```sh
.venv/bin/python -m rail_gen run --profile taiwan --iterations 200 --output runs/taiwan
```

- Population: 689 H3 cells from `data/asia.geojson`, clipped to 119.6–122.4 E, 21.55–25.7 N with a buffer for density smoothing.
- Cities: 39 TW records from `data/cities50000.json`, clipped to scenario bounds 119.9–122.1 E, 21.85–25.4 N. Offshore islands excluded. Metropolitan screening retains 13 candidates.
- `taiwan_dem_30m.tif`: unchanged `output_be.tif` from the supplied GEDTM30 archive; WGS84 single-band float32 elevation. Elevations assumed metres; no vertical datum or attribution metadata supplied.
- `taiwan_terrain_3km.tif`: mean elevations resampled directly into EPSG:3035 at 3 km. Nearest valid elevations extend up to two cells (6 km) into coastal nodata to support gradients. Sea remains blocked using `data/country_boundaries.geojson`.
- `taiwan_slope_3km.tif`: dimensionless rise/run from `numpy.gradient` at 3000 m spacing; multiply by 100 for percent slope.
- `taiwan_land_relief.tif`: neutral RGB placeholder (160,160,160), enabling existing relief-plus-DEM slope-cost mode. Land/sea comes from country boundaries. The run consumes elevation and calculates slopes on its own grid; do not configure the slope raster as an elevation DEM.

Verified candidate graph: 13 cities, 36 pairs. Hualien has no feasible alignment under the coarse coastline/terrain routing; the other 12 candidates connect. One-iteration smoke run generated maps/reports in `runs/taiwan-smoke` and ended with expected capacity_shortfall (89.76% spill), so this is not a converged network.

Limitations: 3 km terrain smooths mountains. The generator uses Europe-centered EPSG:3035, distorting Taiwan distances and slopes. Demand/population are uncalibrated; these are development inputs, not engineering design data.

The archive was deleted after verifying the extracted DEM byte-for-byte by SHA-256:
`d29fb08bed3bb27a318f653e1bc774f39ad06fe3aa820ed0e79944c8a0941464`
