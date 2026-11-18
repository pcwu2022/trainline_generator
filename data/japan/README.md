# Japan railway inputs

Run from the repository root:

```bash
.venv/bin/python -m rail_gen run \
  --profile japan \
  --output runs/japan \
  --iterations 100 \
  --snapshot-every 1 \
  --gif-frame-ms 100 \
  --batch-accept-limit 20
```

Add `--overwrite` to replace an existing run directory.
Select the bundled profile with `--profile japan`.

Coverage is central Honshu, not all Japan. The supplied
`data/rasters_GEDTM30.tar.gz` contains a single `output_be.tif`, with WGS84
bounds approximately 134.27569–141.61125 E and 33.41958–38.61264 N.
The profile stays inside those bounds: 134.28–141.61 E, 33.42–38.61 N.
Hokkaido, Kyushu, Okinawa and much of western Japan are outside this scenario.

- `japan_dem_30m.tif`: the archive's original single-band float32 elevation
  raster, extracted and renamed. Elevations are assumed to be metres.
- `japan_terrain_3km.tif`: mean elevations resampled directly from that raster
  to EPSG:3035 at 3 km, using `tools.generate_terrain.generate_terrain`. Coastal nodata
  is filled from the nearest valid elevation within two cells (6 km), matching
  the Taiwan preparation. Remaining nodata is retained.
- `japan_land_relief.tif`: neutral RGB placeholder (160,160,160), aligned to
  the 3 km DEM. As in Taiwan, relief-plus-DEM mode uses ground slopes for costs;
  country boundaries supply the land/sea mask.
- `japan_cities.json`: 402 JP city records inside the profile bounds from
  `data/cities50000.json`, preserving original source indices as IDs.
- `japan_population.geojson`: 3,695 population points from `data/asia.geojson`,
  clipped to the profile bounds plus a 0.3-degree density-smoothing buffer.

Validation prepared 100 candidate cities and 337 candidate pairs, and built
terrain cost surfaces successfully. Full alignment routing and optimization
have not been run. The profile inherits current optimizer defaults, including
phase-based station sampling weights, and Taiwan's center coefficient of 0.15.

Limitations: the generator uses Europe-centered EPSG:3035, which distorts
distances and slopes in Japan. Three-kilometre averaging smooths mountains.
City populations and demand remain uncalibrated development inputs.

The raw archive and TIFF rasters are local, Git-ignored inputs. Existing tracked
rasters elsewhere in the repository remain tracked.
