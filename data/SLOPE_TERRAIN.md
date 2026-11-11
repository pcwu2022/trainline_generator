# Slope-cost trial elevation data

Run `.venv/bin/python fetch_slope_terrain.py` to reproduce the trial dataset.
It fetches zoom-6 GeoTIFF tiles covering European scenario bounds from
[Mapzen Terrain Tiles on AWS](https://registry.opendata.aws/terrain-tiles/)
and averages their elevations onto a 3 km EPSG:3035 grid. Downloaded tiles,
the VRT mosaic and the derived DEM are ignored by Git. Source tiles are cached
in `data/slope_tiles`; the DEM is `data/slope_trial/europe_terrain_3km.tif`.

This is a coarse regional trial, not the high-resolution Copernicus download
provided by `fetch_terrain.py`. Resampling smooths slopes and omits small-scale
obstacles. The source may include bathymetry: water detection still comes from
the existing relief/land-class data, not from elevation sign.

Provider credits for the European coverage, following the
[provider attribution instructions](https://github.com/tilezen/joerd/blob/master/docs/attribution.md):

- European terrain includes EU-funded Copernicus EU-DEM data.
- GMTED2010 and SRTM terrain data courtesy of the U.S. Geological Survey.
- ETOPO1 bathymetry courtesy of the U.S. National Oceanic and Atmospheric Administration.
- Austrian terrain: © offene Daten Österreichs, Digitales Geländemodell Österreich.
- Norwegian terrain: © Kartverket.
- UK terrain: © Environment Agency copyright and/or database right 2015, all rights reserved.
- ArcticDEM was produced from DigitalGlobe imagery, with funding under NSF awards
  1043681, 1559691 and 1542736.

See the provider's [data-source description](https://github.com/tilezen/joerd/blob/master/docs/data-sources.md)
for the source mixture at each zoom level. The derived grid is an averaged,
reprojected adaptation; the slope/cost scenario is this project's interpretation.
