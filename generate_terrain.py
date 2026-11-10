"""Average a Europe DEM onto metric 3 km and 22 km elevation grids."""
import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def generate_terrain(source, output_dir, resolutions_km=(3, 22)):
    """Write float32 mean elevations, excluding source nodata, in EPSG:3035.

    Elevation units are preserved from the source. Each grid is resampled directly
    from the original DEM, rather than from another coarse output.
    """
    source, output_dir = Path(source), Path(output_dir)
    resolutions_km = tuple(resolutions_km)
    if not resolutions_km or any(not math.isfinite(km) or km <= 0 for km in resolutions_km):
        raise ValueError('Resolutions must be finite positive distances in km')
    outputs = [output_dir / f'europe_terrain_{km:g}km.tif' for km in resolutions_km]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Duplicate output resolutions')
    for output in outputs:
        if output.resolve() == source.resolve() or output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError('Source DEM must declare its coordinate reference system')
        if src.count != 1:
            raise ValueError('Expected a single-band elevation DEM')
        output_dir.mkdir(parents=True, exist_ok=True)
        for km, output in zip(resolutions_km, outputs):
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:3035', src.width, src.height, *src.bounds,
                resolution=km * 1000,
            )
            # Reproject band-to-band so the high-resolution DEM is not loaded
            # into a full-sized NumPy array. GDAL handles chunked reads/warping.
            try:
                with rasterio.open(
                    output, 'w', driver='GTiff', width=width, height=height,
                    count=1, dtype='float32', crs='EPSG:3035', transform=transform,
                    nodata=np.nan, compress='deflate', predictor=3, tiled=True,
                    BIGTIFF='IF_SAFER',
                ) as dst:
                    reproject(
                        source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                        src_nodata=src.nodata, dst_nodata=np.nan,
                        resampling=Resampling.average, warp_mem_limit=256,
                    )
                    dst.update_tags(**src.tags())
                    dst.update_tags(resampling='average', source_dem=source.name,
                                    grid_resolution_m=str(km * 1000))
                    dst.set_band_description(1, 'Mean elevation')
                    if src.units[0]:
                        dst.set_band_unit(1, src.units[0])
            except BaseException:
                output.unlink(missing_ok=True)
                raise
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, default=Path('data/europe_dem_90m.tif'))
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--resolutions-km', nargs='+', type=float, default=[3, 22])
    args = parser.parse_args()
    try:
        outputs = generate_terrain(args.source, args.output_dir, args.resolutions_km)
    except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
        parser.exit(1, f'Terrain generation failed: {exc}\n')
    for output in outputs:
        print(f'Generated {output}')


if __name__ == '__main__':
    main()
