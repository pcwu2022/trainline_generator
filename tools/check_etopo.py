"""Validate an ETOPO DEM and export model-grid slope and trial construction costs.

Run from the repository root: python -m tools.check_etopo --profile europe
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rail_gen.alignment import cost_surfaces
from rail_gen.config import load_config
from rail_gen.projection import regional_projection

SOURCE = 'data/ETOPO_2022_v1_60s_N90W180_bed.tif'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(SOURCE))
    parser.add_argument('--profile', default='europe')
    parser.add_argument('--output', type=Path, default=Path('runs/etopo-validation'))
    args = parser.parse_args()
    output = args.output / args.profile
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.source) as src:
        if src.count != 1 or not src.crs or src.units[0] != 'metre':
            raise ValueError('Expected a georeferenced, single-band ETOPO elevation raster in metres')
        metadata = dict(source=str(args.source), crs=src.crs.to_string(), units=src.units[0],
            dimensions=[src.width, src.height], bounds=list(src.bounds), nodata=src.nodata,
            resolution_degrees=list(src.res), tags=src.tags())
        valid, missing, minimum, maximum = 0, 0, float('inf'), -float('inf')
        for _, window in src.block_windows(1):
            values = src.read(1, window=window, masked=True)
            valid += int(values.count())
            missing += int(values.size - values.count())
            if values.count():
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
        metadata.update(valid_pixels=valid, missing_pixels=missing, elevation_range_m=[minimum, maximum])
    c = load_config(profile=args.profile, overrides=dict(terrain_kind='etopo',
        terrain_file=str(args.source), terrain_dem_file=''))
    projection = regional_projection(c)
    samples = (np.array([[0., 0.]]), None, np.array([0.]))
    terrain, water, grade, transform = cost_surfaces(c, samples, terrain_only=True)
    costs, _, _, _ = cost_surfaces(c, samples)
    for name, values in [('slope_fraction', grade), ('terrain_cost_meur_km', terrain),
                         ('cr_cost_meur_km', costs[0]), ('hsr_cost_meur_km', costs[1]),
                         ('water_mask', water.astype(float))]:
        values = np.where(np.isfinite(values), values, np.nan).astype('float32')
        with rasterio.open(output / f'{name}.tif', 'w', driver='GTiff',
            width=values.shape[1], height=values.shape[0], count=1, dtype='float32',
            crs=projection.crs, transform=transform, nodata=np.nan, compress='deflate') as dst:
            dst.write(values, 1)
    finite_land = np.isfinite(terrain)
    if not finite_land.any() or not np.isfinite(costs[:, water]).all():
        raise ValueError('Missing land coverage or invalid water costs')
    if np.any(~water & ~finite_land):
        raise ValueError('Missing slope/terrain coverage on land')
    if not np.isfinite(costs).all():
        raise ValueError('Unexpected blocked construction cells in ETOPO trial')
    metadata.update(metric_crs=projection.crs.to_wkt(),
        projection_center=[sum(c['bbox'][::2])/2, sum(c['bbox'][1::2])/2],
        terrain_kind=c['terrain_kind'], profile=args.profile, grid_shape=list(grade.shape), cell_km=c['alignment']['cell_km'],
        land_cells=int(finite_land.sum()), water_cells=int(water.sum()),
        slope_percentiles_percent=np.percentile(grade[finite_land]*100, [0, 50, 90, 99, 100]).tolist(),
        terrain_cost_range_meur_km=[float(terrain[finite_land].min()), float(terrain[finite_land].max())],
        cr_blocked_land_cells=int((~water & ~np.isfinite(costs[0])).sum()),
        zero_density_trial=True,
        source_url='https://www.ncei.noaa.gov/products/etopo-global-relief-model',
        slope_method='Average elevation into regional metric cells, then hypot(dz/dx, dz/dy)',
        water_method='Existing country boundaries and optional land_class_file; never elevation sign')
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    for ax, values, title, cmap, vmax in zip(axes,
        [np.where(water, np.nan, grade*100), terrain, np.where(np.isfinite(costs[0]), costs[0], np.nan)],
        ['Ground slope (%)', 'Land terrain cost (M€/km)', 'CR construction cost (M€/km)'],
        ['magma', 'YlOrBr', 'viridis'], [20, 30, 120]):
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=.65)
        ax.set_title(title); ax.set_axis_off()
    fig.suptitle(f'ETOPO 2022 • {args.profile} • {c["alignment"]["cell_km"]} km model grid • no urban surcharge')
    fig.savefig(output / 'cost_map.png', dpi=160)
    plt.close(fig)
    (output / 'validation.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
