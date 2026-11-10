"""Render terrain and total construction costs on the router's H3 map tiles."""
import argparse
import json
from pathlib import Path

import h3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np

from train_optimizer.config import Config
from train_optimizer.geography import Router
from train_optimizer.outputs import map_projector


def plot_costs(config, output):
    west, south, east, north = config.bbox
    records = json.loads(Path(config.population_file).read_text())['features']
    cells = []
    for feature in records:
        lon, lat = feature['geometry']['coordinates'][:2]
        if west <= lon <= east and south <= lat <= north:
            p = feature['properties']
            index = p['h3Index']
            cells.append(dict(h3=index, density=float(p['population']) / h3.cell_area(index, unit='km^2')))
    if not cells:
        raise ValueError('No population tiles within configured bounds')
    router = Router(config, cells)
    resolution = router.resolution
    tiles = sorted(h3.polygon_to_cells(h3.LatLngPoly(
        [(south, west), (south, east), (north, east), (north, west)]), resolution))
    project = map_projector(config)
    polygons, base, total = [], [], []
    for tile in tiles:
        if not router.allowed(tile):
            continue
        polygons.append(project([(lon, lat) for lat, lon in h3.cell_to_boundary(tile)]))
        density = router.density.get(tile, 0)
        cost = router.cost(tile)
        total.append(cost / 1e6)
        factor = router.terrain.get(tile, {}).get('multiplier', 1)
        base.append((cost - config.urban_cost_coefficient * density * factor) / 1e6)
    if not polygons:
        raise ValueError('No allowed terrain tiles within configured bounds')
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), facecolor='#f5f7fa')
    norm = Normalize(vmin=min(base), vmax=max(total))
    boundary = project([(west, south), (east, south), (east, north), (west, north)])
    for ax, values, title in zip(axes, [base, total],
                                 ['Terrain baseline', 'Terrain + population surcharge']):
        ax.set_facecolor('#e2e7ed')
        collection = PolyCollection(polygons, array=np.asarray(values), cmap='magma_r', norm=norm,
                                    edgecolors='#ffffff', linewidths=.12)
        ax.add_collection(collection)
        ax.set_xlim(boundary[:, 0].min(), boundary[:, 0].max())
        ax.set_ylim(boundary[:, 1].min(), boundary[:, 1].max())
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('East / west (km)')
        ax.set_ylabel('North / south (km)')
        ax.spines[['top', 'right']].set_visible(False)
    # Label major nearby cities without requiring country outlines or downloads.
    cities = (json.loads(Path(config.cities_file).read_text())
              if config.cities_file and Path(config.cities_file).exists() else [])
    selected = []
    for city in sorted(cities, key=lambda x: -int(x['pop'])):
        lon, lat = float(city['lon']), float(city['lat'])
        if not (west <= lon <= east and south <= lat <= north):
            continue
        point = project([(lon, lat)])[0]
        if any(np.linalg.norm(point - other) < 38 for other in selected):
            continue
        selected.append(point)
        for ax in axes:
            ax.plot(*point, 'o', ms=3, color='#12354a')
            ax.annotate(city['name'], point, xytext=(4, 4), textcoords='offset points',
                        fontsize=8, color='#12354a', bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1))
        if len(selected) >= 10:
            break
    fig.suptitle('Rail construction cost by map tile', fontsize=22, fontweight='bold', y=.96)
    fig.text(.5, .915, f'H3 resolution {resolution} • {len(polygons):,} tiles • {west:g}° to {east:g}° E, {south:g}° to {north:g}° N',
             ha='center', fontsize=11, color='#42556a')
    fig.subplots_adjust(left=.07, right=.86, top=.86, bottom=.15, wspace=.22)
    cax = fig.add_axes([.89, .25, .018, .5])
    fig.colorbar(collection, cax=cax, label='Construction cost (million €/km)')
    fig.text(.07, .075, 'RGB relief colors approximate terrain; no measured slopes or engineering feasibility. Water uses bridge cost.',
             fontsize=10, color='#42556a')
    fig.text(.07, .045, 'Missing population: zero surcharge assumed. Gray tiles: excluded or outside terrain coverage. Costs sampled at H3 centers.',
             fontsize=10, color='#42556a')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved {output}: {len(polygons)} tiles, total costs {min(total):.2f}–{max(total):.2f} M€/km')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.json'))
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'))
    parser.add_argument('--output', type=Path, default=Path('runs/terrain_cost_tiles.png'))
    args = parser.parse_args()
    data = json.loads(Path('config.json').read_text())
    if args.config.resolve() != Path('config.json').resolve():
        data.update(json.loads(args.config.read_text()))
    if args.bbox:
        data['bbox'] = args.bbox
    plot_costs(Config.from_dict(data), args.output)


if __name__ == '__main__':
    main()
