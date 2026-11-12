"""Map the largest fixed city-pair demand markets on a saved run's basemap.

Usage: python -m rail_gen.demand_map --run-dir runs/europe-live --top 500
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

from .alignment import load_graph
from .config import load_config
from .maps import MapBackground
from train_optimizer.map_style import CITY_STYLE, population_collection
from train_optimizer.map_terrain import add_terrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('runs/railgen-cache'))
    parser.add_argument('--top', type=int, default=500)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error('--top must be positive')
    config = load_config(args.run_dir / 'run_config.json', profile=None)
    manifest = json.loads((args.run_dir / 'alignment_manifest.json').read_text())
    graph = load_graph(args.cache_dir / manifest['fingerprint'])
    background = MapBackground(graph, config)
    evaluator = background.evaluator
    selected = np.asarray(evaluator.ranked_od[:args.top], dtype=int)
    demand = evaluator.reference_trips[selected]
    if not len(demand):
        raise ValueError('No positive demand pairs in this run')
    origins = evaluator.origins[selected]
    destinations = evaluator.destinations[selected]
    stem = args.run_dir / f'top_{args.top}_demand_pairs'
    with stem.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'origin_id', 'origin', 'destination_id', 'destination',
                         'annual_two_way_reference_trips', 'distance_km'])
        for rank, (a, b, trips, od) in enumerate(zip(origins, destinations, demand, selected), 1):
            writer.writerow([rank, graph.cities[a].id, graph.cities[a].name,
                             graph.cities[b].id, graph.cities[b].name,
                             float(trips), float(evaluator.od_distance[od])])
    figure = plt.figure(figsize=background.figure_size, layout='none')
    ax = figure.add_axes([.09, .16, .72, .72])
    density_axis = figure.add_axes([.85, .28, .025, .46])
    try:
        ax.set_facecolor('#eef2f5')
        for coordinates in background.country_lines:
            ax.plot(*coordinates.T, color='#c2c8c5', linewidth=.6, zorder=0)
        collection = population_collection(background.polygons, background.densities)
        ax.add_collection(collection)
        figure.colorbar(collection, cax=density_axis,
                        label='Population density (people/km², log scale)')
        add_terrain(ax, figure, background.terrain, background.project)
        # Exact linear width scaling, with no offset, clipping or log transform.
        width_scale = 6 / demand.max()
        segments = np.stack([background.node_xy[origins], background.node_xy[destinations]], axis=1)
        ax.add_collection(LineCollection(segments[::-1], linewidths=(demand * width_scale)[::-1],
                                        colors='#164f79', alpha=.5, zorder=4))
        for i, city in enumerate(graph.cities):
            xy = background.node_xy[i]
            ax.plot(*xy, **CITY_STYLE)
            if i in background.labelled_cities:
                ax.annotate(city.name, xy, xytext=(4, 5), textcoords='offset points', fontsize=8, zorder=6)
        xmin, xmax, ymin, ymax = background.viewport
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
               xlabel='East of regional center (km)', ylabel='North of regional center (km)')
        ax.set_aspect('equal')
        ax.grid(alpha=.15)
        ax.set_title(f'Europe · Top {len(selected):,} city-pair demand markets\n'
                     'Fixed reference demand · annual trips, both directions', fontsize=11)
        references = np.array([.1, .25, .5, 1]) * demand.max()
        handles = [Line2D([], [], color='#164f79', alpha=.5, linewidth=value * width_scale,
                          label=f'{value / 1e6:.2f}M') for value in references]
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.45, .055),
                      ncol=4, frameon=False, fontsize=8, handlelength=3,
                      title='Annual two-way trips · line width proportional to demand', title_fontsize=8)
        figure.supxlabel('Direct city-pair connections · Synthetic gravity reference demand, before network elasticities\n'
                           f'{len(graph.cities):,} cities · {len(selected):,} unique pairs · '
                           f'Range: {demand.min():,.0f}–{demand.max():,.0f} trips/year', fontsize=8, y=.005)
        with matplotlib.rc_context({'savefig.bbox': None}):
            figure.savefig(stem.with_suffix('.png'), dpi=180)
        print(f'Created {stem}.png and {stem}.csv; {len(selected)} unique pairs')
    finally:
        plt.close(figure)


if __name__ == '__main__':
    main()
