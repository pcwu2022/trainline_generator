"""Prepare a static browser scenario using the normal Python geography pipeline.

python -m tools.export_web --profile france --output web/scenarios/france
"""
import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy import sparse
from rail_gen.config import load_config
from rail_gen.alignment import cached_graph, fingerprint


def export(config, output, cache_dir, reuse=True):
    if reuse and all((output/name).is_file() for name in ('scenario.json','graph.zip','background.png')):
        previous=json.loads((output/'scenario.json').read_text())
        if previous['config']==config:
            with zipfile.ZipFile(output/'graph.zip') as archive:
                old=json.loads(archive.read('graph.json'))
            if old['metadata'].get('fingerprint')==fingerprint(config):
                print(f'Reusing prepared scenario: {output}',flush=True)
                return len(old['cities'])
    from rail_gen.maps import MapBackground
    from rail_gen.map_style import population_collection
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    graph = cached_graph(config, cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    background = MapBackground(graph, config)
    xmin, xmax, ymin, ymax = background.viewport
    fig = plt.figure(figsize=(12, 12 * (ymax-ymin)/(xmax-xmin)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#eef2f5')
    for xy in background.country_lines:
        ax.plot(xy[:, 0], xy[:, 1], color='#9baeb3', linewidth=.7)
    ax.add_collection(population_collection(background.polygons, background.densities))
    # Rasterize the already computed terrain polygons into the fixed map image.
    if background.terrain and background.terrain[0]:
        polygons, costs, low, high = background.terrain
        from matplotlib.colors import Normalize
        ax.add_collection(PolyCollection([background.project(p) for p in polygons],
            array=np.asarray(costs), cmap='Greys', norm=Normalize(low, high), alpha=.4, linewidth=0))
    ax.set(xlim=(xmin,xmax), ylim=(ymin,ymax)); ax.set_axis_off()
    fig.savefig(output/'background.png', dpi=120); plt.close(fig)
    descriptor = dict(version=1, config=config, metadata=graph.metadata,
        cities=[asdict(c) for c in graph.cities], edges=[asdict(e) for e in graph.edges],
        viewport=list(background.viewport),
        nodes=background.node_xy.tolist(),
        paths={f'{i}:{track}': points.tolist() for (i,track),points in background.edge_xy.items()})
    with zipfile.ZipFile(output/'graph.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('graph.json', json.dumps(descriptor, allow_nan=False))
        data=io.BytesIO(); np.save(data, graph.tile_costs, allow_pickle=False)
        archive.writestr('tile_costs.npy',data.getvalue())
        for i,matrix in enumerate(graph.matrices):
            data=io.BytesIO(); sparse.save_npz(data,matrix); archive.writestr(f'matrix{i}.npz',data.getvalue())
    (output/'scenario.json').write_text(json.dumps(dict(config=config, viewport=list(background.viewport),
        nodes=background.node_xy.tolist(), cities=[c.name for c in graph.cities],
        paths=descriptor['paths'], terrain=config['terrain_kind'], actual_city_count=len(graph.cities)),indent=2)+'\n')
    print(f'Browser scenario saved to {output}',flush=True)
    return len(graph.cities)


PROFILE_LABELS = {
    'france': 'France', 'taiwan': 'Taiwan', 'usa': 'United States',
    'uk': 'United Kingdom', 'japan': 'Japan', 'germany': 'Germany',
    'spain': 'Spain', 'italy': 'Italy', 'poland': 'Poland', 'benelux': 'Benelux',
    'switzerland': 'Switzerland', 'india': 'India', 'scandinavia': 'Scandinavia',
    'usa_northeast': 'United States — Northeast',
}


def export_profile(profile, output, args):
    choices=[]
    for count in sorted(set(args.city_counts or [args.max_cities])):
        config=load_config(args.config,profile,{'prep':{'max_cities':count},
            'alignment':{'cell_km':args.cell_km}, 'search':{'population':8,'generations':8,'workers':1}})
        folder = output / f'cities-{count}' if args.city_counts else output
        actual=export(config,folder,args.cache_dir,reuse=not args.force)
        choices.append(dict(city_count=count, actual_city_count=actual, path=f'cities-{count}' if args.city_counts else '.'))
    index=dict(profile=profile, label=PROFILE_LABELS.get(profile,profile.replace('_',' ').title()),
        default_city_count=8 if any(c['city_count']==8 for c in choices) else choices[0]['city_count'], scenarios=choices)
    (output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    # Register all prepared profiles, never offer profiles without exported graphs.
    profiles=[]
    for path in sorted(output.parent.glob('*/index.json')):
        prepared=json.loads(path.read_text())
        if 'profile' in prepared:
            profiles.append(dict(id=prepared['profile'], label=prepared['label'], path=path.parent.name))
    (output.parent/'index.json').write_text(json.dumps(dict(
        default_profile='france' if any(p['id']=='france' for p in profiles) else profiles[0]['id'],
        profiles=profiles),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--profile',default='france')
    selection.add_argument('--profiles',nargs='+',help='Export multiple named profiles into the output directory')
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,help='Profile directory, or parent directory when using --profiles')
    p.add_argument('--cache-dir',type=Path,default=Path('runs/railgen-cache'))
    p.add_argument('--city-counts',type=int,nargs='+',help='Export selectable city limits, e.g. 4 8 12 20 50 100')
    p.add_argument('--max-cities',type=int,default=8)
    p.add_argument('--cell-km',type=float,default=12)
    p.add_argument('--force',action='store_true',help='Re-render unchanged prepared bundles')
    args=p.parse_args()
    if args.profiles:
        for profile in args.profiles:
            export_profile(profile,(args.output or Path('web/scenarios'))/profile,args)
    else:
        export_profile(args.profile,args.output or Path('web/scenarios')/args.profile,args)


if __name__=='__main__': main()
