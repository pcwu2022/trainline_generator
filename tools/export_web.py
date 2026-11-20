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



def render_background(config, graph, destination, viewport=None):
    """Render population and mean-elevation hexagons without changing route costs."""
    from rail_gen.maps import MapBackground
    from rail_gen.map_style import population_collection
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from rail_gen.map_elevation import add_elevation
    background = MapBackground(graph, config)
    if viewport is not None:
        background.viewport = viewport
    xmin, xmax, ymin, ymax = background.viewport
    fig = plt.figure(figsize=(12, 12 * (ymax-ymin)/(xmax-xmin)))
    ax = fig.add_axes([0, 0, 1, 1])
    add_elevation(ax, background.elevation)
    ax.add_collection(population_collection(background.polygons, background.densities))
    for xy in background.country_lines:
        ax.plot(xy[:, 0], xy[:, 1], color='#9baeb3', linewidth=.7)
    ax.set(xlim=(xmin,xmax), ylim=(ymin,ymax)); ax.set_axis_off()
    fig.savefig(destination, dpi=120); plt.close(fig)
    return background


def export(config, output, cache_dir, reuse=True):
    if reuse and all((output/name).is_file() for name in ('scenario.json','graph.zip','background.png')):
        previous=json.loads((output/'scenario.json').read_text())
        if previous['config']==config:
            with zipfile.ZipFile(output/'graph.zip') as archive:
                old=json.loads(archive.read('graph.json'))
            if old['metadata'].get('fingerprint')==fingerprint(config):
                print(f'Reusing prepared scenario: {output}',flush=True)
                return len(old['cities'])
    graph = cached_graph(config, cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    background = render_background(config, graph, output/'background.png')
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
    'south_korea': 'South Korea — Mainland',
    'france': 'France', 'taiwan': 'Taiwan', 'usa': 'United States',
    'uk': 'United Kingdom', 'japan': 'Japan', 'germany': 'Germany',
    'spain': 'Spain', 'italy': 'Italy', 'poland': 'Poland', 'benelux': 'Benelux',
    'switzerland': 'Switzerland', 'india': 'India', 'scandinavia': 'Scandinavia',
    'california': 'California', 'texas': 'Texas',
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
    p.add_argument('--background-only', action='store_true', help='Refresh existing web images without changing prepared routes')
    args=p.parse_args()
    if args.background_only:
        root = args.output or Path('web/scenarios')
        import tempfile
        from rail_gen.alignment import Alignment, Edge, CandidateGraph
        from rail_gen.prep import City
        import shutil
        groups = {}
        for path in sorted(root.rglob('scenario.json')):
            scenario = json.loads(path.read_text())
            config = scenario['config']
            key = json.dumps([scenario['viewport'], {k:v for k,v in config.items() if k not in ('prep', 'search', 'reporting')}], sort_keys=True)
            if key in groups:
                shutil.copyfile(groups[key], path.parent/'background.png')
            else:
                with tempfile.TemporaryDirectory() as directory:
                    with zipfile.ZipFile(path.parent/'graph.zip') as archive:
                        archive.extractall(directory)
                    folder = Path(directory)
                    descriptor = json.loads((folder/'graph.json').read_text())
                    graph = CandidateGraph([City(**c) for c in descriptor['cities']],
                        [Edge(e['a'], e['b'], {k: Alignment(**v) for k,v in e['alignments'].items()}) for e in descriptor['edges']],
                        tuple(sparse.load_npz(folder/f'matrix{i}.npz') for i in range(2)),
                        np.load(folder/'tile_costs.npy', allow_pickle=False), descriptor['metadata'])
                    render_background(config, graph, path.parent/'background.png', scenario['viewport'])
                groups[key] = path.parent/'background.png'
            print(f'Refreshed {path.parent}', flush=True)
        return
    if args.profiles:
        for profile in args.profiles:
            export_profile(profile,(args.output or Path('web/scenarios'))/profile,args)
    else:
        export_profile(args.profile,args.output or Path('web/scenarios')/args.profile,args)


if __name__=='__main__': main()
