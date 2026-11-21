"""Browser adapter. All preparation, routing, evaluation and search use rail_gen."""
import json
from pathlib import Path
import numpy as np
from js import postMessage
from rail_gen.config import load_config
from rail_gen.prep import City
from rail_gen.alignment import Alignment, Edge, CandidateGraph
from scipy import sparse
from rail_gen.search import search
from rail_gen.report import geojson


def emit(kind, **values):
    postMessage(json.dumps(dict(type=kind, **values)))


def run_browser(config_json, scenario_dir='/project/scenario'):
    overrides = json.loads(config_json)
    config = load_config(profile=None, overrides=overrides)
    if config['search']['workers'] != 1:
        raise ValueError('Browser execution requires search.workers = 1')
    if config['search']['algorithm'] != 'incremental':
        raise ValueError('This browser interface supports the incremental algorithm')
    if config['search']['sensitivity'] or config['backtest_file']:
        raise ValueError('Sensitivity and backtest calibration are available through the Python CLI')
    folder = Path(scenario_dir)
    descriptor = json.loads((folder / 'graph.json').read_text())
    # Saved bundles may predate new defaults or configuration migrations.
    # Compare both configurations using the same current schema.
    original = load_config(profile=None, overrides=descriptor['config'])
    editable = {'demand', 'operations', 'finance', 'constraints', 'search', 'reporting', 'rolling_stock', 'services'}
    for key in config:
        if key not in editable and config[key] != original[key]:
            raise ValueError(f'{key} is baked into this scenario; regenerate with tools.export_web')
    graph = CandidateGraph([City(**c) for c in descriptor['cities']],
        [Edge(e['a'], e['b'], {k: Alignment(**v) for k,v in e['alignments'].items()}) for e in descriptor['edges']],
        tuple(sparse.load_npz(folder / f'matrix{i}.npz') for i in range(2)),
        np.load(folder / 'tile_costs.npy', allow_pickle=False), descriptor['metadata'])
    emit('status', message='Running Python optimizer on prepared terrain alignments…')
    history = []

    def observe(update):
        payload = dict(iteration=update.iteration, phase=update.phase,
                       seconds=update.elapsed_seconds, metrics=update.result.metrics,
                       network=geojson(graph, update.result))
        history.append({k: v for k, v in payload.items() if k != 'network'})
        emit('update', **payload)

    result = search(graph, config, observer=observe)
    emit('done', status=result.status, history=history,
         config=config, evaluations=result.evaluations,
         elapsed_seconds=result.elapsed_seconds)
