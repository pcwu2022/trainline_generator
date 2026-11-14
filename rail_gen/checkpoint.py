"""Validated, pickle-free incremental search restart artifacts."""
import json
from copy import deepcopy

import numpy as np

from .track_layout import MODEL_VERSION


def assumptions(config):
    result = deepcopy(config)
    result.pop('reporting', None)
    for key in ('generations', 'time_limit_seconds', 'workers'):
        result['search'].pop(key, None)
    return result


def load_checkpoint(path, graph, config):
    if config['search']['algorithm'] != 'incremental':
        raise ValueError('--resume currently supports only the incremental algorithm')
    with np.load(path, allow_pickle=False) as saved:
        required = {'genome', 'iteration', 'graph_fingerprint', 'config_json',
                    'model_version', 'track_counts', 'services_json'}
        if not required <= set(saved.files):
            raise ValueError('Checkpoint lacks the network/service metadata required for resume')
        if str(saved['model_version']) != MODEL_VERSION:
            raise ValueError('Checkpoint model version is incompatible')
        if str(saved['graph_fingerprint']) != graph.metadata.get('fingerprint', ''):
            raise ValueError('Checkpoint geography fingerprint does not match')
        if assumptions(json.loads(str(saved['config_json']))) != assumptions(config):
            raise ValueError('Checkpoint configuration does not match (only iterations, time limit, workers and reporting may change)')
        genome = saved['genome'].copy()
        counts = saved['track_counts'].copy()
        if (genome.shape != (len(graph.edges),) or counts.shape != genome.shape
                or not np.issubdtype(genome.dtype, np.integer)
                or not np.isin(genome, [0, 1, 2]).all()
                or not np.isin(counts, [0, 1, 2, 4]).all()):
            raise ValueError('Invalid checkpoint genome or track counts')
        iteration = int(saved['iteration'])
        if iteration < 0:
            raise ValueError('Invalid checkpoint iteration')
        return dict(genome=genome, track_counts=counts,
                    services=json.loads(str(saved['services_json'])), iteration=iteration,
                    state=json.loads(str(saved['search_state_json'])) if 'search_state_json' in saved else {})
