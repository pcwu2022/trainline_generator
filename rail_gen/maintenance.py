"""Shared physical-resource maintenance, independent of rolling stock."""
import numpy as np
from scipy import sparse

from .alignment import TRACKS


def length_matrices(graph):
    """Cell entries are km; explicit crossing entries use full asset length."""
    used = sorted({t for edge in graph.edges for a in edge.alignments.values() for t in a.tiles})
    compact = {tile: i for i, tile in enumerate(used)}
    assets = {a['resource_id']: a['length_km'] for a in graph.metadata.get('water_crossings', [])}
    result = []
    for track in TRACKS:
        rows, cols, values = [], [], []
        for i, edge in enumerate(graph.edges):
            if track not in edge.alignments:
                continue
            a = edge.alignments[track]
            lengths = (a.tile_lengths if a.tile_lengths is not None else
                       [a.length_km / len(a.tiles)] * len(a.tiles))
            for tile, length in zip(a.tiles, lengths):
                rows.append(i); cols.append(compact[tile])
                values.append(assets.get(tile, length))
        result.append(sparse.csr_matrix((values, (rows, cols)), shape=graph.matrices[0].shape))
    return tuple(result)


def maintenance(lengths, x, counts, tracks, network, frequency, annual_hours):
    """Fixed cost once per physical track; variable cost once per train passage.

    K is annual train-km, so variable cost is b*K (not length*b*K).
    Shared raster resources retain the maximum occupied length approximation.
    """
    selected = (lengths[0].multiply((x == 1)[:, None]) +
                lengths[1].multiply((x == 2)[:, None])).tocsr()
    coo = selected.tocoo()
    nr = selected.shape[1]
    resource_length = np.zeros(nr)
    resource_count = np.zeros(nr)
    resource_type = np.zeros(nr, dtype=int)
    np.maximum.at(resource_length, coo.col, coo.data)
    np.maximum.at(resource_count, coo.col, counts[coo.row])
    np.maximum.at(resource_type, coo.col, x[coo.row])
    fixed_rate = np.array([0.] + [tracks[t]['maintenance_fixed_eur_track_km_year'] for t in TRACKS])
    variable_rate = np.array([0.] + [tracks[t]['maintenance_variable_eur_trainkm'] for t in TRACKS])
    physical_km = resource_length * resource_count
    fixed_resource = physical_km * fixed_rate[resource_type]
    # Divide a shared resource's fixed charge among its incident built edges.
    incidence = selected.copy(); incidence.data[:] = 1
    sharing = np.asarray(incidence.sum(axis=0)).ravel()
    fixed = np.asarray(incidence @ np.divide(fixed_resource, sharing,
        out=np.zeros(nr), where=sharing > 0)).ravel()
    variable = np.zeros(len(x))
    for sid, edges in enumerate(network.edges):
        section = selected[list(edges)].tocoo()
        longest = np.zeros(nr)
        np.maximum.at(longest, section.col, section.data)
        summed = np.bincount(section.col, weights=section.data, minlength=nr)
        resource_cost = 2 * frequency[sid] * annual_hours * longest * variable_rate[resource_type]
        # A service crossing a shared resource consumes it once. Split its cost
        # across its participating edge records so corridor nets reconcile.
        shares = section.data / summed[section.col]
        np.add.at(variable, np.asarray(edges)[section.row], resource_cost[section.col] * shares)
    return fixed, variable, float(physical_km.sum())
