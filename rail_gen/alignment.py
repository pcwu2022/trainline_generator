"""Windowed, cached cost-distance routing. Search never opens a raster.

Infrastructure is charged in fixed cell-length units, so shared cell accounting
is exact under this raster approximation. Display smoothing never changes the
engineering footprint or its price.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .prep import City, TO_METRIC, TO_WGS84

CACHE_VERSION = 2
TRACKS = ('CR', 'HSR')


@dataclass
class Alignment:
    length_km: float
    capex_meur: float
    tiles: list
    geometry: list
    pct_water: float = 0.0
    pct_steep: float = 0.0


@dataclass
class Edge:
    a: int
    b: int
    alignments: dict


@dataclass
class CandidateGraph:
    cities: list
    edges: list
    matrices: tuple  # binary (candidate edges x compact infrastructure tiles)
    tile_costs: np.ndarray  # (2, tiles), MEUR
    metadata: dict

    @classmethod
    def build(cls, cities, edges, tile_costs, metadata=None):
        used = sorted({tile for e in edges for a in e.alignments.values() for tile in a.tiles})
        compact = {tile: i for i, tile in enumerate(used)}
        matrices = []
        for track in TRACKS:
            rows, columns = [], []
            for i, edge in enumerate(edges):
                if track in edge.alignments:
                    tiles = sorted(set(edge.alignments[track].tiles))
                    rows.extend([i] * len(tiles))
                    columns.extend(compact[t] for t in tiles)
            matrices.append(sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                                             shape=(len(edges), len(used))))
        costs = np.asarray(tile_costs)[:, used]
        return cls(cities, edges, tuple(matrices), costs, metadata or {})

    def track_capital(self, x):
        hsr = np.asarray(self.matrices[1].T @ (x == 2)).ravel() > 0
        cr = (np.asarray(self.matrices[0].T @ (x == 1)).ravel() > 0) & ~hsr
        # Blocked tiles may have infinite costs; never multiply infinity by zero.
        return float(self.tile_costs[1, hsr].sum() + self.tile_costs[0, cr].sum())


def fingerprint(config):
    keys = ('cities_file', 'population_file', 'boundaries_file', 'terrain_file',
            'land_class_file', 'exclusions_file', 'terrain_kind', 'bbox', 'countries',
            'metro_populations_verified', 'prep', 'alignment', 'demand', 'tracks')
    digest = hashlib.sha256(json.dumps({k: config[k] for k in keys}, sort_keys=True).encode())
    digest.update(str(CACHE_VERSION).encode())
    for key in keys:
        if key.endswith('_file') and config[key]:
            with Path(config[key]).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def _projected_shapes(path):
    from shapely.geometry import shape
    from shapely.ops import transform
    raw = json.loads(Path(path).read_text())
    return [transform(TO_METRIC.transform, shape(f['geometry'])) for f in raw['features']]


def cost_surfaces(config, samples):
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling, transform_bounds
    a = config['alignment']
    step = a['cell_km'] * 1000
    extent = transform_bounds('EPSG:4326', 'EPSG:3035', *config['bbox'], densify_pts=41)
    padding = a['padding_km'] * 1000
    left, bottom, right, top = (extent[0] - padding, extent[1] - padding,
                                extent[2] + padding, extent[3] + padding)
    width, height = math.ceil((right - left) / step), math.ceil((top - bottom) / step)
    transform = from_origin(left, top, step, step)
    shape = (height, width)
    rows, cols = np.indices(shape)
    xy = np.column_stack(((left + (cols.ravel() + .5) * step) / 1000,
                          (top - (rows.ravel() + .5) * step) / 1000))
    sample_xy, _, density = samples
    distance, index = cKDTree(sample_xy).query(xy)
    raster_density = np.where(distance <= max(4, a['cell_km']), density[index], 0).reshape(shape)
    blocked = np.zeros(shape, dtype=bool)
    if config['boundaries_file']:
        # All countries form land, independently of the city-country filter.
        land = rasterize([(g, 1) for g in _projected_shapes(config['boundaries_file'])],
                         out_shape=shape, transform=transform, fill=0, dtype='uint8')
        blocked |= land == 0
    if config['exclusions_file']:
        excluded = rasterize([(g, 1) for g in _projected_shapes(config['exclusions_file'])],
                             out_shape=shape, transform=transform, fill=0, dtype='uint8')
        blocked |= excluded > 0
    water = np.zeros(shape, dtype=bool)
    terrain = np.ones(shape)
    grade = np.zeros(shape)
    def warp(path, bands, resampling):
        with rasterio.open(path) as src:
            if not src.crs or src.count < max(bands):
                raise ValueError(f'Invalid georeferenced raster: {path}')
            result = np.full((len(bands), *shape), np.nan, dtype='float32')
            for target, band in zip(result, bands):
                reproject(rasterio.band(src, band), target, src_transform=src.transform,
                          src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform,
                          dst_crs='EPSG:3035', dst_nodata=np.nan, resampling=resampling)
            return result
    if config['terrain_kind'] != 'flat':
        with rasterio.open(config['terrain_file']) as source:
            if config['terrain_kind'] == 'dem' and source.count != 1:
                raise ValueError('DEM terrain requires a single-band elevation raster in metres')
            if config['terrain_kind'] == 'relief' and (source.count != 3 or any(t != 'uint8' for t in source.dtypes)):
                raise ValueError('Relief terrain requires a three-band uint8 RGB raster')
    if config['terrain_kind'] == 'relief':
        r, g, b = warp(config['terrain_file'], [1, 2, 3], Resampling.nearest)
        blocked |= ~np.isfinite(r + g + b)
        water = (b > r * 1.12) & (b > g * 1.04)
        terrain[(r > g * 1.05) & (r > b * 1.15)] = 1.8
    elif config['terrain_kind'] == 'dem':
        elevation = warp(config['terrain_file'], [1], Resampling.average)[0]
        blocked |= ~np.isfinite(elevation)
        # Missing DEM cells are blocked; gradients adjacent to them are blocked too.
        gy, gx = np.gradient(elevation, step)
        grade = np.hypot(gx, gy)
        blocked |= ~np.isfinite(grade)
        terrain = np.select([grade < .01, grade < .02, grade < .03], [1, 1.3, 1.8], default=3.0)
    if config['land_class_file']:
        # Class codes: 0 land, 1 inland water, 2 open sea, 3 excluded/protected.
        classes = warp(config['land_class_file'], [1], Resampling.nearest)[0]
        blocked |= ~np.isfinite(classes) | (classes >= 2)
        water |= classes == 1
    if not config['boundaries_file'] and not config['land_class_file'] and config['terrain_kind'] != 'flat':
        raise ValueError('Sea blocking requires boundaries_file or land_class_file')
    slope = np.select([grade <= .01, grade <= .02, grade <= .03], [1, 1.2, 1.6], default=2.5)
    surfaces = []
    for track in TRACKS:
        t = config['tracks'][track]
        rate = t['base_meur_km'] * terrain * slope + water * a['water_meur_km']
        rate += a['urban_meur_per_density_km'] * np.maximum(0, raster_density - a['urban_reference'])
        rate[blocked | (grade > t['grade_limit'])] = np.inf
        surfaces.append(rate)
    return np.asarray(surfaces), water, grade, transform


def _smooth(points, surface, transform, tolerance):
    from shapely.geometry import LineString
    from rasterio.transform import rowcol
    if len(points) < 3 or tolerance <= 0:
        return points
    simple = np.asarray(LineString(points).simplify(tolerance).coords)
    smooth = [simple[0]]
    for p, q in zip(simple, simple[1:]):
        smooth.extend((.75 * p + .25 * q, .25 * p + .75 * q))
    smooth.append(simple[-1])
    smooth = np.asarray(smooth)
    # Keep smoothing inside the exact priced footprint; never cut a sea corner.
    original_rows, original_cols = rowcol(transform, points[:, 0], points[:, 1])
    footprint = set(zip(original_rows, original_cols))
    for p, q in zip(smooth, smooth[1:]):
        length = np.linalg.norm(q - p)
        for fraction in np.linspace(0, 1, max(2, math.ceil(length / abs(transform.a) * 4))):
            r, c = rowcol(transform, *(p + fraction * (q - p)))
            if (r, c) not in footprint or not np.isfinite(surface[r, c]):
                return points
    return smooth


def route(cities, pairs, config, samples, progress=print):
    from rasterio.transform import rowcol
    from skimage.graph import MCP_Geometric
    surfaces, water, grade, transform = cost_surfaces(config, samples)
    step = config['alignment']['cell_km']
    height, width = surfaces.shape[1:]
    cells = []
    for city in cities:
        r, c = rowcol(transform, *(city.xy * 1000))
        # The nearest city-centre cell is fixed across track types. No snapping
        # across water/exclusions to fabricate a feasible station.
        cells.append((int(r), int(c)))
    records = {pair: {} for pair in pairs}
    for source in sorted({a for a, _ in pairs}):
        targets = [b for a, b in pairs if a == source]
        radius = max(np.linalg.norm(cities[b].xy - cities[source].xy) for b in targets)
        radius = math.ceil(radius * config['alignment']['window_factor'] / step) + 2
        r, c = cells[source]
        r0, r1 = max(0, r - radius), min(height, r + radius + 1)
        c0, c1 = max(0, c - radius), min(width, c + radius + 1)
        for track_index, track in enumerate(TRACKS):
            surface = surfaces[track_index]
            local = surface[r0:r1, c0:c1]
            start = (r - r0, c - c0)
            if not (0 <= r < height and 0 <= c < width) or not np.isfinite(surface[r, c]):
                continue
            ends = [(cells[b][0] - r0, cells[b][1] - c0) for b in targets]
            mcp = MCP_Geometric(local, fully_connected=True, sampling=(step, step))
            costs, _ = mcp.find_costs([start], ends=ends)
            cardinal = cardinal_costs = None
            for target, end in zip(targets, ends):
                if not np.isfinite(costs[end]):
                    continue
                path = np.asarray(mcp.traceback(end)) + (r0, c0)
                # Diagonals must not cut a blocked corner (especially coastlines).
                if len(path) > 1:
                    previous, following = path[:-1], path[1:]
                    if not (np.isfinite(surface[previous[:, 0], following[:, 1]]).all() and
                            np.isfinite(surface[following[:, 0], previous[:, 1]]).all()):
                        if cardinal is None:
                            cardinal = MCP_Geometric(local, fully_connected=False, sampling=(step, step))
                            cardinal_costs, _ = cardinal.find_costs([start], ends=ends)
                        if not np.isfinite(cardinal_costs[end]):
                            continue
                        path = np.asarray(cardinal.traceback(end)) + (r0, c0)
                tiles = np.unique(path[:, 0] * width + path[:, 1]).tolist()
                capex = float(surface.ravel()[tiles].sum() * step)
                if config['alignment']['max_capex_meur'] and capex > config['alignment']['max_capex_meur']:
                    continue
                points = np.column_stack((transform.c + (path[:, 1] + .5) * transform.a,
                                          transform.f + (path[:, 0] + .5) * transform.e))
                # Include station-to-cell offsets in operating distance, not tile capex.
                points = np.vstack((cities[source].xy * 1000, points, cities[target].xy * 1000))
                points = _smooth(points, surface, transform,
                                 config['alignment']['smooth_cells'] * step * 1000)
                length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() / 1000)
                lon, lat = TO_WGS84.transform(points[:, 0], points[:, 1])
                geometry = np.column_stack((lon, lat)).tolist()
                records[(source, target)][track] = Alignment(length, capex, tiles, geometry,
                    float(water.ravel()[tiles].mean()), float((grade.ravel()[tiles] > .02).mean()))
        if progress:
            progress(f'Alignment sweep {source + 1}/{len(cities)}: {cities[source].name}', flush=True)
    edges = [Edge(a, b, records[(a, b)]) for a, b in pairs if records[(a, b)]]
    if not edges:
        raise ValueError('No feasible candidate alignments; check land mask, terrain, and station cells')
    metadata = dict(cache_version=CACHE_VERSION, grid_shape=[height, width],
                    cell_km=step, rejected_pairs=len(pairs) - len(edges),
                    terrain_kind=config['terrain_kind'],
                    metro_populations_verified=config['metro_populations_verified'],
                    tile_accounting='fixed cell_km per occupied cell; HSR overrides CR')
    return CandidateGraph.build(cities, edges, surfaces.reshape(2, -1) * step, metadata)


def save_graph(graph, directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i, edge in enumerate(graph.edges):
        for track, alignment in edge.alignments.items():
            records.append(dict(edge_id=i, node_a=edge.a, node_b=edge.b, track_type=track,
                                feasible=True, **asdict(alignment)))
    pq.write_table(pa.Table.from_pylist(records), directory / 'edges.parquet')
    for track, matrix in zip(TRACKS, graph.matrices):
        sparse.save_npz(directory / f'tiles_{track}.npz', matrix)
    np.save(directory / 'tile_costs.npy', graph.tile_costs, allow_pickle=False)
    (directory / 'manifest.json').write_text(json.dumps(dict(
        metadata=graph.metadata, cities=[asdict(c) for c in graph.cities]), indent=2))


def load_graph(directory):
    import pyarrow.parquet as pq
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['metadata']['cache_version'] != CACHE_VERSION:
        raise ValueError('Unsupported alignment cache version')
    by_id = {}
    for r in pq.read_table(directory / 'edges.parquet').to_pylist():
        edge = by_id.setdefault(r['edge_id'], Edge(r['node_a'], r['node_b'], {}))
        edge.alignments[r['track_type']] = Alignment(**{k: r[k] for k in Alignment.__dataclass_fields__})
    return CandidateGraph([City(**c) for c in manifest['cities']], [by_id[i] for i in sorted(by_id)],
        tuple(sparse.load_npz(directory / f'tiles_{t}.npz') for t in TRACKS),
        np.load(directory / 'tile_costs.npy', allow_pickle=False), manifest['metadata'])


def cached_graph(config, cache_dir, progress=print):
    from .prep import prepare
    key = fingerprint(config)
    directory = Path(cache_dir) / key
    if directory.exists():
        if progress:
            progress(f'Loading cached alignments: {directory}', flush=True)
        return load_graph(directory)
    cities, pairs, samples = prepare(config)
    graph = route(cities, pairs, config, samples, progress)
    graph.metadata['fingerprint'] = key
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir, prefix='.alignments-') as temporary:
        staging = Path(temporary) / 'graph'
        save_graph(graph, staging)
        staging.rename(directory)
    return graph
