"""Explicit shared crossing assets and routing over land plus portal links."""
import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .prep import TO_METRIC


def crossing_cost(lengths, rates, water, growth_km):
    """Legacy continuous-water growth, invariant to subdivision of a segment."""
    total = run_length = run_base = 0.0
    for length, rate, wet in zip(lengths, rates, water):
        base = float(length * rate)
        total += base
        if wet:
            run_length += length
            run_base += base
        else:
            total += run_base * math.expm1(min(100, run_length / growth_km))
            run_length = run_base = 0.0
    return total + run_base * math.expm1(min(100, run_length / growth_km))


def prepare_crossings(config, surfaces, water, transform):
    """Sample configured geometry; never snap portals through invalid cells."""
    from rasterio.transform import rowcol
    a = config['alignment']
    if not a['sea_crossings_enabled']:
        return []
    height, width = water.shape
    assets = []
    for item in a['water_crossings']:
        xy = np.array([TO_METRIC.transform(*p) for p in item['geometry']]) / 1000
        points = []
        for start, end in zip(xy, xy[1:]):
            count = max(1, math.ceil(np.linalg.norm(end - start) / a['water_sample_km']))
            points.extend(start + np.arange(count)[:, None] / count * (end - start))
        points.append(xy[-1])
        points = np.asarray(points)
        # Include the exact portal-to-cell-centre connectors in cost and geometry.
        pr, pc = rowcol(transform, points[[0, -1], 0] * 1000, points[[0, -1], 1] * 1000)
        pr, pc = np.asarray(pr), np.asarray(pc)
        if np.any((pr < 0) | (pr >= height) | (pc < 0) | (pc >= width)):
            continue  # Configured asset lies outside this routing extent.
        if water[pr, pc].any():
            raise ValueError(f"Water crossing {item['id']}: portals must be on land; move the portal inland")
        centers = np.column_stack(((transform.c + (pc + .5) * transform.a) / 1000,
                                    (transform.f + (pr + .5) * transform.e) / 1000))
        points = np.vstack((centers[0], points, centers[1]))
        # Resample connectors too, so exclusions cannot be skipped.
        sampled = []
        for start, end in zip(points, points[1:]):
            count = max(1, math.ceil(np.linalg.norm(end - start) / a['water_sample_km']))
            fractions = list(np.arange(count) / count)
            # Split at raster boundaries as well as sample spacing. Water length
            # and exclusions then do not depend on where input vertices fall.
            for axis, origin, pixel in [(0, transform.c / 1000, transform.a / 1000),
                                         (1, transform.f / 1000, transform.e / 1000)]:
                if abs(end[axis] - start[axis]) < 1e-12:
                    continue
                lo, hi = sorted(((start[axis] - origin) / pixel, (end[axis] - origin) / pixel))
                for boundary in range(math.floor(lo) + 1, math.ceil(hi)):
                    fraction = (origin + boundary * pixel - start[axis]) / (end[axis] - start[axis])
                    if 0 < fraction < 1:
                        fractions.append(fraction)
            sampled.extend(start + np.array(sorted(set(fractions)))[:, None] * (end - start))
        sampled.append(points[-1])
        points = np.asarray(sampled)
        mid = (points[:-1] + points[1:]) / 2
        rr, cc = rowcol(transform, mid[:, 0] * 1000, mid[:, 1] * 1000)
        rr, cc = np.asarray(rr), np.asarray(cc)
        if np.any((rr < 0) | (rr >= height) | (cc < 0) | (cc >= width)):
            continue
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        wet = water[rr, cc]
        costs = []
        for surface in surfaces:
            rates = surface[rr, cc]
            costs.append(crossing_cost(lengths, rates, wet, a['water_growth_km'])
                         if np.isfinite(rates).all() and np.isfinite(surface[pr, pc]).all() else math.inf)
        assets.append(dict(id=item['id'], start=int(pr[0] * width + pc[0]),
            end=int(pr[1] * width + pc[1]), points=points, length_km=float(lengths.sum()),
            water_km=float(lengths[wet].sum()), costs=costs))
    return assets


def land_link_graph(surface, step, assets, track, bounds, full_width):
    """Sparse raster graph with one virtual resource node per physical crossing."""
    r0, r1, c0, c1 = bounds
    local = surface[r0:r1, c0:c1]
    h, w = local.shape
    ids = np.arange(h * w).reshape(h, w)
    rows, cols, values = [], [], []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ra = slice(0, h - dr)
        ca = slice(max(0, -dc), min(w, w - dc))
        rb = slice(dr, h)
        cb = slice(max(0, dc), min(w, w + dc))
        x, y = local[ra, ca], local[rb, cb]
        valid = np.isfinite(x) & np.isfinite(y)
        if dr and dc:
            valid &= np.isfinite(local[ra, cb]) & np.isfinite(local[rb, ca])
        u, v = ids[ra, ca][valid], ids[rb, cb][valid]
        cost = (x[valid] + y[valid]) * .5 * step * math.hypot(dr, dc)
        rows.extend([u, v]); cols.extend([v, u]); values.extend([cost, cost])
    for index, asset in enumerate(assets):
        endpoints = [divmod(asset[k], full_width) for k in ('start', 'end')]
        if not math.isfinite(asset['costs'][track]) or not all(r0 <= r < r1 and c0 <= c < c1 for r, c in endpoints):
            continue
        u, v = [(r - r0) * w + c - c0 for r, c in endpoints]
        if not np.isfinite(local.ravel()[[u, v]]).all():
            continue
        node = h * w + index
        rows.append(np.array([u, node, v, node])); cols.append(np.array([node, v, node, u]))
        values.append(np.full(4, max(asset['costs'][track] / 2, 1e-12)))
    graph = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(h * w + len(assets),) * 2)
    return graph


def linked_paths(surface, step, assets, track, bounds, full_width, start, ends):
    graph = land_link_graph(surface, step, assets, track, bounds, full_width)
    width = bounds[3] - bounds[2]
    distances, previous = dijkstra(graph, indices=start[0] * width + start[1], return_predecessors=True)
    paths = []
    for r, c in ends:
        node = r * width + c
        if not np.isfinite(distances[node]):
            paths.append(None)
            continue
        path = [node]
        while previous[path[-1]] >= 0:
            path.append(int(previous[path[-1]]))
        paths.append(path[::-1])
    return paths
