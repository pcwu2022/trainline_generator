"""Population-density maps with the original optimizer's shared visual design."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import h3
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from train_optimizer.map_style import (CITY_STYLE, STATION_STYLE, CATCHMENT_STYLE,
    population_collection, satisfaction_scale, track_style)
from train_optimizer.outputs import map_figure_size, map_projector
from .alignment import TRACKS
from .evaluator import Evaluator


class MapBackground:
    """Prepare population geometry, projection and fixed scales once per run."""
    def __init__(self, graph, config):
        self.graph, self.config = graph, config
        self.project = map_projector(SimpleNamespace(bbox=config['bbox']))
        self.evaluator = Evaluator(graph, config)
        west, south, east, north = config['bbox']
        self.country_lines = []
        region = None
        if config['boundaries_file']:
            features = json.loads(Path(config['boundaries_file']).read_text())['features']
            selected = [shape(f['geometry']) for f in features
                        if f['properties'].get('iso_a2') in config['countries']]
            if selected:
                region = prep(unary_union(selected))
            for feature in features:
                geometry = shape(feature['geometry'])
                w, s, e, n = geometry.bounds
                if e < west or w > east or n < south or s > north:
                    continue
                polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
                self.country_lines.extend(self.project(np.asarray(p.exterior.coords)) for p in polygons)
        self.polygons, densities = [], []
        features = json.loads(Path(config['population_file']).read_text())['features']
        for feature in features:
            geometry, properties = feature['geometry'], feature['properties']
            if geometry['type'] == 'Point':
                lon, lat = geometry['coordinates'][:2]
            else:
                center = shape(geometry).centroid
                lon, lat = center.x, center.y
            if not (west <= lon <= east and south <= lat <= north):
                continue
            if region is not None and not region.covers(Point(lon, lat)):
                continue
            population = float(properties['population'])
            index = properties.get('h3Index')
            area = h3.cell_area(index, unit='km^2') if index else float(properties['area_km2'])
            if not math.isfinite(population) or population < 0 or not math.isfinite(area) or area <= 0:
                raise ValueError('Population map requires finite nonnegative population and positive cell area')
            density = population / area
            if index:
                boundary = [(x, y) for y, x in h3.cell_to_boundary(index)]
                polygons = [self.project(boundary)]
            elif geometry['type'] in ('Polygon', 'MultiPolygon'):
                polygon = shape(geometry)
                parts = polygon.geoms if polygon.geom_type == 'MultiPolygon' else [polygon]
                polygons = [self.project(np.asarray(part.exterior.coords)) for part in parts]
            elif geometry['type'] == 'Point':
                # Non-H3 point cells use their declared area as a square footprint.
                center = self.project([[lon, lat]])[0]
                half = math.sqrt(area) / 2
                polygons = [center + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])]
            else:
                raise ValueError('Population map supports Point, Polygon and MultiPolygon cells')
            self.polygons.extend(polygons)
            densities.extend([density] * len(polygons))
        if not self.polygons:
            raise ValueError('No population cells within the map bounding box/country selection')
        self.densities = np.asarray(densities)
        self.node_xy = self.project([[city.lon, city.lat] for city in graph.cities])
        self.edge_xy = {(i, track): self.project(alignment.geometry)
                        for i, edge in enumerate(graph.edges) for track, alignment in edge.alignments.items()}
        lon, lat = np.linspace(west, east, 101), np.linspace(south, north, 101)
        perimeter = self.project([(x, south) for x in lon] + [(x, north) for x in lon]
                                 + [(west, y) for y in lat] + [(east, y) for y in lat])
        points = np.vstack([perimeter, *self.polygons, *self.edge_xy.values()])
        lower, upper = points.min(axis=0), points.max(axis=0)
        radius = config['prep']['catchment_km']
        lower = np.minimum(lower, self.node_xy.min(axis=0) - radius)
        upper = np.maximum(upper, self.node_xy.max(axis=0) + radius)
        margin = max(5, float((upper - lower).max()) * .03)
        self.viewport = (float(lower[0] - margin), float(upper[0] + margin),
                         float(lower[1] - margin), float(upper[1] + margin))
        self.figure_size = map_figure_size(self.viewport)
        self.labelled_cities = set(sorted(range(len(graph.cities)),
                                         key=lambda i: -graph.cities[i].population)[:20])

    def satisfaction(self, result):
        """Carried trips / fixed capturable OD markets using each edge, transfers included.

        The fast model's eligible market is its annual gravity base (after fixed
        station capture and distance screening), before rail share and spill.
        Each unordered OD is counted once in both numerator and denominator,
        using two-way annual units. The denominator is independent of frequency.
        """
        e = self.evaluator
        active = np.flatnonzero(result.genome)
        length = e.lengths[np.arange(e.m), result.genome]
        speeds = np.choose(result.genome, [1, self.config['tracks']['CR']['speed_kmh'],
                                         self.config['tracks']['HSR']['speed_kmh']])
        time = length / speeds + (result.genome > 0) * self.config['operations']['dwell_hours']
        incidence, _, _, _ = e._paths(active, time)
        eligible = np.asarray(incidence @ np.where(result.trips > 0, 2 * e.base, 0)).ravel()
        ratios = np.divide(2 * result.directional_load, eligible,
                           out=np.zeros(e.m), where=eligible > 0)
        return np.clip(ratios, 0, 1)

    def draw(self, result, path, title):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Use the original fixed axes positions: colorbars cannot shift the map
        # when the network changes or a local matplotlibrc enables tight layout.
        figure = plt.figure(figsize=self.figure_size, layout='none')
        ax = figure.add_axes([.09, .11, .72, .77])
        density_axis = figure.add_axes([.85, .28, .025, .46])
        ratio_axis = figure.add_axes([.85, .08, .025, .13])
        try:
            ax.set_facecolor('#eef2f5')
            for coordinates in self.country_lines:
                ax.plot(coordinates[:, 0], coordinates[:, 1], color='#c2c8c5', linewidth=.6, zorder=0)
            collection = population_collection(self.polygons, self.densities)
            ax.add_collection(collection)
            figure.colorbar(collection, cax=density_axis,
                            label='Population density (people/km², log scale)')
            cmap, norm = satisfaction_scale()
            figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ratio_axis,
                            label='Rail-market satisfaction')
            for i, city in enumerate(self.graph.cities):
                xy = self.node_xy[i]
                ax.plot(*xy, **CITY_STYLE)
                if i in self.labelled_cities:
                    ax.annotate(city.name, xy, xytext=(4, 5), textcoords='offset points', fontsize=8, zorder=6)
            ratios = self.satisfaction(result)
            served = set()
            for i in np.flatnonzero(result.genome):
                edge = self.graph.edges[i]
                track = TRACKS[result.genome[i] - 1]
                coordinates = self.edge_xy[i, track]
                ax.plot(coordinates[:, 0], coordinates[:, 1],
                        **track_style(result.frequency[i], track, ratios[i]))
                served.update((edge.a, edge.b))
            for i in sorted(served):
                xy = self.node_xy[i]
                ax.add_patch(Circle(xy, self.config['prep']['catchment_km'], **CATCHMENT_STYLE))
                ax.plot(*xy, **STATION_STYLE)
            xmin, xmax, ymin, ymax = self.viewport
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
                   xlabel='East of regional center (km)', ylabel='North of regional center (km)')
            ax.set_aspect('equal')
            ax.grid(alpha=.15)
            ax.set_title('Rail network optimization\n' + title, fontsize=11)
            figure.supxlabel('Color: carried passengers / fixed capturable OD market using each segment (including transfers).\n'
                'Solid: high-speed · Dashed: conventional · Width: trains/hour/direction\n'
                'White dots: stations · Dashed green circles: catchments', fontsize=8)
            with matplotlib.rc_context({'savefig.bbox': None}):
                figure.savefig(path, dpi=130)
        finally:
            plt.close(figure)
