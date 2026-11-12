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
    population_collection, satisfaction_scale, track_style, add_track_reference)
from train_optimizer.outputs import city_label_indices, map_figure_size, map_projector
from train_optimizer.map_terrain import fast_terrain, add_terrain
from .alignment import TRACKS
from .track_layout import track_type
from .evaluator import Evaluator


class MapBackground:
    """Prepare population geometry, projection and fixed scales once per run."""
    def __init__(self, graph, config):
        self.graph, self.config = graph, config
        self.project = map_projector(SimpleNamespace(bbox=config['bbox']))
        self.evaluator = Evaluator(graph, config)
        self.terrain = fast_terrain(config)
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
        radii = np.asarray([city.radius_km for city in graph.cities])[:, None]
        lower = np.minimum(lower, (self.node_xy - radii).min(axis=0))
        upper = np.maximum(upper, (self.node_xy + radii).max(axis=0))
        margin = max(5, float((upper - lower).max()) * .03)
        self.viewport = (float(lower[0] - margin), float(upper[0] + margin),
                         float(lower[1] - margin), float(upper[1] + margin))
        self.figure_size = map_figure_size(self.viewport)
        self.labelled_cities = city_label_indices(
            [dict(population=city.population) for city in graph.cities])

    def satisfaction(self, result):
        """Carried trips / fixed capturable OD markets using each edge, transfers included.

        The fast model's eligible market is its annual gravity base (after fixed
        station capture and distance screening), before rail share and spill.
        Each unordered OD is counted once in both numerator and denominator,
        using two-way annual units. The denominator is independent of frequency.
        """
        from .analysis import corridor_satisfaction
        return corridor_satisfaction(self.evaluator, result)[0]

    def draw(self, result, path, title, color_by='satisfaction', *, reevaluate=True):
        if color_by not in {'satisfaction', 'revenue_cost'}:
            raise ValueError('Unknown map coloring: ' + color_by)
        financial = color_by == 'revenue_cost'
        if financial:
            financial_result = (self.evaluator.evaluate(result.genome, details=True,
                services=result.service_plan, track_counts=result.track_counts) if reevaluate else result)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Use the original fixed axes positions: colorbars cannot shift the map
        # when the network changes or a local matplotlibrc enables tight layout.
        figure = plt.figure(figsize=self.figure_size, layout='none')
        ax = figure.add_axes([.09, .16, .72, .72])
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
            add_terrain(ax, figure, self.terrain, self.project)
            if financial:
                from matplotlib.colors import LinearSegmentedColormap, Normalize
                cmap = LinearSegmentedColormap.from_list('cost_recovery', ['#ad3035', '#edca79', '#124625'])
                norm = Normalize(0, 2, clip=True)
                colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ratio_axis,
                    label='Annual revenue / cost', ticks=[0, 1, 2], extend='max')
                colorbar.ax.set_yticklabels(['0', '1', '≥2'])
            else:
                cmap, norm = satisfaction_scale()
                figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ratio_axis,
                                label='Rail-market satisfaction')
            for i, city in enumerate(self.graph.cities):
                xy = self.node_xy[i]
                ax.plot(*xy, **CITY_STYLE)
                if i in self.labelled_cities:
                    ax.annotate(city.name, xy, xytext=(4, 5), textcoords='offset points', fontsize=8, zorder=6)
            if financial:
                costs = financial_result.edge_revenue - financial_result.edge_net
                ratios = np.divide(financial_result.edge_revenue, costs, out=np.full(len(costs), np.nan), where=costs > 0)
            else:
                ratios = self.satisfaction(result)
            served = set()
            for i in np.flatnonzero(result.genome):
                edge = self.graph.edges[i]
                track = track_type(result.genome[i])
                coordinates = self.edge_xy[i, track]
                style = track_style(result.frequency[i], track, 0 if financial else ratios[i])
                if financial:
                    style['color'] = cmap(norm(ratios[i])) if np.isfinite(ratios[i]) else '#777777'
                ax.plot(coordinates[:, 0], coordinates[:, 1], **style)
                served.update((edge.a, edge.b))
            for i in sorted(served):
                xy = self.node_xy[i]
                ax.add_patch(Circle(xy, self.graph.cities[i].radius_km, **CATCHMENT_STYLE))
                ax.plot(*xy, **STATION_STYLE)
            xmin, xmax, ymin, ymax = self.viewport
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
                   xlabel='East of regional center (km)', ylabel='North of regional center (km)')
            ax.set_aspect('equal')
            ax.grid(alpha=.15)
            heading = 'Rail network optimization · revenue / cost' if financial else 'Rail network optimization'
            ax.set_title(heading + '\n' + title, fontsize=11)
            add_track_reference(figure)
            ax.text(.01, .01, 'Symbols: infrastructure · Dots: HSR on CR (violet), CR on HSR (orange)',
                    transform=ax.transAxes, fontsize=7, va='bottom', bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
            # Overlay train types only where they differ from infrastructure.
            for service in result.services:
                for edge_id in service['edge_ids']:
                    infrastructure = track_type(result.genome[edge_id])
                    if service['train_type'] != infrastructure:
                        coordinates = self.edge_xy[edge_id, infrastructure]
                        color = '#7936ac' if service['train_type'] == 'HSR' else '#e48319'
                        ax.plot(coordinates[:, 0], coordinates[:, 1], color=color,
                                linestyle=':', linewidth=1.1, zorder=4.5)
            color_label = ('Color: annual revenue / allocated annual cost · Red: below break-even · Green: surplus · Gray: undefined.'
                           if financial else 'Color: carried passengers / fixed capturable OD market using each segment (including transfers).')
            figure.supxlabel(color_label + '\nWhite-dashed: HSR · Solid: CR · White dots: stations · Dashed green: catchments', fontsize=8, y=.005)
            with matplotlib.rc_context({'savefig.bbox': None}):
                figure.savefig(path, dpi=130)
        finally:
            plt.close(figure)
