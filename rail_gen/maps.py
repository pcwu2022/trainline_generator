"""Population-density maps for railway networks."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import h3
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from rail_gen.map_style import (CITY_STYLE, STATION_STYLE, CATCHMENT_STYLE,
    population_collection, satisfaction_scale, track_style)
from rail_gen.map_utils import city_label_indices, map_figure_size, map_projector
from rail_gen.map_elevation import elevation_hexagons, add_elevation
from .alignment import TRACKS
from .track_layout import track_type
from .evaluator import Evaluator


class MapBackground:
    """Prepare population geometry, projection and fixed scales once per run."""
    def __init__(self, graph, config):
        self.graph, self.config = graph, config
        self.project = map_projector(SimpleNamespace(bbox=config['bbox']))
        self.evaluator = Evaluator(graph, config)
        self.elevation = elevation_hexagons(config, self.project)
        west, south, east, north = config['bbox']
        uk_mainland = config['countries'] == ['GB']
        if uk_mainland:
            # Frame Great Britain and Northern Ireland, leaving Orkney and
            # Shetland out of the map without changing the simulation region.
            west, south, east, north = -8.3, 49.8, 1.9, 58.75
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
                self.country_lines.extend(self.project(np.asarray(p.exterior.coords)) for p in polygons
                                          if not uk_mainland or p.bounds[1] <= north)
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
        self.station_xy = self.project([[
            city.lon if city.station_lon is None else city.station_lon,
            city.lat if city.station_lat is None else city.station_lat]
            for city in graph.cities])
        self.edge_xy = {(i, track): self.project(alignment.geometry)
                        for i, edge in enumerate(graph.edges) for track, alignment in edge.alignments.items()}
        lon, lat = np.linspace(west, east, 101), np.linspace(south, north, 101)
        perimeter = self.project([(x, south) for x in lon] + [(x, north) for x in lon]
                                 + [(west, y) for y in lat] + [(east, y) for y in lat])
        points = np.vstack([perimeter, *self.polygons, *self.edge_xy.values()])
        lower, upper = points.min(axis=0), points.max(axis=0)
        radii = np.asarray([city.radius_km for city in graph.cities])[:, None]
        lower = np.minimum(lower, (self.station_xy - radii).min(axis=0))
        upper = np.maximum(upper, (self.station_xy + radii).max(axis=0))
        # Reserve extra horizontal space only to the east for the overlays. A square
        # projected extent fills the canvas without stretching the geography.
        span = upper - lower
        side = max(float(span[0]) / .75, float(span[1]) / (.94 if uk_mainland else .88), 10.)
        left = float(lower[0])
        if config['countries'] == ['TW']:
            # Taiwan's tall shape leaves spare horizontal space on a square
            # canvas. Center the island in the area west of the legend rather
            # than anchoring it to the western edge of the sampling bounds.
            island = np.vstack([*self.polygons, *self.edge_xy.values(),
                                self.station_xy - radii, self.station_xy + radii])
            island_center = float((island[:, 0].min() + island[:, 0].max()) / 2)
            left = island_center - side * .375
        # Keep every region's output square; reserve overlay space inside it.
        self.right_trim = 0.
        if uk_mainland:
            # Balance the visible land between the canvas edge and the legend.
            # Include the panel's rounded-box padding in its actual left edge.
            land = np.vstack(self.polygons)
            land_width = float(np.ptp(land[:, 0]))
            side = max(side, land_width / .679)
            margin = (.759 * side - land_width) / 2
            left = float(land[:, 0].min()) - margin
        bottom = float((lower[1] + upper[1] - side) / 2)
        width = side * (1 - self.right_trim)
        self.viewport = (left, left + width, bottom, bottom + side)
        square_size = map_figure_size(self.viewport)[0]
        self.figure_size = (square_size * (1 - self.right_trim), square_size)
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
        figure = plt.figure(figsize=self.figure_size, layout='none')
        figure.set_facecolor('#eef2f5')
        ax = figure.add_axes([0, 0, 1, 1])
        def panel_x(x):
            return (x - self.right_trim) / (1 - self.right_trim)

        panel_scale = 1 / (1 - self.right_trim)
        # All map keys share one floating panel in the reserved eastern strip.
        panel = FancyBboxPatch((panel_x(.765), .16), .22 * panel_scale, .385, transform=figure.transFigure,
            boxstyle='round,pad=0.006', facecolor='white', edgecolor='#bac3c8',
            alpha=.94, linewidth=.7, zorder=8)
        figure.add_artist(panel)
        density_axis = figure.add_axes([panel_x(.785), .485, .177 * panel_scale, .012], zorder=9)
        ratio_axis = figure.add_axes([panel_x(.785), .415, .177 * panel_scale, .012], zorder=9)
        try:
            ax.set_facecolor('#eef2f5')
            for coordinates in self.country_lines:
                ax.plot(coordinates[:, 0], coordinates[:, 1], color='#c2c8c5', linewidth=.6, zorder=0)
            collection = population_collection(self.polygons, self.densities)
            ax.add_collection(collection)
            figure.text(panel_x(.785), .51, 'Population / km²', fontsize=8, zorder=10)
            figure.colorbar(collection, cax=density_axis, orientation='horizontal')
            density_axis.tick_params(labelsize=7, length=2)
            density_axis.minorticks_off()
            figure.text(panel_x(.785), .44, 'Revenue / cost' if financial else 'Demand fulfillment',
                        fontsize=8, zorder=10)
            elevation = add_elevation(ax, self.elevation)
            if elevation is not None:
                elevation_axis = figure.add_axes([panel_x(.785), .345, .177 * panel_scale, .012], zorder=9)
                figure.text(panel_x(.785), .37, 'Elevation (m)', fontsize=7, zorder=10)
                figure.colorbar(elevation, cax=elevation_axis, orientation='horizontal',
                                ticks=[0, 1000, 2000, 3000, 4000], extend='both')
                elevation_axis.tick_params(labelsize=7, length=2)
            if financial:
                from matplotlib.colors import LinearSegmentedColormap, Normalize
                cmap = LinearSegmentedColormap.from_list('cost_recovery', ['#ad3035', '#edca79', '#124625'])
                norm = Normalize(0, 2, clip=True)
                colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ratio_axis,
                    orientation='horizontal', ticks=[0, 1, 2], extend='max')
                colorbar.ax.set_xticklabels(['0', '1', '≥2'])
            else:
                cmap, norm = satisfaction_scale()
                figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ratio_axis,
                                orientation='horizontal', ticks=[0, .5, 1])
                from matplotlib.ticker import PercentFormatter
                ratio_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ratio_axis.tick_params(labelsize=7, length=2)
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
                xy = self.station_xy[i]
                ax.add_patch(Circle(xy, self.graph.cities[i].radius_km, **CATCHMENT_STYLE))
                ax.plot(*xy, **STATION_STYLE)
            xmin, xmax, ymin, ymax = self.viewport
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.text(.5, .5, '@maps.az', transform=ax.transAxes,
                    ha='center', va='center', fontsize=32, fontweight='bold',
                    color='#c4c4c4', alpha=.2, zorder=7)
            ax.text(.025, .975,
                    f'{self.config["reporting"]["region_name"]} Rail Network Generation',
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    color='#16334a', va='top', linespacing=1.3, zorder=10,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
            ax.text(.975, .975, title, transform=ax.transAxes, fontsize=10,
                    ha='right', va='top', multialignment='left', linespacing=1.5, zorder=10,
                    bbox=dict(boxstyle='round,pad=.5', facecolor='white', alpha=.9,
                              edgecolor='#bac3c8', linewidth=.7))
            handles = [Line2D([], [], label=f'Every {60 / frequency:g} mins',
                       **track_style(frequency, 'CR', 1)) for frequency in (1, 2, 4, 8)]
            handles += [Line2D([], [], label='High Speed Rail', **track_style(4, 'HSR', 1)),
                Line2D([], [], color='#7936ac', linestyle=':', label='Hybrid'),
                Line2D([], [], label='Station', linestyle='none', **STATION_STYLE)]
            legend = figure.legend(handles=handles, loc='upper left',
                bbox_to_anchor=(panel_x(.775), .30 if elevation is not None else .38), fontsize=7, frameon=False,
                handlelength=2.5, labelspacing=.7, borderaxespad=0)
            legend.set_zorder(10)
            # Text sizes are in points while the canvas varies by region.
            # Measure the legend so its last row always has panel padding.
            figure.canvas.draw()
            legend_bottom = legend.get_window_extent(
                figure.canvas.get_renderer()).transformed(figure.transFigure.inverted()).y0
            panel_bottom = min(.16, legend_bottom - .012)
            panel.set_bounds(panel_x(.765), panel_bottom, .22 * panel_scale, .545 - panel_bottom)
            # Overlay train types only where they differ from infrastructure.
            for service in result.services:
                for edge_id in service['edge_ids']:
                    infrastructure = track_type(result.genome[edge_id])
                    if service['train_type'] == 'HSR' and infrastructure == 'CR':
                        coordinates = self.edge_xy[edge_id, infrastructure]
                        color = '#7936ac'
                        ax.plot(coordinates[:, 0], coordinates[:, 1], color=color,
                                linestyle=':', linewidth=1.1, zorder=4.5)
            with matplotlib.rc_context({'savefig.bbox': None}):
                figure.savefig(path, dpi=130)
        finally:
            plt.close(figure)
