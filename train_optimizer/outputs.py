"""Iteration-triggered live outputs and a bounded-memory progress GIF."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import shutil
import threading
import time

import h3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
import numpy as np
from PIL import GifImagePlugin, Image
from pyproj import Transformer
from .cost_log import cost_breakdown_log


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf8')
    temporary.replace(path)


def network_geojson(result):
    features = []
    for station in result.get('stations', []):
        features.append(dict(type='Feature', geometry=dict(type='Point', coordinates=station['coordinate']),
                             properties=dict(feature_kind='station', **{k: v for k, v in station.items() if k != 'coordinate'})))
    for track in result.get('tracks', []):
        features.append(dict(type='Feature', geometry=dict(type='LineString', coordinates=track['geometry']),
                             properties=dict(feature_kind='track', **{k: v for k, v in track.items() if k != 'geometry'})))
    for line in result.get('lines', []):
        features.append(dict(type='Feature', geometry=dict(type='LineString', coordinates=line['geometry']),
                             properties=dict(feature_kind='service_line', **{k: v for k, v in line.items() if k != 'geometry'})))
    return dict(type='FeatureCollection', features=features)


def city_label_indices(cities):
    """Label the twenty largest cities, preserving all city markers."""
    return set(sorted(range(len(cities)),
                      key=lambda i: -float(cities[i].get('population', cities[i].get('pop', 0))))[:20])


def map_projector(config):
    west, south, east, north = config.bbox
    crs = f'+proj=aeqd +lat_0={(south+north)/2} +lon_0={(west+east)/2} +datum=WGS84 +units=m'
    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    def project(coordinates):
        array = np.asarray(coordinates, dtype=float)
        if len(array) == 1:
            x, y = transformer.transform(float(array[0, 0]), float(array[0, 1]))
            return np.array([[x/1000, y/1000]])
        x, y = transformer.transform(array[:, 0], array[:, 1])
        return np.column_stack((x, y))/1000
    return project


def map_viewport(config, problem):
    """Fit the complete, fixed candidate geography, never the current solution.

    Sample the bbox perimeter because a projected geographic rectangle can bow
    beyond its four projected corners. Include whole population hexagons and
    candidate corridors/catchments, which can extend beyond the input bbox.
    """
    west, south, east, north = config.bbox
    lon = np.linspace(west, east, 101)
    lat = np.linspace(south, north, 101)
    coordinates = ([(x, south) for x in lon] + [(x, north) for x in lon]
                   + [(west, y) for y in lat] + [(east, y) for y in lat])
    if problem is not None:
        for cell in problem.cells:
            coordinates.extend((lon, lat) for lat, lon in h3.cell_to_boundary(cell['h3']))
        coordinates.extend(city['center'] for city in problem.cities)
        coordinates.extend(location['coordinate'] for location in problem.locations)
        for edge in problem.edges:
            coordinates.extend(edge['geometry'])
    project = map_projector(config)
    projected = project(coordinates)
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    if problem is not None:
        for station in problem.stations:
            point = project([problem.locations[station['location']]['coordinate']])[0]
            lower = np.minimum(lower, point-station['radius_km'])
            upper = np.maximum(upper, point+station['radius_km'])
    margin = max(5, float((upper-lower).max())*0.03)
    return (float(lower[0]-margin), float(upper[0]+margin),
            float(lower[1]-margin), float(upper[1]+margin))


def map_figure_size(viewport):
    """Square canvas growing with sqrt of the longest projected span in km.

    A 1,000 km span uses 10 inches; an 8-inch floor keeps labels legible.
    Use the fixed viewport so solution changes cannot resize animation frames.
    """
    xmin, xmax, ymin, ymax = viewport
    side = max(8.0, 10.0 * math.sqrt(max(xmax-xmin, ymax-ymin) / 1000.0))
    return side, side


def segment_satisfaction(problem, result):
    """Full OD demand behind each used segment, counting each OD once.

    Flow records include transfer legs. Sum both travel directions consistently
    in numerator and denominator; unused segments have zero satisfaction.
    """
    demand = {od['id']: od['volume'] for od in problem.demand}
    lines = {line['id']: line for line in result.get('lines', [])}
    loads, markets = {}, {}
    for flow in result.get('passenger_flows', []):
        if flow['line'] not in lines:
            continue
        for segment in range(*sorted((flow['board_stop'], flow['alight_stop']))):
            key = (flow['line'], segment)
            loads[key] = loads.get(key, 0) + flow['trips_per_hour']
            markets.setdefault(key, set()).add(flow['od'])
    return {key: loads[key] / sum(demand[od] for od in ods)
            for key, ods in markets.items() if sum(demand[od] for od in ods) > 0}


def demand_connections(problem, config):
    pairs = {}
    for od in problem.demand:
        key = tuple(sorted((od['origin'], od['destination'])))
        pairs[key] = pairs.get(key, 0) + od['volume']
    ranked = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
    # Never draw the complete graph, even when the configured percentile is 100.
    limit = min(max(0, len(ranked)-1), math.ceil(len(ranked)*config.demand_map_top_percent/100))
    return [(pair, volume) for pair, volume in ranked[:limit]
            if volume > 0 and volume >= config.demand_map_min_trips_per_hour]


def draw_demand_map(path, config, problem, viewport=None):
    draw_map(path, config, problem, {}, viewport=viewport, demand_only=True)


def draw_map(path, config, problem, result, viewport=None, demand_only=False):
    project = map_projector(config)
    viewport = viewport if viewport is not None else map_viewport(config, problem)
    # Explicit axes prevent the colorbar from stealing space from the map.
    # Disable automatic layout/cropping even when enabled in local matplotlibrc.
    figure = plt.figure(figsize=map_figure_size(viewport), layout='none')
    ax = figure.add_axes([0.09, 0.11, 0.72, 0.77])
    color_axis = figure.add_axes([0.85, 0.28, 0.025, 0.46])
    color_axis.set_visible(problem is not None)
    ax.set_facecolor('#eef2f5')
    if problem:
        polygons, values = [], []
        for cell in problem.cells:
            boundary = [(lon, lat) for lat, lon in h3.cell_to_boundary(cell['h3'])]
            polygons.append(project(boundary))
            values.append(max(0.1, cell['density']))
        collection = PolyCollection(polygons, array=np.asarray(values), cmap='YlOrRd',
                                    norm=LogNorm(vmin=1, vmax=max([10] + values)),
                                    edgecolors='#ffffff', linewidths=0.2, alpha=0.65)
        ax.add_collection(collection)
        figure.colorbar(collection, cax=color_axis,
                        label='Population density (people/km², log scale)')
        labeled_cities = city_label_indices(problem.cities)
        for city_index, city in enumerate(problem.cities):
            xy = project([city['center']])[0]
            ax.plot(*xy, marker='+', color='#5d6370', markersize=7, zorder=3)
            if city_index in labeled_cities:
                ax.annotate(city['name'], xy, xytext=(4, 5), textcoords='offset points', fontsize=8, zorder=6)
        if demand_only:
            connections = demand_connections(problem, config)
            maximum = max((volume for _, volume in connections), default=1)
            for (a, b), volume in connections:
                xy = project([problem.cities[a]['center'], problem.cities[b]['center']])
                ax.plot(xy[:, 0], xy[:, 1], color='#215c98', alpha=0.6,
                        linewidth=6*volume/maximum, zorder=2)
            for fraction in (0.25, 0.5, 1):
                ax.plot([], [], color='#215c98', linewidth=6*fraction,
                        label=f'{maximum*fraction:,.0f} trips/h')
            ax.legend(title='Two-way demand', loc='lower left', fontsize=8)
        elif result.get('feasible') or result.get('exploratory'):
            ratios = segment_satisfaction(problem, result) if hasattr(problem, 'demand') else {}
            cmap, norm = plt.get_cmap('viridis'), Normalize(0, 1)
            ratio_axis = figure.add_axes([0.85, 0.08, 0.025, 0.13])
            figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ratio_axis,
                            label='Throughput / demand')
            for line in result['lines']:
                frequency = line['frequency_per_hour_per_direction']
                edges = line.get('edges', [])
                geometries = [problem.edges[edge]['geometry'] for edge in edges] or [line['geometry']]
                for segment, geometry in enumerate(geometries):
                    xy = project(geometry)
                    ax.plot(xy[:, 0], xy[:, 1], color=cmap(norm(ratios.get((line['id'], segment), 0))),
                            linewidth=1.7+frequency*0.6, zorder=4)
            for station in result['stations']:
                xy = project([station['coordinate']])[0]
                ax.add_patch(Circle(xy, station['radius_km'], fill=False, color='#217c55',
                                    linewidth=1, linestyle='--', alpha=0.6, zorder=3))
                ax.plot(*xy, marker='o', markerfacecolor='white', markeredgecolor='#124625',
                        markeredgewidth=2, markersize=7, zorder=5)
    xmin, xmax, ymin, ymax = viewport
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)
    ax.set_xlabel('East of regional center (km)')
    ax.set_ylabel('North of regional center (km)')
    if demand_only:
        result = dict(phase='fixed demand', elapsed_seconds=0, iterations=0, snapshot=0)
    phase_label = ('Exploratory — coverage targets may be unmet' if result.get('exploratory')
                   else result['phase'].replace('_', ' ').capitalize())
    if not result.get('exploratory') and result.get('solver', {}).get('stage'):
        phase_label += ' / ' + result['solver']['stage']
    subtitle = (f"{phase_label} · {result['elapsed_seconds']:.1f}s · "
                f"progress step {result['iterations']:,} · update {result['snapshot']}")
    if result.get('feasible') or result.get('exploratory'):
        metrics = result['metrics']
        subtitle += (f"\n{metrics['served_trips_per_hour']:,.0f} trips/h served "
                     f"({metrics['served_fraction']:.1%}) · {metrics['total_track_km']:.0f} km track · "
                     f"{config.accounting_years}-year profit {metrics['horizon_profit']/1e6:,.1f}M")
        if metrics.get('needs_satisfaction_fraction') is not None:
            subtitle += f"\nRail-eligible passenger needs satisfied: {metrics['needs_satisfaction_fraction']:.1%}"
        if 'lifecycle' in metrics:
            appraisal = metrics['lifecycle']
            subtitle += f" · {appraisal['years']}-year NPV {appraisal['npv']/1e6:,.1f}M"
    elif problem is not None:
        subtitle += ('\nSearch ended without a validated network — review the service and budget requirements'
                     if result['phase'] in ('search_exhausted', 'no_feasible_solution', 'infeasible') else
                     '\nConstructing and evaluating networks — markers show candidate markets')
    else:
        subtitle += '\nLoading population data — animation begins when the map is ready'
        ax.text(0.5, 0.5, 'Preparing map…', transform=ax.transAxes, ha='center', va='center', fontsize=16)
    if demand_only:
        subtitle = (f'{len(connections)} strongest city pairs · top {config.demand_map_top_percent:g}% · '
                    f'minimum {config.demand_map_min_trips_per_hour:g} trips/h')
    ax.set_title(('Candidate city demand' if demand_only else 'Rail network optimization') + '\n' + subtitle, fontsize=11)
    footer = ('Colored lines: all-stop services (frequency per direction) · White dots: stations · Dashed green: catchments\n'
              'Scenario estimates; coarse population. Missing terrain/land data permits unsuitable alignments.'
              if not config.terrain_file or not config.land_file else
              'Scenario estimates; H3 terrain routing is coarse and is not an engineering alignment.')
    footer = ('Width: total two-way demand (trips/hour). Demand is fixed before network selection.' if demand_only else
              'Color: carried passengers / full OD demand using each segment (including transfers).\n'
              'Width: trains/hour/direction · White dots: stations · Dashed green: catchments')
    figure.supxlabel(footer, fontsize=8)
    try:
        with matplotlib.rc_context({'savefig.bbox': None}):
            figure.savefig(path, dpi=130)
    finally:
        plt.close(figure)


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


class LiveState:
    def __init__(self):
        self.lock = threading.Lock()
        self.problem = self.model = self.vector = None
        self.preview = None
        self.phase = 'preparing'
        self.message = 'Loading population and city data'
        self.solver = {}
        self.iterations = 0
        self.iteration_unit = 'solver_progress_callback'

    def advance_iteration(self):
        """Count a native solver progress callback, including MIP root search.

        HiGHS reports -1 for live LP iteration counts in MIP callbacks, so using
        that field as a live counter would only produce a final progress frame.
        """
        with self.lock:
            self.iterations += 1
            return self.iterations

    def update(self, **values):
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def snapshot(self):
        with self.lock:
            solver = dict(self.solver, iterations=self.iterations, iteration_unit=self.iteration_unit)
            return self.problem, self.model, self.vector, self.phase, self.message, solver

    def report_snapshot(self):
        """Atomically pair the live exploration view with its saved incumbent."""
        with self.lock:
            solver = dict(self.solver, iterations=self.iterations, iteration_unit=self.iteration_unit)
            incumbent = self.problem, self.model, self.vector
            exploratory = self.preview is not None and (self.phase == 'solving' or self.vector is None)
            if exploratory:
                problem, model, vector, deficit = self.preview
            else:
                problem, model, vector = incumbent
                deficit = 0
            return ((problem, model, vector, self.phase, self.message, solver),
                    exploratory, deficit, incumbent)


class ProgressGif:
    """Append full frames using Pillow's documented GIF block encoder.

    Copy compressed bytes from the existing GIF, replace its trailer with the
    new frame and trailer, and atomically replace the file. Only the new image
    and small copy buffers are held in memory, not all historical frames. Local
    palettes preserve each frame's colors. No individual frame files are kept.
    """
    def __init__(self, path, duration_ms=500):
        if type(duration_ms) is not int or not 10 <= duration_ms <= 655350 or duration_ms % 10:
            raise ValueError('GIF frame duration must be a multiple of 10 ms between 10 and 655350')
        self.path, self.duration_ms = Path(path), duration_ms
        self.frames, self.size = 0, None
        if self.path.exists():
            with Image.open(self.path) as previous:
                if previous.format != 'GIF':
                    raise ValueError('progress.gif must be a valid GIF')
                self.frames, self.size = previous.n_frames, previous.size

    def append(self, png_path):
        with Image.open(png_path) as source:
            rgb = source.convert('RGB')
        if self.size and rgb.size != self.size:
            rgb = rgb.resize(self.size, Image.Resampling.LANCZOS)
        frame = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        temporary = self.path.with_name(self.path.name+'.tmp')
        try:
            with temporary.open('wb') as stream:
                if self.path.exists():
                    with self.path.open('rb') as previous:
                        previous.seek(-1, 2)
                        if previous.read(1) != b';':
                            raise ValueError('Cannot append to an incomplete progress GIF')
                        previous.seek(0)
                        shutil.copyfileobj(previous, stream, length=1024*1024)
                    stream.seek(-1, 2)  # Overwrite the previous GIF trailer.
                else:
                    header, _ = GifImagePlugin.getheader(frame, info=dict(loop=0, optimize=False))
                    for block in header:
                        stream.write(block)
                for block in GifImagePlugin.getdata(frame, duration=self.duration_ms, disposal=2,
                                                   include_color_table=True):
                    stream.write(block)
                stream.write(b';')
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
            frame.close()
            rgb.close()
        self.frames += 1
        self.size = self.size or frame.size


DASHBOARD = '''<!doctype html><html><head><meta charset="utf-8"><title>Rail optimizer</title>
<style>body{font:16px system-ui;margin:24px;background:#f2f5f8;color:#203047}img{max-width:100%;background:white}
pre{white-space:pre-wrap;background:white;padding:16px}a{color:#1265bb}</style></head><body>
<h1>Rail optimizer — live search</h1><p id="status">Waiting for first update…</p>
<p><a href="latest.json">Current network</a> · <a href="best_feasible.json">Best feasible network</a> · <a href="exploration.png">Latest exploration map</a> · <a href="latest.geojson">GeoJSON</a> ·
<a href="latest_stations.csv">Stations CSV</a> · <a href="latest_lines.csv">Lines CSV</a> ·
<a href="demand.png">Candidate demand map</a> · <a href="progress.gif">Progress animation</a></p>
<img id="map" src="latest.png" alt="Current rail network"><pre id="metrics"></pre>
<script>window.updateRail=function(d){document.getElementById('status').textContent=
(d.exploratory?'Exploratory | ':'')+d.phase+(d.solver.stage?' / '+d.solver.stage:'')+' | progress step '+d.iterations+' | '+d.elapsed_seconds.toFixed(1)+'s | update '+d.snapshot+' | '+d.message;
document.getElementById('metrics').textContent=JSON.stringify(d.metrics||{feasible:false,warnings:d.warnings},null,2);
if(window.railUpdateVersion!==d.utc_time){document.getElementById('map').src='latest.png?t='+encodeURIComponent(d.utc_time);
window.railUpdateVersion=d.utc_time;}};
function refresh(){let s=document.createElement('script');s.src='latest.js?t='+Date.now();
s.onload=s.onerror=()=>s.remove();document.body.appendChild(s);}refresh();setInterval(refresh,2000);
</script></body></html>'''


class Reporter:
    def __init__(self, state, config, output, every=5, gif_frame_ms=500):
        if type(every) is not int or every <= 0:
            raise ValueError('snapshot cadence must be a positive number of iterations')
        self.state, self.config, self.output, self.every = state, config, Path(output), every
        self.output.mkdir(parents=True, exist_ok=True)
        latest = self.output/'latest.json'
        self.counter = json.loads(latest.read_text())['snapshot'] if latest.exists() else 0
        self.animation = ProgressGif(self.output/'progress.gif', gif_frame_ms)
        self.viewport = None
        self.demand_signature = None
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.requested = threading.Event()
        self.render_lock = threading.Lock()
        self.cadence_lock = threading.Lock()
        self.last_requested_bucket = 0
        self.growth_frames = []
        self.growth_line_counts = set()
        self.failure = None
        self.thread = threading.Thread(target=self._loop, name='rail-snapshots', daemon=True)
        atomic_text(self.output/'index.html', DASHBOARD)

    def start(self):
        # Population preparation runs before the first image. On resume, keep
        # the previous published image until the new complete map is ready.
        if self.state.snapshot()[0] is not None:
            self.write()
        self.thread.start()

    def notify_iterations(self):
        """Nonblocking callback hook; coalesce requests if rendering falls behind."""
        iterations = self.state.snapshot()[-1]['iterations']
        bucket = iterations // self.every
        with self.cadence_lock:
            if bucket > self.last_requested_bucket:
                self.last_requested_bucket = bucket
                self.requested.set()

    def capture_growth(self):
        """Preserve the first one-, two- and three-service frames despite coalescing."""
        snapshot = self.state.report_snapshot()
        view, exploratory, deficit, incumbent = snapshot
        model = view[1]
        count = len(model.lines) if exploratory and model is not None else 0
        with self.cadence_lock:
            if 1 <= count <= 3 and count not in self.growth_line_counts:
                self.growth_line_counts.add(count)
                self.growth_frames.append(snapshot)
                self.requested.set()

    def capture_exploration(self):
        """Preserve the last full exploratory view before selecting a final incumbent."""
        snapshot = self.state.report_snapshot()
        if snapshot[1]:
            with self.cadence_lock:
                self.growth_frames.append(snapshot)
                self.requested.set()

    def _loop(self):
        while True:
            self.requested.wait()
            self.requested.clear()
            if self.stop.is_set():
                return
            try:
                self.write()
                with self.cadence_lock:
                    if self.growth_frames:
                        self.requested.set()
            except Exception as error:
                self.failure = error
                return

    def finish(self):
        self.stop.set()
        self.requested.set()
        self.thread.join()
        if self.failure:
            raise RuntimeError(f'Snapshot writer failed: {self.failure}') from self.failure
        while self.growth_frames:
            self.write()
        self.write()

    def write(self):
        with self.render_lock:
            self._write()

    def _write(self):
        with self.cadence_lock:
            snapshot = self.growth_frames.pop(0) if self.growth_frames else None
        view, exploratory, deficit, incumbent = snapshot or self.state.report_snapshot()
        problem, model, vector, phase, message, solver = view
        result = model.describe(vector) if model is not None and vector is not None else dict(feasible=False)
        best_problem, best_model, best_vector = incumbent
        best_result = (best_model.describe(best_vector) if best_model is not None and best_vector is not None else None)
        if exploratory:
            result.update(feasible=False, exploratory=True, coverage_deficit=deficit,
                          final_requirements_met=False)
        result['exploratory'] = exploratory
        result['service_lines'] = len(result.get('lines', []))
        result['best_feasible_available'] = bool(best_result and best_result.get('feasible'))
        self.counter += 1
        result.update(snapshot=self.counter, iterations=solver['iterations'], iteration_unit=solver['iteration_unit'], phase=phase, message=message,
                      elapsed_seconds=time.monotonic()-self.started,
                      utc_time=datetime.now(timezone.utc).isoformat(), solver=solver,
                      assumptions=asdict(self.config),
                      warnings=problem.warnings if problem else ['Preparing candidate markets and tracks.'])
        result = clean(result)
        if model is not None:
            result['model_summary'] = dict(variables=len(model.names), constraints=len(model.row_lower),
                                           candidate_stations=len(problem.stations), candidate_tracks=len(problem.edges),
                                           candidate_lines=len(model.lines), potential_trips_per_hour=model.total_demand,
                                           capture_upper_bound_trips_per_hour=model.capture_upper_bound,
                                           rail_eligible_needs_per_hour=model.needs_upper_bound,
                                           minimum_required_trips_per_hour=self.config.minimum_served_fraction*model.total_demand)
        atomic_text(self.output/'latest.geojson', json.dumps(clean(network_geojson(result)), allow_nan=False))
        station_rows = [dict(s, longitude=s['coordinate'][0], latitude=s['coordinate'][1]) for s in result.get('stations', [])]
        station_temp = self.output/'latest_stations.csv.tmp'
        write_csv(station_temp, ['id', 'name', 'longitude', 'latitude', 'radius_km',
                                         'covered_population', 'uptake', 'center_offset_km',
                                         'boardings_per_hour_including_transfers', 'departures_per_hour'], station_rows)
        line_rows = []
        for line in result.get('lines', []):
            row = dict(line, forward_load=line['directional_loads'][0], reverse_load=line['directional_loads'][1],
                       stops=' → '.join(line['stop_names']))
            for key, location in [('from_city', line['a']), ('to_city', line['b'])]:
                row[key] = problem.cities[problem.locations[location]['city']]['name']
            line_rows.append(row)
        line_temp = self.output/'latest_lines.csv.tmp'
        write_csv(line_temp, ['id', 'from_city', 'to_city', 'stops', 'km', 'frequency_per_hour_per_direction',
                                     'capacity_per_direction_per_hour', 'forward_load', 'reverse_load',
                                     'fleet', 'capital_cost', 'annual_operating_cost'], line_rows)
        station_temp.replace(self.output/'latest_stations.csv')
        line_temp.replace(self.output/'latest_lines.csv')
        if problem is not None:
            map_temp = self.output/'latest.tmp.png'
            if self.viewport is None:
                self.viewport = map_viewport(self.config, problem)
            if getattr(problem, 'demand', []):
                signature = tuple((od['origin'], od['destination'], od['volume']) for od in problem.demand)
                if signature != self.demand_signature:
                    demand_temp = self.output/'demand.tmp.png'
                    draw_demand_map(demand_temp, self.config, problem, viewport=self.viewport)
                    demand_temp.replace(self.output/'demand.png')
                    self.demand_signature = signature
            draw_map(map_temp, self.config, problem, result, viewport=self.viewport)
            map_temp.replace(self.output/'latest.png')
            self.animation.append(self.output/'latest.png')
        result['animation_frames'] = self.animation.frames
        atomic_text(self.output/'latest.json', json.dumps(result, indent=2, allow_nan=False))
        if exploratory:
            atomic_text(self.output/'exploration.json', json.dumps(result, indent=2, allow_nan=False))
            shutil.copyfile(self.output/'latest.png', self.output/'exploration.png')
        atomic_text(self.output/'cost_breakdown.log', cost_breakdown_log(result))
        atomic_text(self.output/'latest.js', 'window.updateRail('+json.dumps(result, allow_nan=False)+');')
        if best_result and best_result.get('feasible'):
            atomic_text(self.output/'best_feasible.json', json.dumps(clean(best_result), indent=2, allow_nan=False))
            temporary = self.output/'checkpoint.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, vector=best_vector, fingerprint=best_model.fingerprint(),
                                    **({'annealing': best_model.annealing_checkpoint}
                                       if hasattr(best_model, 'annealing_checkpoint') else {}))
            temporary.replace(self.output/'checkpoint.npz')
        with (self.output/'progress.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps({k: result[k] for k in ('snapshot', 'iterations', 'animation_frames', 'phase', 'elapsed_seconds', 'feasible', 'exploratory', 'service_lines', 'solver')})+'\n')
        phase_display = phase + (' / '+solver['stage'] if solver.get('stage') else '')
        if 'coverage_fraction' in solver:
            label = 'passenger needs satisfied' if solver['coverage_metric'] == 'needs' else solver['coverage_metric']+' coverage'
            phase_display += f" ({solver['coverage_fraction']:.1%}/{solver['coverage_target']:.0%} {label})"
        if 'component_count' in solver:
            phase_display += f"; {solver['component_count']} components, {solver['largest_component_population_share']:.0%} population in largest"
        print(f"[{result['elapsed_seconds']:7.1f}s] update {self.counter}, progress step {result['iterations']:,}: {phase_display}; "
              + (f"{result['metrics']['served_trips_per_hour']:,.1f} trips/h, "
                 f"profit {result['metrics']['horizon_profit']/1e6:,.2f}M" if result.get('metrics') else ('search ended without a validated network' if phase in
                     ('search_exhausted', 'no_feasible_solution', 'infeasible') else
                     'constructing and evaluating networks')),
              flush=True)
