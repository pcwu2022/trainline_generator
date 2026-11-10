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
from matplotlib.colors import LogNorm
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


def draw_map(path, config, problem, result, viewport=None):
    project = map_projector(config)
    # Explicit axes prevent the colorbar from stealing space from the map.
    # Disable automatic layout/cropping even when enabled in local matplotlibrc.
    figure = plt.figure(figsize=(10, 10), layout='none')
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
                                    norm=LogNorm(vmin=1, vmax=max(10, max(values))),
                                    edgecolors='#ffffff', linewidths=0.2, alpha=0.65)
        ax.add_collection(collection)
        figure.colorbar(collection, cax=color_axis,
                        label='Population density (people/km², log scale)')
        for city in problem.cities:
            xy = project([city['center']])[0]
            ax.plot(*xy, marker='+', color='#5d6370', markersize=7, zorder=3)
            ax.annotate(city['name'], xy, xytext=(4, 5), textcoords='offset points', fontsize=8, zorder=6)
        if result.get('feasible'):
            palette = ['#1265bb', '#8c39a8', '#e17a16', '#008a83', '#b34e66']
            for line_number, line in enumerate(result['lines']):
                xy = project(line['geometry'])
                frequency = line['frequency_per_hour_per_direction']
                color = palette[line_number % len(palette)]
                ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.7+frequency*0.6, zorder=4)
            for station in result['stations']:
                xy = project([station['coordinate']])[0]
                ax.add_patch(Circle(xy, station['radius_km'], fill=False, color='#217c55',
                                    linewidth=1, linestyle='--', alpha=0.6, zorder=3))
                ax.plot(*xy, marker='o', markerfacecolor='white', markeredgecolor='#124625',
                        markeredgewidth=2, markersize=7, zorder=5)
    xmin, xmax, ymin, ymax = viewport if viewport is not None else map_viewport(config, problem)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)
    ax.set_xlabel('East of regional center (km)')
    ax.set_ylabel('North of regional center (km)')
    phase_label = result['phase'].replace('_', ' ').capitalize()
    subtitle = (f"{phase_label} · {result['elapsed_seconds']:.1f}s · "
                f"progress step {result['iterations']:,} · update {result['snapshot']}")
    if result.get('feasible'):
        metrics = result['metrics']
        subtitle += (f"\n{metrics['served_trips_per_hour']:,.0f} trips/h served "
                     f"({metrics['served_fraction']:.1%}) · {metrics['total_track_km']:.0f} km track · "
                     f"{config.accounting_years}-year profit {metrics['horizon_profit']/1e6:,.1f}M")
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
    ax.set_title('Rail network optimization\n' + subtitle, fontsize=11)
    footer = ('Colored lines: all-stop services (frequency per direction) · White dots: stations · Dashed green: catchments\n'
              'Scenario estimates; coarse population. Missing terrain/land data permits unsuitable alignments.'
              if not config.terrain_file or not config.land_file else
              'Scenario estimates; H3 terrain routing is coarse and is not an engineering alignment.')
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
<h1>Rail optimizer — live incumbent</h1><p id="status">Waiting for first update…</p>
<p><a href="latest.json">Full solution</a> · <a href="latest.geojson">GeoJSON</a> ·
<a href="latest_stations.csv">Stations CSV</a> · <a href="latest_lines.csv">Lines CSV</a> ·
<a href="progress.gif">Progress animation</a></p>
<img id="map" src="latest.png" alt="Current rail network"><pre id="metrics"></pre>
<script>window.updateRail=function(d){document.getElementById('status').textContent=
d.phase+' | progress step '+d.iterations+' | '+d.elapsed_seconds.toFixed(1)+'s | update '+d.snapshot+' | '+d.message;
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
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.requested = threading.Event()
        self.cadence_lock = threading.Lock()
        self.last_requested_bucket = 0
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

    def _loop(self):
        while True:
            self.requested.wait()
            self.requested.clear()
            if self.stop.is_set():
                return
            try:
                self.write()
            except Exception as error:
                self.failure = error
                return

    def finish(self):
        self.stop.set()
        self.requested.set()
        self.thread.join()
        if self.failure:
            raise RuntimeError(f'Snapshot writer failed: {self.failure}') from self.failure
        self.write()

    def write(self):
        problem, model, vector, phase, message, solver = self.state.snapshot()
        result = model.describe(vector) if model is not None and vector is not None else dict(feasible=False)
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
            draw_map(map_temp, self.config, problem, result, viewport=self.viewport)
            map_temp.replace(self.output/'latest.png')
            self.animation.append(self.output/'latest.png')
        result['animation_frames'] = self.animation.frames
        atomic_text(self.output/'latest.json', json.dumps(result, indent=2, allow_nan=False))
        atomic_text(self.output/'cost_breakdown.log', cost_breakdown_log(result))
        atomic_text(self.output/'latest.js', 'window.updateRail('+json.dumps(result, allow_nan=False)+');')
        if vector is not None and result.get('feasible'):
            temporary = self.output/'checkpoint.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, vector=vector, fingerprint=model.fingerprint())
            temporary.replace(self.output/'checkpoint.npz')
        with (self.output/'progress.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps({k: result[k] for k in ('snapshot', 'iterations', 'animation_frames', 'phase', 'elapsed_seconds', 'feasible', 'solver')})+'\n')
        print(f"[{result['elapsed_seconds']:7.1f}s] update {self.counter}, progress step {result['iterations']:,}: {phase}; "
              + (f"{result['metrics']['served_trips_per_hour']:,.1f} trips/h, "
                 f"profit {result['metrics']['horizon_profit']/1e6:,.2f}M" if result.get('feasible') else ('search ended without a validated network' if phase in
                     ('search_exhausted', 'no_feasible_solution', 'infeasible') else
                     'constructing and evaluating networks')),
              flush=True)
