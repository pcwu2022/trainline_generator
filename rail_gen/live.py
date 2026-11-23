"""Iteration-triggered live map, charts, accounting and bounded-memory MP4."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .alignment import TRACKS
from .track_layout import track_type, MODEL_VERSION
from .evaluator import Evaluator
from .report import _plotting, geojson, plot_map
from .search import SearchUpdate


def snapshot_stats(result, config, city_count, unit, iteration, phase):
    """Display annual network metrics; financial evaluation remains in euros."""
    m, reporting = result.metrics, config['reporting']
    profit = m['net_eur'] * reporting['eur_to_local'] / 1e6
    return (f'{unit}: {iteration}\n'
            f'{phase.replace("_", " ").capitalize()}\n'
            f'Ridership: {m["passengers"]/1e6:,.2f}M trips/year\n'
            f'Need satisfaction: {m["need_satisfaction_ratio"]:.1%}\n'
            f'Cities served: {m["active_stations"]}/{city_count}\n'
            f'Net profit: {profit:,.1f}M {reporting["currency_symbol"]}/year')


def atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        temporary.write_text(content, encoding='utf8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def draw_progress(path, history, unit, region_name=''):
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from .social_charts import canvas, card, footer, save, INK, MUTED, BLUE, GREEN, RED, GOLD
    plt = _plotting()
    latest = history[-1]['metrics'] if history else {}
    panels = [
        ('How big is the network?', 'Stations and city-to-city infrastructure links', 'Count',
         [('active_stations', 'Stations', 1, BLUE), ('active_edges', 'Links', 1, GOLD)]),
        ('How many trips does rail carry?', 'Passenger journeys in both directions each year', 'Million trips / year',
         [('passengers', 'Passenger trips', 1e6, BLUE)]),
        ('Who can use the network?', 'Population in catchments; stations in main component', 'Percent',
         [('population_percent', 'Population served', 1, BLUE), ('component_percent', 'Connected stations', 1, GREEN)]),
        ('How much track and rolling stock?', 'Track length and train units use separate scales', 'Track km',
         [('track_km', 'Track km · left', 1, BLUE), ('fleet', 'Train units · right', 1, GOLD)]),
        ('How much investment is needed?', 'One-time infrastructure, stations and fleet capital', '€ billion upfront',
         [('upfront_eur', 'Initial investment', 1e9, GOLD)]),
        ('Does annual revenue cover costs?', 'Net = revenue − cost; below zero means a loss', '€ billion / year',
         [('annual_cost_eur', 'Cost', 1e9, RED), ('revenue_eur', 'Revenue', 1e9, BLUE), ('net_eur', 'Net', 1e9, GREEN)]),
    ]
    fig = canvas(plt, f'{region_name + " · " if region_name else ""}Building a rail network',
        'Read left to right: each step is an optimization update, not a calendar year.',
        [(f"{latest.get('passengers', 0)/1e6:,.1f}M", 'latest passenger trips / year'),
         (str(latest.get('active_stations', 0)), 'latest active stations'),
         (f"€{latest.get('net_eur', 0)/1e9:+,.2f}B", 'latest annual net result')])
    try:
        iterations = [row['iteration'] for row in history]
        for i, (heading, description, ylabel, series) in enumerate(panels):
            x, y, w, h = .045 + (i % 2) * .475, .562 - (i // 2) * .232, .435, .217
            card(fig, x, y, w, h, heading, description)
            ax = fig.add_axes([x + .057, y + .038, w - (.115 if i == 3 else .075), .105])
            axes, handles = [ax], []
            for j, (key, label, scale, color) in enumerate(series):
                target = ax
                if i == 3 and j == 1:
                    target = ax.twinx()
                    axes.append(target)
                    target.set_ylabel('Train units', color=GOLD, fontsize=9)
                    target.tick_params(axis='y', labelcolor=GOLD)
                values = [row['metrics'][key] / scale for row in history]
                line, = target.plot(iterations, values, color=color, linewidth=2.3,
                                    linestyle='--' if j == 1 else '-', label=label)
                if values:
                    target.plot(iterations[-1], values[-1], 'o', color=color, markersize=4)
                handles.append(line)
            for target in axes:
                target.spines[['top', 'right']].set_visible(False)
                target.spines['left'].set_color('#c6d1d8')
                target.spines['bottom'].set_color('#c6d1d8')
                target.tick_params(labelsize=9, colors=MUTED)
                target.yaxis.set_major_locator(MaxNLocator(nbins=4))
                target.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if abs(v) >= 100 else f'{v:g}'))
            ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
            ax.set_xlabel('Design step' if unit == 'Network iteration' else unit, fontsize=9, color=MUTED, labelpad=3)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.grid(axis='y', color='#e2e9ee', linewidth=.8)
            if i == 2:
                ax.set_ylim(0, 105)
            if i == 5:
                ax.axhline(0, color=MUTED, linewidth=.8, linestyle=':')
            ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-.03, 1.01),
                      ncol=len(series), fontsize=9, frameon=False, handlelength=1.4,
                      columnspacing=1, borderaxespad=0)
        footer(fig, 'Each point shows one modelled design. Annual figures describe a full year of operation at that design.\n'
               'Population served = people in station catchments. Connected stations = share in the largest connected group.')
        save(fig, path)
    finally:
        plt.close(fig)


def cost_log(graph, result, iteration, phase):
    m = result.metrics
    lines = [f'Railway network — iteration {iteration} — {phase}',
        'All money is constant-price EUR. Trips include both directions.', '',
        f'Active stations: {m["active_stations"]}; corridors: {m["active_edges"]}; track: {m["track_km"]:,.1f} km',
        f'Annual carried trips: {m["passengers"]:,.0f}; captured population: {m["population_served"]:,.0f}',
        f'Fleet CR/HSR: {m["fleet_CR"]}/{m["fleet_HSR"]}', '',
        f'Track capital:         {m["track_capital_eur"]:18,.2f}',
        f'Station capital:       {m["station_capital_eur"]:18,.2f}',
        f'Fleet capital:         {m["fleet_capital_eur"]:18,.2f}',
        f'Total upfront capital: {m["upfront_eur"]:18,.2f}', '',
        f'Annual capital:        {m["annual_capital_eur"]:18,.2f}',
        f'Annual train operation:{m["train_opex_eur"]:18,.2f}',
        f'Fixed track maintenance:{m["track_fixed_maintenance_eur"]:17,.2f}',
        f'Variable maintenance: {m["track_variable_maintenance_eur"]:18,.2f}',
        f'Annual operating cost: {m["operating_cost_eur"]:18,.2f}',
        f'Annual total cost:     {m["annual_cost_eur"]:18,.2f}',
        f'Annual revenue:        {m["revenue_eur"]:18,.2f}',
        f'Annual net:            {m["net_eur"]:18,.2f}', '',
        'Per-corridor net includes allocated shared capital, station and fleet costs.']
    for i in np.flatnonzero(result.genome):
        edge = graph.edges[i]
        track = track_type(result.genome[i])
        lines.append(f'{graph.cities[edge.a].name} — {graph.cities[edge.b].name} [{track}, {int(result.track_counts[i])} tracks]: '
            f'{result.frequency[i]:g} aggregate trains/h/direction; max {result.train_units[i]} units/train; '
            f'{result.directional_load[i]:,.0f} annual trips/direction; '
            f'allocated annual net EUR {result.edge_net[i]:,.2f}')
    lines += ['', 'Actual train services (infrastructure changes do not imply transfers):']
    for service in result.services:
        route = ' — '.join(graph.cities[node].name for node in service['route'])
        lines.append(f"{service['train_type']}: {route}; {service['frequency']:g} trains/h/direction; {service['formation']} units/train; {service['cycle_hours']:.3f} h round trip incl. turnaround; fleet {service['fleet_units']} units; annual operation EUR {service['operating_cost_eur']:,.2f}")
    return '\n'.join(lines) + '\n'


class LiveReporter:
    def __init__(self, graph, config, output, calibration=None, resume=False, compact=False):
        from rail_gen.map_utils import ProgressVideo
        self.graph, self.config, self.output = graph, config, Path(output)
        self.compact = compact
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluator = Evaluator(graph, config)
        from .maps import MapBackground
        self.background = MapBackground(graph, config)
        from .output_names import migrate_visual_names
        self.visuals = migrate_visual_names(self.output, config)
        self.every = config['reporting']['snapshot_every']
        self.animation = ProgressVideo(self.output / self.visuals['animation'], config['reporting']['snapshot_hold_ms'])
        self.history = []
        if resume and (self.output / 'history.jsonl').exists():
            self.history = [json.loads(line) for line in (self.output / 'history.jsonl').read_text().splitlines() if line.strip()]
        self.snapshot = 0
        self.last_update = None
        self.calibration = calibration or dict(status='not_calibrated', reason='No user-supplied observed backtest table')
        self.total_population = sum(city.population * city.kappa for city in graph.cities)
        self.unit = 'Network iteration' if config['search']['algorithm'] == 'incremental' else 'Population generation'
        if compact:
            return
        atomic_json(self.output / 'run_config.json', config)
        atomic_json(self.output / 'backtest.json', self.calibration)
        atomic_json(self.output / 'alignment_manifest.json', graph.metadata)
        self.manifest = dict(model_version=MODEL_VERSION, algorithm=config['search']['algorithm'], iteration_unit=self.unit,
            snapshot_every=self.every, snapshot_hold_ms=config['reporting']['snapshot_hold_ms'],
            graph=graph.metadata, calibration=self.calibration['status'],
            current_network='latest.geojson', alternatives='pareto/report.md', status='running')
        atomic_json(self.output / 'manifest.json', self.manifest)

    def __call__(self, update):
        self.last_update = update
        metrics = update.result.metrics.copy()
        metrics.update(population_percent=100 * metrics['population_served'] / max(1, self.total_population),
                       component_percent=100 * metrics['component_share'],
                       fleet=metrics['fleet_CR'] + metrics['fleet_HSR'])
        row = dict(iteration=update.iteration, phase=update.phase, accepted=update.accepted,
            parent_iteration=update.iteration - 1 if update.iteration else None,
            parent_genome=update.parent_genome.tolist() if update.parent_genome is not None else None,
            genome=update.result.genome.tolist(), services=update.result.services, track_counts=update.result.track_counts.tolist(), changes=[list(c) for c in update.changes],
            front_size=update.front_size, evaluations=update.evaluations,
            seconds=update.elapsed_seconds, metrics=metrics)
        row.update(update.metadata or {})
        if self.compact:
            for key in ('parent_genome', 'genome', 'services', 'track_counts', 'search_state'):
                row.pop(key, None)
        self.history.append(row)
        with (self.output / 'history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        # Initial frame contains the real region and city nodes, never a blank canvas.
        if update.iteration == 0 or update.iteration % self.every == 0:
            self.render(update)

    def render(self, update):
        # Search updates already contain complete assignments and allocations.
        result = update.result
        title = snapshot_stats(result, self.config, len(self.graph.cities),
                               self.unit, update.iteration, update.phase)
        map_temp = self.output / '.latest.tmp.png'
        chart_temp = self.output / '.progress.tmp.png'
        try:
            plot_map(self.graph, result, map_temp, title, self.config, background=self.background)
            map_temp.replace(self.output / self.visuals['map'])
            from .analysis import write_visual_analysis
            write_visual_analysis(self.graph, self.config, result, self.output,
                                  self.background, title, evaluator=self.evaluator, export_metrics=not self.compact)
            draw_progress(chart_temp, self.history, self.unit, self.config['reporting']['region_name'])
            chart_temp.replace(self.output / self.visuals['progress'])
        finally:
            map_temp.unlink(missing_ok=True)
            chart_temp.unlink(missing_ok=True)
        self.animation.append(self.output / self.visuals['map'])
        self.snapshot += 1
        if self.compact:
            atomic_text(self.output / 'cost_breakdown.log', cost_log(self.graph, result, update.iteration, update.phase))
            return
        data = dict(model_version=MODEL_VERSION, services=result.services, track_counts=result.track_counts.tolist(), iteration=update.iteration, iterations=update.iteration, iteration_unit=self.unit,
            phase=update.phase, algorithm=self.config['search']['algorithm'], snapshot=self.snapshot,
            animation_frames=self.animation.frames, feasible=result.feasible,
            accepted=update.accepted, changes=[list(c) for c in update.changes],
            parent_genome=update.parent_genome.tolist() if update.parent_genome is not None else None,
            genome=result.genome.tolist(), objectives=result.objectives.tolist(),
            constraints=result.constraints.tolist(), metrics=result.metrics,
            front_size=update.front_size, evaluations=update.evaluations,
            elapsed_seconds=update.elapsed_seconds, calibration=self.calibration['status'],
            utc_time=datetime.now(timezone.utc).isoformat())
        atomic_json(self.output / 'latest.json', data)
        atomic_json(self.output / 'latest.geojson', geojson(self.graph, result))
        atomic_text(self.output / 'cost_breakdown.log', cost_log(self.graph, result, update.iteration, update.phase))
        # Evaluated network and optional incremental continuation state.
        checkpoint = self.output / '.checkpoint.tmp.npz'
        try:
            np.savez_compressed(checkpoint, genome=result.genome, objectives=result.objectives,
                iteration=update.iteration, graph_fingerprint=self.graph.metadata.get('fingerprint', ''),
                config_json=json.dumps(self.config, sort_keys=True), model_version=MODEL_VERSION,
                track_counts=result.track_counts, services_json=json.dumps(result.services),
                search_state_json=json.dumps((update.metadata or {}).get('search_state', {})))
            checkpoint.replace(self.output / 'checkpoint.npz')
        finally:
            checkpoint.unlink(missing_ok=True)
        self.write_summary(data)

    def write_summary(self, data):
        m = data['metrics']
        lines = ['# Railway network evolution', '',
            f'Algorithm: **{data["algorithm"]}** · {self.unit.lower()} **{data["iteration"]}** · **{data["phase"]}**.', '',
            f'Calibration: **{self.calibration["status"]}**. Parameters and bundled city populations remain unverified unless separately sourced.', '',
            f'![Current network]({self.visuals["map"]})', '',
            f'{m["active_stations"]} stations, {m["active_edges"]} corridors, {m["track_km"]:,.1f} km; '
            f'{m["passengers"]/1e6:.2f} million carried trips/year; net EUR {m["net_eur"]/1e6:.1f} million/year.', '',
            f'Demand fulfillment: **{m["need_satisfaction_ratio"]:.2%}** ' +
            ('(carried / eligible annual trips: candidate potential for connected pairs, reference rail trips for unreachable pairs).'
             if self.config['demand']['model'] == 'reference_elasticity' else
             '(carried trips / fixed two-way annual capturable gravity demand across all candidate OD pairs).'), '',
            f'[Animated evolution]({self.visuals["animation"]}) · [Current GeoJSON](latest.geojson) · [Current metrics](latest.json) · [Cost breakdown](cost_breakdown.log)', '',
            f'![Cities and demand rankings]({self.visuals["analysis1"]})', '',
            f'![Financial rankings]({self.visuals["analysis2"]})', '',
            '[Full station and corridor metrics](analysis.json)', '',
            f'![Progress over iterations]({self.visuals["progress"]})', '',
            'The live network is the accepted state carried between iterations. '
            'Rejected proposals leave it unchanged. Growth and integration aim to serve all candidate cities in one component '
            'before cost consolidation with a fixed service baseline. Satisfaction controls the objective weights. '
            f'{m["unserved_stations"]} candidate cities remain unserved; {m["active_components"]} active components.', '',
            '[Pareto alternatives and backtests](pareto/report.md) are comparison designs, not animation frames.']
        if data['algorithm'] != 'incremental':
            lines[-3] = 'Population mode displays the current Pareto compromise per generation; it can switch between lineages.'
        if self.config['demand']['model'] == 'reference_elasticity':
            lines.extend(['', f'Fixed synthetic reference: {m["reference_rail_trips"]:,.0f} annual rail trips. '
                f'Potential demand: {m["potential_rail_trips"]:,.0f}; '
                f'modal shift change: {m["modal_shift_change"]:+,.0f}; '
                f'induced demand change: {m["induced_demand_change"]:+,.0f}; '
                f'capacity spill: {m["lost_trips"]:,.0f}. '
                'Changes are signed, before capacity spill, and apply equally across borders.'])
        lines += ['', '## Train services', '',
            'All infrastructure is newly built and fully charged. Compatible gauge, electrification and signalling are assumed. Infrastructure symbols and train types are separate.', '',
            '| Train type | Route | Trains/h/direction | Units/train | Fleet units |',
            '| --- | --- | ---: | ---: | ---: |']
        for service in data['services']:
            route = ' → '.join(self.graph.cities[node].name.replace('|', '\\|') for node in service['route'])
            lines.append(f"| {service['train_type']} | {route} | {service['frequency']:g} | {service['formation']} | {service['fleet_units']} |")
        atomic_text(self.output / 'report.md', '\n'.join(lines) + '\n')

    def finish(self, run):
        if self.last_update is None:
            return
        result = self.last_update.result
        if run.current_genome is not None and not np.array_equal(run.current_genome, result.genome):
            result = self.evaluator.evaluate(run.current_genome)
        terminal = SearchUpdate(self.last_update.iteration, run.status, result,
            len(run.genomes), run.evaluations, run.elapsed_seconds,
            self.last_update.accepted, self.last_update.changes, self.last_update.parent_genome)
        # Always include the final state, even when it falls between snapshot steps.
        self.render(terminal)
        if self.compact:
            self.animation.state_path.unlink(missing_ok=True)
            atomic_text(self.output / 'run.log',
                        f'Status: {run.status}\nIterations: {terminal.iteration}\n'
                        f'Evaluations: {run.evaluations}\nElapsed seconds: {run.elapsed_seconds:.2f}\n')
            return
        self.manifest.update(status=run.status, iterations=terminal.iteration,
                             animation_frames=self.animation.frames)
        atomic_json(self.output / 'manifest.json', self.manifest)
        atomic_json(self.output / 'search_history.json', run.history)
