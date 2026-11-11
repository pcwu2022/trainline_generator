"""Iteration-triggered live map, charts, accounting and bounded-memory GIF."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .alignment import TRACKS
from .evaluator import Evaluator
from .report import _plotting, geojson, plot_map
from .search import SearchUpdate


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


def draw_progress(path, history, unit):
    plt = _plotting()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    panels = [
        ('Network size', 'Count', [('active_stations', 'Stations', 1), ('active_edges', 'Corridors', 1)]),
        ('Passengers', 'Million trips/year', [('passengers', 'Carried trips', 1e6)]),
        ('Population and connectivity', 'Percent', [('population_percent', 'Capturable population served', 1),
                                                   ('component_percent', 'Active nodes in largest component', 1)]),
        ('Infrastructure and fleet', 'km / trainsets', [('track_km', 'Track km', 1), ('fleet', 'Fleet', 1)]),
        ('Upfront capital', 'EUR million', [('upfront_eur', 'Upfront capital', 1e6)]),
        ('Annual finances', 'EUR million/year', [('annual_cost_eur', 'Total cost', 1e6),
            ('revenue_eur', 'Revenue', 1e6), ('net_eur', 'Net', 1e6)]),
    ]
    try:
        iterations = [row['iteration'] for row in history]
        for ax, (title, ylabel, series) in zip(axes.flat, panels):
            for key, label, scale in series:
                ax.plot(iterations, [row['metrics'][key] / scale for row in history], label=label)
            ax.set(title=title, xlabel=unit, ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Evolution of the current railway network')
        fig.savefig(path, dpi=110)
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
        f'Annual operating cost: {m["operating_cost_eur"]:18,.2f}',
        f'Annual total cost:     {m["annual_cost_eur"]:18,.2f}',
        f'Annual revenue:        {m["revenue_eur"]:18,.2f}',
        f'Annual net:            {m["net_eur"]:18,.2f}', '',
        'Per-corridor net includes allocated shared capital, station and fleet costs.']
    for i in np.flatnonzero(result.genome):
        edge = graph.edges[i]
        track = TRACKS[result.genome[i] - 1]
        lines.append(f'{graph.cities[edge.a].name} — {graph.cities[edge.b].name} [{track}]: '
            f'{result.frequency[i]:g} trains/h/direction; {result.train_units[i]} units/train; '
            f'{result.directional_load[i]:,.0f} annual trips/direction; '
            f'allocated annual net EUR {result.edge_net[i]:,.2f}')
    return '\n'.join(lines) + '\n'


class LiveReporter:
    def __init__(self, graph, config, output, calibration=None):
        # Reuse the original optimizer's streaming GIF encoder, which appends
        # compressed frames atomically without retaining historical image buffers.
        from train_optimizer.outputs import ProgressGif
        self.graph, self.config, self.output = graph, config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluator = Evaluator(graph, config)
        from .maps import MapBackground
        self.background = MapBackground(graph, config)
        self.every = config['reporting']['snapshot_every']
        self.animation = ProgressGif(self.output / 'progress.gif', config['reporting']['gif_frame_ms'])
        self.history = []
        self.snapshot = 0
        self.last_update = None
        self.calibration = calibration or dict(status='not_calibrated', reason='No user-supplied observed backtest table')
        self.total_population = sum(city.population * city.kappa for city in graph.cities)
        self.unit = 'Network iteration' if config['search']['algorithm'] == 'incremental' else 'Population generation'
        atomic_json(self.output / 'run_config.json', config)
        atomic_json(self.output / 'backtest.json', self.calibration)
        atomic_json(self.output / 'alignment_manifest.json', graph.metadata)
        self.manifest = dict(algorithm=config['search']['algorithm'], iteration_unit=self.unit,
            snapshot_every=self.every, gif_frame_ms=config['reporting']['gif_frame_ms'],
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
            genome=update.result.genome.tolist(), changes=[list(c) for c in update.changes],
            front_size=update.front_size, evaluations=update.evaluations,
            seconds=update.elapsed_seconds, metrics=metrics)
        self.history.append(row)
        with (self.output / 'history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        # Initial frame contains the real region and city nodes, never a blank canvas.
        if update.iteration == 0 or update.iteration % self.every == 0:
            self.render(update)

    def render(self, update):
        result = self.evaluator.evaluate(update.result.genome, details=True)
        phase = update.phase.replace('_', ' ')
        title = (f'{self.unit} {update.iteration}: {phase} · '
            f'{result.metrics["passengers"]/1e6:.2f}M trips/year · '
            f'net EUR {result.metrics["net_eur"]/1e6:.1f}M/year')
        map_temp = self.output / '.latest.tmp.png'
        chart_temp = self.output / '.progress.tmp.png'
        try:
            plot_map(self.graph, result, map_temp, title, self.config, background=self.background)
            map_temp.replace(self.output / 'latest.png')
            from .analysis import write_visual_analysis
            write_visual_analysis(self.graph, self.config, result, self.output,
                                  self.background, title, evaluator=self.evaluator)
            draw_progress(chart_temp, self.history, self.unit)
            chart_temp.replace(self.output / 'progress.png')
        finally:
            map_temp.unlink(missing_ok=True)
            chart_temp.unlink(missing_ok=True)
        self.animation.append(self.output / 'latest.png')
        self.snapshot += 1
        data = dict(iteration=update.iteration, iterations=update.iteration, iteration_unit=self.unit,
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
        # A current-state artifact, not a claim of resumable solver/RNG state.
        checkpoint = self.output / '.checkpoint.tmp.npz'
        try:
            np.savez_compressed(checkpoint, genome=result.genome, objectives=result.objectives,
                iteration=update.iteration, graph_fingerprint=self.graph.metadata.get('fingerprint', ''),
                config_json=json.dumps(self.config, sort_keys=True))
            checkpoint.replace(self.output / 'checkpoint.npz')
        finally:
            checkpoint.unlink(missing_ok=True)
        self.write_summary(data)

    def write_summary(self, data):
        m = data['metrics']
        lines = ['# Railway network evolution', '',
            f'Algorithm: **{data["algorithm"]}** · {self.unit.lower()} **{data["iteration"]}** · **{data["phase"]}**.', '',
            f'Calibration: **{self.calibration["status"]}**. Parameters and bundled city populations remain unverified unless separately sourced.', '',
            '![Current network](latest.png)', '',
            f'{m["active_stations"]} stations, {m["active_edges"]} corridors, {m["track_km"]:,.1f} km; '
            f'{m["passengers"]/1e6:.2f} million carried trips/year; net EUR {m["net_eur"]/1e6:.1f} million/year.', '',
            f'Need satisfaction: **{m["need_satisfaction_ratio"]:.2%}** '
            '(carried trips / fixed two-way annual capturable gravity demand across all candidate OD pairs).', '',
            '[Animated evolution](progress.gif) · [Current GeoJSON](latest.geojson) · [Current metrics](latest.json) · [Cost breakdown](cost_breakdown.log)', '',
            '![Revenue / annual cost map](latest_revenue_cost.png)', '',
            '![Station and corridor rankings](analysis.png)', '',
            '[Full station and corridor metrics](analysis.json)', '',
            '![Progress over iterations](progress.png)', '',
            'The live network is the accepted state carried between iterations. '
            'Rejected proposals leave it unchanged. Growth and integration aim to serve all candidate cities in one component '
            'before cost consolidation with a fixed service baseline. '
            f'{m["unserved_stations"]} candidate cities remain unserved; {m["active_components"]} active components.', '',
            '[Pareto alternatives and backtests](pareto/report.md) are comparison designs, not animation frames.']
        if data['algorithm'] != 'incremental':
            lines[-3] = 'Population mode displays the current Pareto compromise per generation; it can switch between lineages.'
        atomic_text(self.output / 'report.md', '\n'.join(lines) + '\n')

    def finish(self, run):
        if self.last_update is None:
            return
        result = self.evaluator.evaluate(run.current_genome if run.current_genome is not None
                                         else self.last_update.result.genome)
        terminal = SearchUpdate(self.last_update.iteration, run.status, result,
            len(run.genomes), run.evaluations, run.elapsed_seconds,
            self.last_update.accepted, self.last_update.changes, self.last_update.parent_genome)
        # Always include the final state, even when it falls between snapshot steps.
        self.render(terminal)
        self.manifest.update(status=run.status, iterations=terminal.iteration,
                             animation_frames=self.animation.frames)
        atomic_json(self.output / 'manifest.json', self.manifest)
        atomic_json(self.output / 'search_history.json', run.history)
