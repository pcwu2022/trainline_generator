"""Comparable annual station/corridor rankings and allocated financial analysis."""
import json
from pathlib import Path

import numpy as np

from .alignment import TRACKS
from .track_layout import track_type, MODEL_VERSION
from .evaluator import Evaluator


def demand_denominator(evaluator, result):
    """Candidate potential for connected pairs, reference demand otherwise."""
    if evaluator.config['demand']['model'] == 'reference_elasticity':
        return np.where(np.isfinite(result.journey_time), result.potential_trips,
                        result.reference_trips)
    return 2 * evaluator.base


def corridor_satisfaction(evaluator, result):
    """Carried / potential assigned demand (legacy mode: fixed gravity markets)."""
    e = evaluator
    incidence = result.edge_incidence
    eligible = np.asarray(incidence @ np.where(np.isfinite(result.journey_time), demand_denominator(e, result), 0)).ravel()
    carried = 2 * result.directional_load
    ratios = np.divide(carried, eligible, out=np.zeros(e.m), where=eligible > 0)
    return np.clip(ratios, 0, 1), eligible


def network_analysis(graph, config, result, evaluator=None, *, reevaluate=True):
    """Analyze allocations; reuse current evaluations within the reporting pipeline."""
    e = evaluator or Evaluator(graph, config)
    if reevaluate:
        result = e.evaluate(result.genome, details=True, services=result.service_plan, track_counts=result.track_counts)
    active = np.flatnonzero(result.genome)
    served = set(np.r_[e.a[active], e.b[active]].tolist())
    station_market = np.bincount(np.r_[e.origins, e.destinations],
        weights=np.tile(demand_denominator(e, result), 2), minlength=e.n)
    station_trips = np.bincount(np.r_[e.origins, e.destinations],
        weights=np.tile(result.trips, 2), minlength=e.n)
    ratios, markets = corridor_satisfaction(e, result)
    stations = [dict(id=city.id, name=city.name, served=i in served,
        carried_trips=float(station_trips[i]), eligible_trips=float(station_market[i]),
        satisfaction=float(station_trips[i] / station_market[i]) if station_market[i] > 0 else None)
        for i, city in enumerate(graph.cities)]
    corridors = []
    for i in active:
        edge = graph.edges[i]
        revenue, net = float(result.edge_revenue[i]), float(result.edge_net[i])
        cost = revenue - net
        corridors.append(dict(edge_id=int(i), name=f'{graph.cities[edge.a].name} – {graph.cities[edge.b].name}',
            track=track_type(result.genome[i]), track_count=int(result.track_counts[i]), service_types=sorted({s["train_type"] for s in result.services if int(i) in s["edge_ids"]}), satisfaction=float(ratios[i]) if markets[i] > 0 else None,
            carried_trips=float(2 * result.directional_load[i]), eligible_trips=float(markets[i]),
            revenue_eur=revenue, annual_cost_eur=cost, annual_net_eur=net,
            revenue_cost_ratio=revenue / cost if cost > 0 else None,
            frequency=float(result.frequency[i])))
    od_demand = [dict(origin=graph.cities[a].id, destination=graph.cities[b].id,
        reference_hours=float(e.reference_hours[j]), reference_rail_trips=float(result.reference_trips[j]),
        journey_hours=float(result.journey_time[j]) if np.isfinite(result.journey_time[j]) else None,
        potential_trips=float(result.potential_trips[j]), carried_trips=float(result.trips[j]),
        modal_shift_change=float(result.shift_change[j]), induced_demand_change=float(result.induced_change[j]))
        for j, (a, b) in enumerate(zip(e.origins, e.destinations))
        if config['demand']['model'] == 'reference_elasticity']
    return dict(region_name=config['reporting']['region_name'], model_version=MODEL_VERSION, od_demand=od_demand, services=result.services, track_counts=result.track_counts.tolist(), genome=result.genome.tolist(), metrics=result.metrics, stations=stations, corridors=corridors,
        definitions=dict(station_satisfaction='Carried two-way annual endpoint trips / eligible demand, including unserved destinations; transfers excluded. Elasticity mode uses candidate potential for connected pairs and reference rail trips otherwise; legacy mode uses fixed gravity markets.',
            corridor_satisfaction='Carried two-way annual leg trips / potential OD demand assigned to that corridor (fixed gravity markets in legacy mode); transfers included; same as map.',
            demand_changes='Signed changes before capacity spill; reference + modal shift + induced change = potential trips. All pairs use identical elasticities; reference is synthetic and fixed for the run. Breakdown is only available in reference_elasticity mode.',
            annual_cost='Operations plus annualized capital, maintenance, station and fleet allocations; shared infrastructure counted once.',
            profit='Annual allocated revenue minus annual allocated cost; not upfront investment payback.',
            lines='Infrastructure corridors; separate services list contains actual through-running train routes.'))


def _count(value):
    return f'{value/1e6:,.2f}M' if abs(value) >= 1e6 else f'{value/1e3:,.1f}k' if abs(value) >= 1000 else f'{value:,.0f}'


def analysis_financial_path(path):
    path = Path(path)
    return path.with_name(f'{path.stem}_financial{path.suffix}')


def draw_analysis(path, analysis, title='Current network', limit=6):
    """Square social dashboard; complete rankings remain in analysis.json."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from textwrap import shorten
    from .social_charts import canvas, card, footer, save, INK, MUTED, BLUE, GREEN, RED, GOLD

    m = analysis['metrics']
    stations = [r for r in analysis['stations'] if r['served']]
    corridors = analysis['corridors']
    panels = [
        ('Which cities meet the most demand?', 'Share of eligible trips carried · higher is better', stations,
         'satisfaction', 100, GREEN, lambda r: f"{r['satisfaction']:.1%}"),
        ('Which cities handle the most trips?', 'Annual trips starting or ending at each city', stations,
         'carried_trips', 1e-6, BLUE, lambda r: _count(r['carried_trips'])),
        ('Which links meet the most demand?', 'Share of assigned trips carried · higher is better', corridors,
         'satisfaction', 100, GREEN, lambda r: f"{r['satisfaction']:.1%}"),
        ('Which links cover their costs?', 'Annual revenue ÷ cost · 1× means break-even', corridors,
         'revenue_cost_ratio', 1, GOLD, lambda r: f"{r['revenue_cost_ratio']:.2f}×"),
        ('Which links cost the most?', 'Allocated annual cost · € millions', corridors,
         'annual_cost_eur', 1e-6, GOLD, lambda r: f"€{r['annual_cost_eur']/1e6:,.1f}M"),
        ('Which links earn the most revenue?', 'Annual ticket revenue · € millions', corridors,
         'revenue_eur', 1e-6, BLUE, lambda r: f"€{r['revenue_eur']/1e6:,.1f}M"),
        ('Which links have the best net result?', 'Revenue minus annual cost · negative means loss', corridors,
         'annual_net_eur', 1e-6, None, lambda r: f"€{r['annual_net_eur']/1e6:+,.1f}M"),
        ('Which links lose the most money?', 'Annual allocated loss · € millions',
         [r for r in corridors if r['annual_net_eur'] < 0],
         'annual_net_eur', -1e-6, RED, lambda r: f"€{-r['annual_net_eur']/1e6:,.1f}M"),
    ]
    region = analysis.get('region_name', '')
    for page, page_path in enumerate((Path(path), analysis_financial_path(path))):
        fig = canvas(plt, f'{region + " · " if region else ""}Inside the rail network',
            f'{page + 1} / 2 · Top {limit} in each ranking · ' + ('Cities & demand' if page == 0 else 'Revenue & costs'),
            [(_count(m['passengers']), 'passenger trips / year'),
             (f"{m['need_satisfaction_ratio']:.1%}", 'eligible travel demand carried'),
             (str(m['active_stations']), 'active stations')])
        try:
            for i, (heading, description, rows, key, scale, color, label) in enumerate(panels[page * 4:(page + 1) * 4]):
                x, y, w, h = .045 + (i % 2) * .475, .45 - (i // 2) * .355, .435, .33
                card(fig, x, y, w, h, heading, description)
                ax = fig.add_axes([x + .015, y + .009, w - .03, .258])
                rows = sorted([r for r in rows if r[key] is not None and np.isfinite(r[key])],
                              key=lambda r: (-r[key] * scale, r['name']))[:limit]
                ax.set_axis_off()
                if not rows:
                    ax.text(0, .5, 'No qualifying cities or links yet', color=MUTED, fontsize=11)
                    continue
                values = np.array([r[key] * scale for r in rows])
                # A short diverging bar sits below each label; values stay in a fixed right column.
                low, high = min(0, values.min()), max(0, values.max())
                span = max(high - low, 1)
                ax.set(xlim=(low, high if high > low else low + span), ylim=(0, len(rows)))
                ax.invert_yaxis()
                ax.barh(np.arange(len(rows)) + .80, values, height=.15,
                        color=color or [GREEN if v >= 0 else RED for v in values], alpha=.8)
                for j, row in enumerate(rows):
                    ax.text(0, 1 - (j + .08) / len(rows),
                            shorten(row['name'], width=40, placeholder='…'),
                            transform=ax.transAxes, fontsize=11, color=INK, va='top')
                    ax.text(1, 1 - (j + .08) / len(rows), label(row),
                            transform=ax.transAxes, fontsize=11, color=INK, weight='bold', ha='right', va='top',
                            bbox=dict(facecolor='white', edgecolor='none', pad=1))
                if low < 0:
                    ax.axvline(0, color=MUTED, linewidth=.6)
            footer(fig, 'Trips count both directions; city totals count each journey at both ends. Link totals include transfer legs.\n'
                   'Costs include operations, maintenance and annualized capital. Allocated net results are not route-closure savings.')
            save(fig, page_path)
        finally:
            plt.close(fig)


def write_analysis(graph, config, result, image_path, json_path=None, title='Current network', evaluator=None, *, reevaluate=True):
    data = network_analysis(graph, config, result, evaluator, reevaluate=reevaluate)
    data['title'] = title
    draw_analysis(image_path, data, title)
    if json_path is not None:
        Path(json_path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    return data


def write_visual_analysis(graph, config, result, output, background, title, evaluator=None, *, export_metrics=True):
    """Publish complete images/metrics atomically at the regular snapshot cadence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    financial = output / '.latest_revenue_cost.tmp.png'
    dashboard = output / '.analysis.tmp.png'
    metrics = output / '.analysis.tmp.json'
    try:
        background.draw(result, financial, title, color_by='revenue_cost', reevaluate=False)
        data = write_analysis(graph, config, result, dashboard, metrics if export_metrics else None, title, evaluator, reevaluate=False)
        financial.replace(output / 'latest_revenue_cost.png')
        dashboard.replace(output / 'analysis.png')
        analysis_financial_path(dashboard).replace(output / 'analysis_financial.png')
        if export_metrics:
            metrics.replace(output / 'analysis.json')
    finally:
        for path in (financial, dashboard, analysis_financial_path(dashboard), metrics):
            path.unlink(missing_ok=True)
    return data


def main():
    """Regenerate visual analysis for a saved snapshot without running a search."""
    import argparse
    from .alignment import load_graph
    from .config import load_config
    from .maps import MapBackground
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('runs/railgen-cache'))
    parser.add_argument('--reevaluate-legacy', action='store_true', help='Explicitly re-evaluate historical genomes under the new service model; old metrics are not preserved')
    args = parser.parse_args()
    snapshot = json.loads((args.run_dir / 'latest.json').read_text())
    config = load_config(args.run_dir / 'run_config.json', profile=None)
    manifest = json.loads((args.run_dir / 'alignment_manifest.json').read_text())
    graph = load_graph(args.cache_dir / manifest['fingerprint'])
    evaluator = Evaluator(graph, config)
    legacy = snapshot.get('model_version') != MODEL_VERSION
    if legacy and not args.reevaluate_legacy:
        parser.error('Historical results need explicit --reevaluate-legacy; service-model metrics are not compatible with old metrics')
    genome, counts = snapshot['genome'], snapshot.get('track_counts')
    if legacy and any(state > 2 for state in genome):
        from .track_layout import migrate_legacy_genome
        genome, counts = migrate_legacy_genome(genome)
    result = evaluator.evaluate(genome, details=True, services=None if legacy else snapshot.get('services'), track_counts=counts)
    title = f"Snapshot {snapshot['snapshot']} · iteration {snapshot['iteration']} · {snapshot['phase'].replace('_', ' ')}"
    data = write_visual_analysis(graph, config, result, args.run_dir,
                                MapBackground(graph, config), title, evaluator)
    report = args.run_dir / 'report.md'
    if report.exists() and 'latest_revenue_cost.png' not in report.read_text():
        with report.open('a') as stream:
            stream.write('\n## Station and corridor analysis\n\n'
                '![Revenue / annual cost map](latest_revenue_cost.png)\n\n'
                '![Cities and demand rankings](analysis.png)\n\n'
                '![Financial rankings](analysis_financial.png)\n\n'
                '[Full station and corridor metrics](analysis.json)\n')
    print(f"Generated {args.run_dir / 'latest_revenue_cost.png'} and {args.run_dir / 'analysis.png'} and {args.run_dir / 'analysis_financial.png'}")
    print(f"Annual allocated corridor costs: €{sum(r['annual_cost_eur'] for r in data['corridors'])/1e6:,.1f}M; "
          f"network: €{data['metrics']['annual_cost_eur']/1e6:,.1f}M")


if __name__ == '__main__':
    main()
