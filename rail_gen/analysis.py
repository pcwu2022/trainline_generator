"""Comparable annual station/corridor rankings and allocated financial analysis."""
import json
from pathlib import Path

import numpy as np

from .alignment import TRACKS
from .evaluator import Evaluator


def corridor_satisfaction(evaluator, result):
    """Same fixed capturable path markets as the satisfaction map, transfers included."""
    e = evaluator
    active = np.flatnonzero(result.genome)
    length = e.lengths[np.arange(e.m), result.genome]
    speeds = np.choose(result.genome, [1, e.config['tracks']['CR']['speed_kmh'],
                                      e.config['tracks']['HSR']['speed_kmh']])
    time = length / speeds + (result.genome > 0) * e.config['operations']['dwell_hours']
    incidence, _, _, _ = e._paths(active, time)
    eligible = np.asarray(incidence @ np.where(result.trips > 0, 2 * e.base, 0)).ravel()
    carried = 2 * result.directional_load
    ratios = np.divide(carried, eligible, out=np.zeros(e.m), where=eligible > 0)
    return np.clip(ratios, 0, 1), eligible


def network_analysis(graph, config, result, evaluator=None):
    """Recompute detailed allocations; station markets count OD endpoints, not transfers."""
    e = evaluator or Evaluator(graph, config)
    result = e.evaluate(result.genome, details=True)
    active = np.flatnonzero(result.genome)
    served = set(np.r_[e.a[active], e.b[active]].tolist())
    station_market = np.bincount(np.r_[e.origins, e.destinations],
        weights=np.tile(2 * e.base, 2), minlength=e.n)
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
            track=TRACKS[result.genome[i] - 1], satisfaction=float(ratios[i]) if markets[i] > 0 else None,
            carried_trips=float(2 * result.directional_load[i]), eligible_trips=float(markets[i]),
            revenue_eur=revenue, annual_cost_eur=cost, annual_net_eur=net,
            revenue_cost_ratio=revenue / cost if cost > 0 else None,
            frequency=float(result.frequency[i])))
    return dict(genome=result.genome.tolist(), metrics=result.metrics, stations=stations, corridors=corridors,
        definitions=dict(station_satisfaction='Carried two-way annual OD endpoint trips / fixed capturable endpoint market, including unserved destinations; transfers excluded.',
            corridor_satisfaction='Carried two-way annual leg trips / fixed capturable OD markets assigned to that corridor; transfers included; same as satisfaction map.',
            annual_cost='Operations plus annualized capital, maintenance, station and fleet allocations; shared infrastructure counted once.',
            profit='Annual allocated revenue minus annual allocated cost; not upfront investment payback.',
            lines='City-pair corridors, not reconstructed through-running service lines.'))


def _count(value):
    return f'{value/1e6:,.2f}M' if abs(value) >= 1e6 else f'{value/1e3:,.1f}k' if abs(value) >= 1000 else f'{value:,.0f}'


def draw_analysis(path, analysis, title='Current network', limit=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from textwrap import fill

    m = analysis['metrics']
    stations = [row for row in analysis['stations'] if row['served']]
    corridors = analysis['corridors']
    panels = [
        ('Stations · highest satisfaction', stations, 'satisfaction', 100, 'Capturable endpoint market served (%)', '#217c55',
         lambda r: f"{r['satisfaction']:.1%} · {_count(r['carried_trips'])} / {_count(r['eligible_trips'])}"),
        ('Stations · most endpoint trips', stations, 'carried_trips', 1e-6, 'Million endpoint trips / year', '#286a92',
         lambda r: _count(r['carried_trips'])),
        ('Corridors · highest satisfaction', corridors, 'satisfaction', 100, 'Assigned capturable market served (%)', '#217c55',
         lambda r: f"{r['satisfaction']:.1%} · {_count(r['carried_trips'])} / {_count(r['eligible_trips'])}"),
        ('Corridors · best revenue / cost', corridors, 'revenue_cost_ratio', 1, 'Annual revenue / annual allocated cost · 1 = break-even', '#79632b',
         lambda r: f"{r['revenue_cost_ratio']:.2f}×"),
        ('Corridors · highest annual cost', corridors, 'annual_cost_eur', 1e-6, 'EUR million / year', '#aa7648',
         lambda r: f"€{r['annual_cost_eur']/1e6:,.1f}M"),
        ('Corridors · highest annual revenue', corridors, 'revenue_eur', 1e-6, 'EUR million / year', '#286a92',
         lambda r: f"€{r['revenue_eur']/1e6:,.1f}M"),
        ('Corridors · best annual net results', corridors, 'annual_net_eur', 1e-6, 'EUR million / year · negative = loss', None,
         lambda r: f"€{r['annual_net_eur']/1e6:+,.1f}M"),
        ('Corridors · largest annual losses', [r for r in corridors if r['annual_net_eur'] < 0], 'annual_net_eur', -1e-6,
         'Allocated loss · EUR million / year', '#ae403b', lambda r: f"€{-r['annual_net_eur']/1e6:,.1f}M"),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(22, 21), layout='constrained')
    figure.set_facecolor('#f6f7f5')
    try:
        for ax, (heading, rows, key, scale, xlabel, color, label) in zip(axes.flat, panels):
            rows = [r for r in rows if r[key] is not None and np.isfinite(r[key])]
            rows = sorted(rows, key=lambda r: (-r[key] * scale, r['name']))[:limit]
            ax.set_title(heading, loc='left', fontsize=13, fontweight='bold', pad=12)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis='x', alpha=.16)
            if not rows:
                ax.text(.5, .5, 'No qualifying stations / corridors', ha='center', va='center', transform=ax.transAxes)
                ax.set_yticks([])
                continue
            values = np.array([r[key] * scale for r in rows])
            colors = color or ['#217c55' if r[key] >= 0 else '#ae403b' for r in rows]
            bars = ax.barh(np.arange(len(rows)), values, color=colors, height=.64)
            names = [fill(f'{i+1}. ' + r['name'] + (f" [{r['track']}]" if 'track' in r else ''), 32)
                     for i, r in enumerate(rows)]
            ax.set_yticks(np.arange(len(rows)), labels=names, fontsize=8)
            ax.tick_params(axis='y', length=0)
            ax.invert_yaxis()
            ax.bar_label(bars, labels=[label(r) for r in rows], padding=5, fontsize=8)
            # Reserve room for numeric labels, including negative-net bars.
            low, high = min(0, values.min()), max(0, values.max())
            span = max(high-low, 1)
            ax.set_xlim(low - span * (.40 if low < 0 else .01), high + span * .60)
            ax.axvline(0, color='#a1a6a1', linewidth=.7)
            if key == 'revenue_cost_ratio':
                ax.axvline(1, color='#555555', linewidth=.8, linestyle='--')
        ratio = m['revenue_eur'] / m['annual_cost_eur'] if m['annual_cost_eur'] > 0 else None
        ratio_label = f'{ratio:.2f}×' if ratio is not None else 'n/a'
        figure.suptitle(f'Network station & corridor analysis · {title}\n'
            f"{m['active_stations']} active stations · {m['active_edges']} corridors · {_count(m['passengers'])} trips/year · "
            f"Satisfaction {m['need_satisfaction_ratio']:.1%}\n"
            f"Annual revenue €{m['revenue_eur']/1e6:,.1f}M · Cost €{m['annual_cost_eur']/1e6:,.1f}M · "
            f"Net €{m['net_eur']/1e6:+,.1f}M · Revenue/cost {ratio_label}", fontsize=16)
        figure.supxlabel('Top 10 per panel · Numbers on satisfaction bars: carried / eligible two-way annual trips. '
            'Station endpoint counts include a trip at both ends; corridor counts include transfer legs.\n'
            'Station satisfaction includes unserved destinations. Corridors use the satisfaction map’s assigned markets. '
            'Costs include allocated annualized capital, maintenance, operations, stations and fleet.\n'
            'Lines represent modeled city-pair corridors. Annual net is revenue minus annual cost; allocations are not independent route-closure savings.', fontsize=10)
        figure.savefig(path, dpi=140)
    finally:
        plt.close(figure)


def write_analysis(graph, config, result, image_path, json_path=None, title='Current network', evaluator=None):
    data = network_analysis(graph, config, result, evaluator)
    data['title'] = title
    draw_analysis(image_path, data, title)
    if json_path is not None:
        Path(json_path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    return data


def write_visual_analysis(graph, config, result, output, background, title, evaluator=None):
    """Publish complete images/metrics atomically at the regular snapshot cadence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    financial = output / '.latest_revenue_cost.tmp.png'
    dashboard = output / '.analysis.tmp.png'
    metrics = output / '.analysis.tmp.json'
    try:
        background.draw(result, financial, title, color_by='revenue_cost')
        data = write_analysis(graph, config, result, dashboard, metrics, title, evaluator)
        financial.replace(output / 'latest_revenue_cost.png')
        dashboard.replace(output / 'analysis.png')
        metrics.replace(output / 'analysis.json')
    finally:
        for path in (financial, dashboard, metrics):
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
    args = parser.parse_args()
    snapshot = json.loads((args.run_dir / 'latest.json').read_text())
    config = load_config(args.run_dir / 'run_config.json', profile=None)
    manifest = json.loads((args.run_dir / 'alignment_manifest.json').read_text())
    graph = load_graph(args.cache_dir / manifest['fingerprint'])
    evaluator = Evaluator(graph, config)
    result = evaluator.evaluate(snapshot['genome'], details=True)
    title = f"Snapshot {snapshot['snapshot']} · iteration {snapshot['iteration']} · {snapshot['phase'].replace('_', ' ')}"
    data = write_visual_analysis(graph, config, result, args.run_dir,
                                MapBackground(graph, config), title, evaluator)
    report = args.run_dir / 'report.md'
    if report.exists() and 'latest_revenue_cost.png' not in report.read_text():
        with report.open('a') as stream:
            stream.write('\n## Station and corridor analysis\n\n'
                '![Revenue / annual cost map](latest_revenue_cost.png)\n\n'
                '![Station and corridor rankings](analysis.png)\n\n'
                '[Full station and corridor metrics](analysis.json)\n')
    print(f"Generated {args.run_dir / 'latest_revenue_cost.png'} and {args.run_dir / 'analysis.png'}")
    print(f"Annual allocated corridor costs: €{sum(r['annual_cost_eur'] for r in data['corridors'])/1e6:,.1f}M; "
          f"network: €{data['metrics']['annual_cost_eur']/1e6:,.1f}M")


if __name__ == '__main__':
    main()
