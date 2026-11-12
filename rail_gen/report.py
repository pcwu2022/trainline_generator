"""Re-evaluated front, representative maps, GeoJSON and an honest validation report."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from .alignment import TRACKS
from .track_layout import track_type, MODEL_VERSION
from .evaluator import Evaluator
from .search import nondominated, representative_indices, search


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def geojson(graph, result):
    features = []
    served = set()
    for i in np.flatnonzero(result.genome):
        edge = graph.edges[i]
        track = track_type(result.genome[i])
        alignment = edge.alignments[track]
        served.update((edge.a, edge.b))
        features.append(dict(type='Feature', geometry=dict(type='LineString', coordinates=alignment.geometry),
            properties=dict(edge_id=int(i), node_a=graph.cities[edge.a].id, node_b=graph.cities[edge.b].id,
                track_type=track, track_count=int(result.track_counts[i]), infrastructure_type=track, service_ids=[s["service_id"] for s in result.services if int(i) in s["edge_ids"]], service_types=sorted({s["train_type"] for s in result.services if int(i) in s["edge_ids"]}), length_km=alignment.length_km, capex_meur=float(alignment.capex_meur * result.capital_multiplier[i]),
                directional_annual_load=float(result.directional_load[i]),
                frequency_per_hour_direction=float(result.frequency[i]),
                train_units=int(result.train_units[i]), max_formation=int(result.train_units[i]),
                allocated_net_eur=float(result.edge_net[i]), pct_water=alignment.pct_water,
                crossing_ids=list(alignment.crossing_ids), water_km=alignment.water_km,
                pct_steep=alignment.pct_steep)))
    for i, city in enumerate(graph.cities):
        features.append(dict(type='Feature', geometry=dict(type='Point', coordinates=[city.lon, city.lat]),
            properties=dict(node_id=city.id, name=city.name, population=city.population,
                kappa=city.kappa, access_hours=city.access_hours, served=i in served)))
    return dict(type='FeatureCollection', model_version=MODEL_VERSION, services=result.services, features=features)


def plot_map(graph, result, path, title, config, background=None, color_by='satisfaction'):
    from .maps import MapBackground
    background = background if background is not None else MapBackground(graph, config)
    background.draw(result, path, title, color_by=color_by, reevaluate=False)


def sensitivity(graph, config, baseline, progress=print):
    """Six fresh searches on a fixed candidate/geometry set; compare exact genomes."""
    results = []
    baseline_keys = {tuple(x) for x in baseline}
    for family in ('demand_k', 'capex', 'yield'):
        for factor in (.75, 1.25):
            c = deepcopy(config)
            g = graph
            if family == 'demand_k':
                c['demand']['k'] *= factor
            elif family == 'yield':
                for track in c['rolling_stock'].values():
                    track['yield_eur_pkm'] *= factor
            else:
                g = deepcopy(graph)
                g.tile_costs *= factor
                for edge in g.edges:
                    for alignment in edge.alignments.values():
                        alignment.capex_meur *= factor
                for key in ('station_base_meur', 'station_per_million_meur', 'station_cap_meur'):
                    c['finance'][key] *= factor
                for track in c['rolling_stock'].values():
                    track['train_meur'] *= factor
            if progress:
                progress(f'Sensitivity: {family} x {factor}', flush=True)
            run = search(g, c, progress)
            keys = {tuple(x) for x in run.genomes}
            results.append(dict(parameter=family, factor=factor, front_size=len(keys),
                persistent_designs=[i for i, x in enumerate(baseline) if tuple(x) in keys],
                baseline_overlap=len(keys & baseline_keys), elapsed_seconds=run.elapsed_seconds))
    return dict(method='Fresh search on fixed candidate alignments; exact genome persistence; demand varies k only', variations=results)


def write_report(graph, config, run, directory, calibration=None, sensitivity_report=None, map_background=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'designs').mkdir(exist_ok=True)
    evaluator = Evaluator(graph, config)
    # Deduplicate and recompute in float64, then filter the front once more.
    unique = sorted({tuple(x) for x in run.genomes})
    evaluated = [evaluator.evaluate(x, repair=True, details=True) for x in unique]
    by_genome = {tuple(r.genome): r for r in evaluated if r.feasible}
    evaluated = list(by_genome.values())
    if evaluated:
        evaluated = [evaluated[i] for i in nondominated([r.objectives for r in evaluated])]
    objectives = np.array([r.objectives for r in evaluated])
    selected = representative_indices(objectives, config['search']['representatives'],
                                      config['search']['representative_subsidy_eur']) if evaluated else []
    write_json(directory / 'run_config.json', config)
    write_json(directory / 'search_history.json', run.history)
    write_json(directory / 'alignment_manifest.json', graph.metadata)
    write_json(directory / 'model_manifest.json', dict(model_version=MODEL_VERSION, infrastructure_states={'0': 'unbuilt', '1': 'conventional', '2': 'high-speed'}))
    calibration = calibration or dict(status='not_calibrated', reason='No user-supplied observed backtest table')
    write_json(directory / 'backtest.json', calibration)
    if sensitivity_report is not None:
        write_json(directory / 'sensitivity.json', sensitivity_report)
    rows = []
    loss_flags = []
    plt = _plotting()
    for i, result in enumerate(evaluated):
        row = dict(design_id=f'design_{i:04d}', representative=i in selected, feasible=result.feasible)
        row.update({f'O{j+1}': float(value) for j, value in enumerate(result.objectives)})
        row.update(result.metrics)
        rows.append(row)
        write_json(directory / 'designs' / f'design_{i:04d}.geojson', geojson(graph, result))
        write_json(directory / 'designs' / f'design_{i:04d}.services.json', dict(model_version=MODEL_VERSION, genome=result.genome.tolist(), track_counts=result.track_counts.tolist(), services=result.services))
        for edge in np.flatnonzero(result.genome):
            if result.edge_net[edge] < -200e6:
                loss_flags.append(dict(design_id=row['design_id'], edge_id=int(edge), net_eur=float(result.edge_net[edge])))
    fields = list(rows[0]) if rows else ['design_id', 'representative', 'feasible', 'O1', 'O2', 'O3']
    with (directory / 'pareto.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / 'genomes.npz', genomes=np.array([r.genome for r in evaluated], dtype=int).reshape(-1, evaluator.m),
                        objectives=objectives.reshape(-1, 4 if config['search']['include_time_objective'] else 3), model_version=MODEL_VERSION)
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    if rows:
        points = ax.scatter([-r['net_eur'] / 1e6 for r in rows], [r['passengers'] / 1e6 for r in rows],
                            c=[r['population_served'] / 1e6 for r in rows], cmap='viridis')
        fig.colorbar(points, ax=ax, label='Population captured (millions)')
        for i in selected:
            ax.annotate(str(i), (-rows[i]['net_eur'] / 1e6, rows[i]['passengers'] / 1e6), xytext=(4, 4), textcoords='offset points')
    ax.set(xlabel='Net annual cost (EUR millions; negative = surplus)', ylabel='Annual carried trips (millions)',
           title='Non-dominated railway designs · calibration: ' + calibration['status'])
    fig.savefig(directory / 'pareto.png', dpi=150)
    plt.close(fig)
    warnings = []
    if not config['metro_populations_verified']:
        warnings.append('Metro populations are unverified. The bundled city table contains city records, not a curated metro table; nearby cities may overlap.')
    if config['terrain_kind'] == 'relief':
        warnings.append('Terrain uses the supplied RGB relief classifier, not DEM grades; slope feasibility is unassessed.')
    warnings.append('Parameters are illustrative starting values. No operator/procurement source verification is implied.')
    warnings.append('All track is newly built and fully charged. Compatible gauge, electrification and signalling are assumed. Shared cells are priced once and enforce aggregate train movements; overlapping alignments create no passenger junction without a common station. Infrastructure type and train type are independent.')
    if loss_flags:
        warnings.append(f'{len(loss_flags)} design/edge allocations lose over EUR 200M/year; inspect loss_sanity.json before interpretation.')
    write_json(directory / 'loss_sanity.json', dict(status='flagged' if loss_flags else 'passed', flagged_edges=loss_flags,
        interpretation='Annual net allocated by shared-tile ownership, station degree, fleet time and ridden-distance revenue'))
    lines = ['# Railway Pareto run', '', f'Calibration: **{calibration["status"]}**.', '',
        f'{len(graph.cities)} cities, {len(graph.edges)} candidate pairs, {len(rows)} feasible non-dominated designs.',
        f'Search: {run.evaluations} evaluations in {run.elapsed_seconds:.2f} s (includes seeding).', '',
        '## Assumptions and validation', ''] + ['- ' + warning for warning in warnings]
    lines += ['', '![Pareto front](pareto.png)', '', '## Representative designs', '',
              '| Design | Net EUR M/year | Trips M/year | Captured population M | Fleet CR/HSR |',
              '| --- | ---: | ---: | ---: | ---: |']
    for i in selected:
        r = rows[i]
        lines.append(f'| {r["design_id"]} | {r["net_eur"]/1e6:.1f} | {r["passengers"]/1e6:.2f} | {r["population_served"]/1e6:.2f} | {r["fleet_CR"]}/{r["fleet_HSR"]} |')
    if selected and map_background is None:
        from .maps import MapBackground
        map_background = MapBackground(graph, config)
    for i in selected:
        result = evaluated[i]
        title = f'Design {i}: {result.metrics["passengers"]/1e6:.2f}M annual trips · net EUR {result.metrics["net_eur"]/1e6:.1f}M/year'
        map_name = f'design_{i:04d}.png'
        plot_map(graph, result, directory / map_name, title, config, background=map_background)
        financial_name = f'design_{i:04d}_revenue_cost.png'
        analysis_name = f'design_{i:04d}_analysis.png'
        plot_map(graph, result, directory / financial_name, title, config, background=map_background, color_by='revenue_cost')
        from .analysis import write_analysis
        write_analysis(graph, config, result, directory / analysis_name,
            directory / f'design_{i:04d}_analysis.json', title=f'Design {i}', evaluator=evaluator, reevaluate=False)
        lines += ['', f'### Design {i}', '', f'![Map]({map_name})', '', f'![Revenue/cost map]({financial_name})', '',
                  f'![Station and corridor rankings]({analysis_name})', '',
                  '| Corridor | Type | km | Capex EUR M | Directional annual load | Trains/h/dir | Allocated net EUR M/y |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for edge_id in np.flatnonzero(result.genome):
            edge = graph.edges[edge_id]
            track = track_type(result.genome[edge_id])
            alignment = edge.alignments[track]
            names = f'{graph.cities[edge.a].name} – {graph.cities[edge.b].name}'.replace('|', '\\|')
            lines.append(f'| {names} | {track} ({int(result.track_counts[edge_id])} tracks) | {alignment.length_km:.1f} | {alignment.capex_meur * result.capital_multiplier[edge_id]:.1f} | {result.directional_load[edge_id]:,.0f} | {result.frequency[edge_id]:g} | {result.edge_net[edge_id]/1e6:.1f} |')
        lines += ['', 'Service plan (frequency is per hour in each direction):', '']
        for service in result.services:
            route = ' → '.join(graph.cities[node].name for node in service['route'])
            lines.append(f"- {service['train_type']}: {route}; {service['frequency']:g} trains/h; {service['formation']} units; fleet {service['fleet_units']} units.")
    lines += ['', '## Backtest', '', 'See [backtest.json](backtest.json) for data sources, train/test splits, fitted parameters and individual checks.']
    if calibration.get('checks'):
        lines += ['', '| Check | Status |', '| --- | --- |']
        lines += [f'| {name} | {check["status"]} |' for name, check in calibration['checks'].items()]
    if sensitivity_report:
        lines += ['', '## Sensitivity', '', sensitivity_report['method'], '',
                  '| Parameter | Factor | Pareto designs | Exact baseline designs retained |', '| --- | ---: | ---: | ---: |']
        lines += [f'| {v["parameter"]} | {v["factor"]} | {v["front_size"]} | {v["baseline_overlap"]} |' for v in sensitivity_report['variations']]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    return rows
