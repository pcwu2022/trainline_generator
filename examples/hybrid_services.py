"""Small reproducible, fully built HSR trunk + conventional extension example.

Run from the repository root: python -m examples.hybrid_services --output runs/hybrid-example
The example needs no downloaded geography or observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rail_gen.alignment import Alignment, CandidateGraph, Edge
from rail_gen.config import load_config
from rail_gen.evaluator import Evaluator
from rail_gen.prep import City
from rail_gen.services import Service
from rail_gen.report import geojson, write_json, plot_map
from rail_gen.analysis import write_analysis
from rail_gen.live import cost_log
from rail_gen.track_layout import MODEL_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/hybrid-example'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cities = [City('a', 'Trunk origin', 2, 46, 2e6, .8, .1),
              City('b', 'Trunk junction', 3.5, 46, 1e6, .8, .1),
              City('c', 'Extension terminus', 5, 46, 500000, .8, .1)]
    population_path = args.output / 'population.geojson'
    write_json(population_path, dict(type='FeatureCollection', features=[dict(type='Feature',
        geometry=dict(type='Point', coordinates=[city.lon, city.lat]),
        properties=dict(population=city.population, area_km2=10)) for city in cities]))
    edges = []
    for i, (a, b) in enumerate([(0, 1), (1, 2)]):
        geometry = [[cities[a].lon, cities[a].lat], [cities[b].lon, cities[b].lat]]
        edges.append(Edge(a, b, {'CR': Alignment(120, 720, [i], geometry),
                                 'HSR': Alignment(120, 2400, [i], geometry)}))
    graph = CandidateGraph.build(cities, edges, np.array([[720, 720], [2400, 2400]]))
    config = load_config(profile=None, overrides=dict(terrain_kind='flat', terrain_file='', terrain_dem_file='',
        population_file=str(population_path), boundaries_file='', countries=[], bbox=[1.5, 45.5, 5.5, 46.5],
        demand=dict(k=.01)))
    result = Evaluator(graph, config).evaluate([2, 1], services=[
        Service((0, 1, 2), 'HSR', 2, 2), Service((1, 2), 'CR', 1)], details=True)
    write_json(args.output / 'run_config.json', config)
    write_json(args.output / 'network.geojson', geojson(graph, result))
    write_json(args.output / 'services.json', dict(model_version=MODEL_VERSION, genome=result.genome.tolist(),
        track_counts=result.track_counts.tolist(), services=result.services, metrics=result.metrics))
    (args.output / 'cost_breakdown.log').write_text(cost_log(graph, result, 0, 'hybrid example'))
    plot_map(graph, result, args.output / 'network.png', 'HSR through service on newly built conventional extension', config)
    write_analysis(graph, config, result, args.output / 'analysis.png', args.output / 'analysis.json')
    print(json.dumps(dict(feasible=result.feasible, construction_eur=result.metrics['track_capital_eur'],
        services=result.services, output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
