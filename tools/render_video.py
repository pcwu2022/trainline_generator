"""Replay recorded corridor changes as individually highlighted video updates.

Example: python -m tools.render_video runs/germany_production --profile germany
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rail_gen.alignment import cached_graph
from rail_gen.config import load_config
from rail_gen.live import LiveReporter, snapshot_stats
from rail_gen.report import plot_map
from rail_gen.search import SearchUpdate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('runs/railgen-cache'))
    args = parser.parse_args()
    output = args.output or args.run.with_name(args.run.name + '_line_updates')
    if output.exists() and any(output.iterdir()):
        parser.error('Use an empty output directory')
    config_path = args.config or args.run / 'run_config.json'
    config = load_config(config_path if config_path.exists() else None, args.profile)
    graph = cached_graph(config, args.cache_dir)
    rows = [json.loads(line) for line in (args.run / 'history.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        parser.error('History contains no updates')
    reporter = LiveReporter(graph, config, output, compact=True)
    genome = np.zeros(len(graph.edges), dtype=int)
    for row in rows:
        parent = genome.copy()
        if 'genome' in row:
            genome = np.array(row['genome'], dtype=int)
        else:
            for edge, old, new in row['changes']:
                if genome[edge] != old:
                    raise ValueError(f'History lineage mismatch at iteration {row["iteration"]}')
                genome[edge] = new
        result = reporter.evaluator.evaluate(genome, services=row.get('services'),
                                             track_counts=np.array(row['track_counts']) if 'track_counts' in row else None)
        update = SearchUpdate(row['iteration'], row['phase'], result,
                              row['front_size'], row['evaluations'], row['seconds'],
                              row['accepted'], tuple(row['changes']), parent)
        reporter.animate_update(update)
        print(f'Iteration {row["iteration"]}: {reporter.animation.frames} video updates', flush=True)
    title = snapshot_stats(result, config, len(graph.cities), reporter.unit,
                           update.iteration, 'complete')
    final_map = output / reporter.visuals['map']
    plot_map(graph, result, final_map, title, config, background=reporter.background)
    reporter.animation.append(final_map)
    print(f'Saved {output / reporter.visuals["animation"]}', flush=True)


if __name__ == '__main__':
    main()
