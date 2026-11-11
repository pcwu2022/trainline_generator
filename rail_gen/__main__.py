"""Run with python -m rail_gen {precompute,run,calibrate,benchmark}."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from .config import load_config


def parser():
    p = argparse.ArgumentParser(description='Fast multi-objective railway network generator')
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('precompute', 'run', 'calibrate', 'benchmark'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', type=Path)
        sub.add_argument('--profile', default='france')
        sub.add_argument('--backtest', type=Path)
        sub.add_argument('--fit', nargs='+', choices=['k', 'a', 'beta', 't0_hours'], default=['k'])
        if name != 'calibrate':
            sub.add_argument('--cache-dir', type=Path, default=Path('runs/railgen-cache'))
        if name in ('run', 'calibrate'):
            sub.add_argument('--output', type=Path, required=True)
        if name == 'run':
            sub.add_argument('--population', type=int)
            sub.add_argument('--generations', type=int)
            sub.add_argument('--seed', type=int)
            sub.add_argument('--workers', type=int)
            sub.add_argument('--sensitivity', action='store_true')
        if name == 'benchmark':
            sub.add_argument('--evaluations', type=int, default=200)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        overrides = {}
        if args.command == 'run':
            overrides['search'] = {key: getattr(args, key) for key in ('population', 'generations', 'seed', 'workers') if getattr(args, key) is not None}
            if args.sensitivity:
                overrides['search']['sensitivity'] = True
        config = load_config(args.config, args.profile, overrides)
        if args.backtest:
            config['backtest_file'] = str(args.backtest)
        if args.command == 'run' and args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f'Output directory is not empty: {args.output}. Choose a new directory.')
        from .calibrate import calibrate
        calibration = None
        if config['backtest_file']:
            config['demand'], calibration = calibrate(config['backtest_file'], config, tuple(args.fit))
            print(f'Demand fit: k={config["demand"]["k"]:.6g}; backtest {calibration["status"]}', flush=True)
        if args.command == 'calibrate':
            if calibration is None:
                raise ValueError('Provide --backtest or backtest_file; see rail_gen/data/backtest_template.csv')
            if args.output.exists():
                raise ValueError(f'Refusing to overwrite {args.output}')
            from .report import write_json
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, calibration)
            print(f'Saved calibration report: {args.output}', flush=True)
            return 0 if calibration['status'] == 'passed' else 2
        from .alignment import cached_graph
        graph = cached_graph(config, args.cache_dir)
        print(f'Candidate graph: {len(graph.cities)} cities, {len(graph.edges)} pairs; {graph.tile_costs.shape[1]} infrastructure cells', flush=True)
        if args.command == 'precompute':
            return 0
        if args.command == 'benchmark':
            from .evaluator import Evaluator
            if args.evaluations <= 0:
                raise ValueError('--evaluations must be positive')
            evaluator = Evaluator(graph, config)
            rng = np.random.default_rng(config['search']['seed'])
            timings = []
            for _ in range(args.evaluations + 5):
                x = rng.integers(0, 3, evaluator.m)
                x[~evaluator.allowed[np.arange(evaluator.m), x]] = 0
                started = time.perf_counter()
                evaluator.evaluate(x)
                timings.append((time.perf_counter() - started) * 1000)
            timings = np.array(timings[5:])
            print(f'Evaluator: median {np.median(timings):.3f} ms, p95 {np.percentile(timings,95):.3f} ms ({args.evaluations} designs; excludes cache loading)', flush=True)
            return 0
        from .search import search
        from .report import sensitivity, write_report
        run = search(graph, config)
        variations = sensitivity(graph, config, run.genomes) if config['search']['sensitivity'] else None
        rows = write_report(graph, config, run, args.output, calibration, variations)
        print(f'Saved {len(rows)} Pareto designs: {args.output / "report.md"}', flush=True)
        if calibration is None:
            print('Uncalibrated scenario: supply sourced observations with --backtest before trusting ridership or finance.', flush=True)
        return 0 if rows else 2
    except (ValueError, OSError, ImportError) as error:
        p.exit(1, f'Railway generator failed: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
