"""Run with python -m rail_gen {precompute,run,calibrate,benchmark}."""
import argparse
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

import numpy as np

from .config import load_config


def parser():
    p = argparse.ArgumentParser(
        description='Fast multi-objective railway network generator',
        epilog='Use COMMAND --help for options and examples for each command.')
    commands = p.add_subparsers(dest='command', required=True)
    descriptions = {
        'precompute': 'Prepare and cache the candidate graph and terrain alignments.',
        'run': 'Optimize a railway network and save live maps, metrics and Pareto alternatives.',
        'calibrate': 'Fit demand parameters to observed backtest data and save the fit report.',
        'benchmark': 'Measure evaluator latency on random networks using the cached graph.',
    }
    for name, description in descriptions.items():
        sub = commands.add_parser(name, help=description, description=description,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
        if name == 'run':
            sub.epilog = (                'Unspecified search and reporting options use the selected configuration/profile.\n'
                'Profit mode (incremental only): after ceil(iterations / 2) updates, save\n'
                'before_profit.png, before_profit.json and before_profit.geojson, then give\n'
                'normalized annual revenue minus total annual cost 100x the weight of\n'
                'ridership and population rewards. Service reductions are allowed; accepted\n'
                'moves must increase profit. Stop at the iteration limit or profit_saturated\n'
                '(all local moves checked without an acceptable improvement). Time limits\n'
                'and interrupts still apply.\n\n'
                'Outputs: latest.png/json/geojson, progress.gif, progress.png, report.md,\n'
                'cost_breakdown.log and pareto/ alternatives. Use an empty output directory\n'
                'or --overwrite.\n\n'
                'Example: python -m rail_gen run --profile france --iterations 200\n'
                '           --profit --output runs/france-profit --snapshot-every 5')
        elif name == 'calibrate':
            sub.epilog = ('Requires --backtest or backtest_file in the configuration.\n'
                          'Example: python -m rail_gen calibrate --backtest observations.csv\n'
                          '           --fit k beta --output runs/calibration.json')
        sub.add_argument('--config', type=Path, help='JSON overrides for the bundled rail_gen/config.json')
        sub.add_argument('--profile', default='france', help='Named configuration profile (default: france)')
        sub.add_argument('--backtest', type=Path, help='Observed demand CSV; overrides backtest_file and fits demand before running')
        sub.add_argument('--fit', nargs='+', choices=['k', 'a', 'beta', 't0_hours'], default=['k'],
                         help='Demand parameters to fit against backtest data (default: k)')
        if name != 'calibrate':
            sub.add_argument('--cache-dir', type=Path, default=Path('runs/railgen-cache'),
                             help='Reusable geography/alignment cache (default: runs/railgen-cache)')
        if name in ('run', 'calibrate'):
            sub.add_argument('--output', type=Path, required=name == 'calibrate',
                             help=('New calibration report JSON path (required)' if name == 'calibrate'
                                   else 'Run directory (default: runs/<profile>-pareto)'))
        if name == 'run':
            sub.add_argument('--overwrite', action='store_true',
                             help='Delete the output directory contents before starting a fresh run')
            sub.add_argument('--population', type=int, help='Candidates per incremental update or population size (at least 4)')
            sub.add_argument('--generations', '--iterations', dest='generations', type=int,
                             help='Maximum network updates (incremental) or population generations; positive integer')
            sub.add_argument('--seed', type=int, help='Nonnegative random seed for reproducible search')
            sub.add_argument('--workers', type=int, help='Parallel evaluation threads (positive integer; 1 is serial)')
            sub.add_argument('--sensitivity', action='store_true', help='Run six additional searches with demand, capital costs and fare yields varied by +/-25%%')
            sub.add_argument('--profit', action='store_true',
                             help='Switch to profit maximization halfway through; save the pre-switch network (see below)')
            sub.add_argument('--algorithm', choices=['incremental', 'nsga2', 'nsga3'],
                             help='Persistent network edits or NSGA population evolution (bundled default: incremental)')
            sub.add_argument('--time-limit', type=float, help='Positive search time limit in seconds, excluding geography preparation')
            sub.add_argument('--snapshot-every', type=int, help='Positive number of updates/generations between live map/GIF snapshots; final state always saved')
            sub.add_argument('--gif-frame-ms', type=int, help='GIF frame duration: 10..655350 milliseconds, in multiples of 10')
        if name == 'benchmark':
            sub.add_argument('--evaluations', type=int, default=200,
                             help='Timed random designs, excluding 5 warmup evaluations (default: 200)')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        overrides = {}
        if args.command == 'run':
            overrides['search'] = {key: getattr(args, key) for key in ('population', 'generations', 'seed', 'workers') if getattr(args, key) is not None}
            if args.profit:
                overrides['search']['profit'] = True
            if args.sensitivity:
                overrides['search']['sensitivity'] = True
            if args.algorithm:
                overrides['search']['algorithm'] = args.algorithm
            if args.time_limit is not None:
                if args.time_limit <= 0:
                    raise ValueError('--time-limit must be positive')
                overrides['search']['time_limit_seconds'] = args.time_limit
            overrides['reporting'] = {key: value for key, value in
                [('snapshot_every', args.snapshot_every), ('gif_frame_ms', args.gif_frame_ms)]
                if value is not None}
            if args.output is None:
                args.output = Path('runs') / f'{args.profile}-pareto'
        config = load_config(args.config, args.profile, overrides)
        if args.backtest:
            config['backtest_file'] = str(args.backtest)
        if args.command == 'run':
            if args.output.exists() and not args.output.is_dir():
                raise ValueError('--output must be a directory')
            if args.overwrite:
                resolved_output = args.output.resolve()
                if resolved_output == Path.cwd() or resolved_output in Path.cwd().parents:
                    raise ValueError('--overwrite cannot delete the working directory or its ancestors')
                if args.output.is_symlink():
                    raise ValueError('--overwrite requires an output directory that is not a symlink')
            elif args.output.exists() and any(args.output.iterdir()):
                raise ValueError(f'Output directory is not empty: {args.output}. Use --overwrite or a new directory.')
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
        if args.command == 'run' and args.overwrite and args.output.exists():
            shutil.rmtree(args.output)
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
                x = rng.integers(0, evaluator.allowed.shape[1], evaluator.m)
                x[~evaluator.allowed[np.arange(evaluator.m), x]] = 0
                started = time.perf_counter()
                evaluator.evaluate(x)
                timings.append((time.perf_counter() - started) * 1000)
            timings = np.array(timings[5:])
            print(f'Evaluator: median {np.median(timings):.3f} ms, p95 {np.percentile(timings,95):.3f} ms ({args.evaluations} designs; excludes cache loading)', flush=True)
            return 0
        from .search import search
        from .report import sensitivity, write_report
        from .live import LiveReporter
        reporter = LiveReporter(graph, config, args.output, calibration)
        interrupted = threading.Event()
        def stop(signum, frame):
            if interrupted.is_set():
                raise KeyboardInterrupt
            interrupted.set()
            print('Stopping after the current update; saving the evolving network.', flush=True)
        previous_handler = signal.signal(signal.SIGINT, stop)
        try:
            run = search(graph, config, observer=reporter, interrupted=interrupted)
            reporter.finish(run)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        variations = (sensitivity(graph, config, run.genomes)
                      if config['search']['sensitivity'] and run.status != 'interrupted' else None)
        rows = write_report(graph, config, run, args.output / 'pareto', calibration, variations,
                            map_background=reporter.background)
        # Keep the existing machine-readable front paths while grouping all
        # comparison maps/designs separately from the evolving current network.
        for name in ('pareto.csv', 'genomes.npz', 'loss_sanity.json'):
            temporary = args.output / ('.' + name + '.tmp')
            shutil.copyfile(args.output / 'pareto' / name, temporary)
            temporary.replace(args.output / name)
        reason = ('No feasible topology-improving move remains, or the iteration limit was reached, '
                  'before all candidate cities were connected.' if run.status == 'coverage_incomplete'
                  else 'No acceptable move remains in the local neighbourhood.'
                  if run.status == 'neighbourhood_exhausted' else '')
        print(f'Search status: {run.status}' + (f' — {reason}' if reason else ''), flush=True)
        print(f'Current map: {args.output / "latest.png"}\n'
              f'Animation: {args.output / "progress.gif"}\n'
              f'Progress charts: {args.output / "progress.png"}\n'
              f'Report: {args.output / "report.md"}\n'
              f'Pareto alternatives: {len(rows)} designs in {args.output / "pareto"}', flush=True)
        if calibration is None:
            print('Uncalibrated scenario: supply sourced observations with --backtest before trusting ridership or finance.', flush=True)
        return 130 if run.status == 'interrupted' else 0 if rows else 2
    except (ValueError, OSError, ImportError) as error:
        p.exit(1, f'Railway generator failed: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
