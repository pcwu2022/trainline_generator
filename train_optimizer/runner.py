"""CLI, native incumbent callbacks, interrupt handling and warm-start checkpoints."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import shutil
import sys
import threading
import time

import highspy
import numpy as np

from .settings import Config, DEFAULT_CONFIG_PATH, load_config
from .geography import prepare
from .model import NetworkModel
from .heuristic import heuristic_problem, solve_heuristic
from .annealing import input_signature, solve_annealing
from .outputs import LiveState, Reporter, atomic_text, clean


def solve(model, state, time_limit, interrupted, checkpoint=None, reporter=None):
    h = model.solver(model.problem.config, time_limit)
    def improve(event):
        vector = np.array(event.data_out.mip_solution, dtype=float, copy=True)
        if not model.validation(vector)['valid']:
            return
        old = state.snapshot()[2]
        if old is None or model.objective(vector) < model.objective(old) - 1e-9:
            state.update(vector=vector)
    def progress(event):
        out = event.data_out
        state.update(solver=clean(dict(primal_bound=out.mip_primal_bound,
                                      dual_bound=out.mip_dual_bound, relative_gap=out.mip_gap,
                                      nodes=out.mip_node_count)))
        state.advance_iteration()
        if reporter:
            reporter.notify_iterations()
        if interrupted.is_set() or (reporter and reporter.failure):
            event.interrupt()
    def simplex_progress(event):
        state.advance_iteration()
        if reporter:
            reporter.notify_iterations()
        if interrupted.is_set() or (reporter and reporter.failure):
            event.interrupt()
    def ipm_progress(event):
        state.advance_iteration()
        if reporter:
            reporter.notify_iterations()
        if interrupted.is_set() or (reporter and reporter.failure):
            event.interrupt()
    h.cbMipImprovingSolution.subscribe(improve)
    h.cbMipInterrupt.subscribe(progress)
    h.cbSimplexInterrupt.subscribe(simplex_progress)
    h.cbIpmInterrupt.subscribe(ipm_progress)
    if checkpoint:
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['fingerprint']) != model.fingerprint():
                raise ValueError('Checkpoint belongs to a different model/configuration/input dataset')
            vector = saved['vector'].copy()
        if not model.describe(vector).get('feasible'):
            raise ValueError('Checkpoint fails current model feasibility validation')
        if h.setSolution(len(vector), np.arange(len(vector), dtype=np.int32), vector) != highspy.HighsStatus.kOk:
            raise RuntimeError('HiGHS rejected checkpoint warm start')
        state.update(vector=vector)
    state.update(phase='solving', message='Searching for a network with a soft profitability target; publishing the best feasible solution')
    h.run()
    solution = h.getSolution()
    if solution.value_valid:
        vector = np.asarray(solution.col_value, dtype=float)
        old = state.snapshot()[2]
        if model.validation(vector)['valid'] and (old is None or model.objective(vector) < model.objective(old)):
            state.update(vector=vector.copy())
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    state.update(solver=clean(dict(status=status, primal_bound=info.objective_function_value,
                                  dual_bound=info.mip_dual_bound, relative_gap=info.mip_gap,
                                  nodes=info.mip_node_count,
                                  simplex_iterations=info.simplex_iteration_count,
                                  ipm_iterations=info.ipm_iteration_count)))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description='Optimize stations, corridors, all-stop services and passenger flows.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='Optional external JSON overrides')
    parser.add_argument('--profile', help='Named profile from config.json (europe, exploratory, france, italy, netherlands, uk)')
    parser.add_argument('--output', help='Output directory (default: runs/<profile>, or runs/netherlands without a profile)')
    parser.add_argument('--solver', choices=['annealing', 'heuristic', 'milp'], help='Override solver_method')
    parser.add_argument('--time-limit', type=float, default=None, help='Total search seconds; selects bounded annealing unless --budget-mode goals is given')
    parser.add_argument('--budget-mode', choices=['goals', 'bounded'],
                        help='Annealing: finish coverage/integration/consolidation goals, or enforce total budgets')
    parser.add_argument('--snapshot-every', type=int, default=None,
                        help='Solver progress steps between frames: native progress callbacks (default: 5)')
    parser.add_argument('--gif-frame-ms', type=int, default=None,
                        help='Playback duration per GIF frame, in multiples of 10 ms (default: 500)')
    parser.add_argument('--gravity-heuristic', action='store_true', help='Seed conventional rail in descending gravitational-pull order')
    parser.add_argument('--gravity-demand-threshold', type=float, help='Gravity phase: served share of capturable demand (default: 0.75)')
    parser.add_argument('--speed-up', action='store_true', help='Sample at most 10 regional pairs and cost candidates per selection')
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument('--resume', help='Compatible checkpoint.npz to use as an incumbent warm start')
    restart.add_argument('--overwrite', action='store_true',
                         help='Delete the output directory contents before starting a fresh run')
    parser.add_argument('--write-default-config', metavar='PATH', help='Write every configurable assumption and exit')
    args = parser.parse_args(argv)
    try:
        data = asdict(load_config(args.config, args.profile))
        if args.solver:
            data['solver_method'] = args.solver
        if args.budget_mode:
            data['annealing_budget_mode'] = args.budget_mode
        elif args.time_limit is not None:
            data['annealing_budget_mode'] = 'bounded'
        if args.gravity_heuristic:
            data['gravity_heuristic'] = True
        if args.gravity_demand_threshold is not None:
            data['gravity_heuristic_demand_threshold'] = args.gravity_demand_threshold
        if args.speed_up:
            data['speed_up'] = True
        config = Config.from_dict(data)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.write_default_config:
        Path(args.write_default_config).write_text(json.dumps(asdict(config), indent=2)+'\n')
        return 0
    args.time_limit = args.time_limit if args.time_limit is not None else config.solver_time_limit
    args.snapshot_every = args.snapshot_every if args.snapshot_every is not None else config.snapshot_every
    args.gif_frame_ms = args.gif_frame_ms if args.gif_frame_ms is not None else config.gif_frame_ms
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('--time-limit must be finite and positive')
    if args.snapshot_every <= 0:
        parser.error('--snapshot-every must be positive')
    if not 10 <= args.gif_frame_ms <= 655350 or args.gif_frame_ms % 10:
        parser.error('--gif-frame-ms must be a multiple of 10 between 10 and 655350')
    goal_mode = config.solver_method == 'annealing' and config.annealing_budget_mode == 'goals'
    run_settings = dict(budget_mode=config.annealing_budget_mode if config.solver_method == 'annealing' else 'bounded',
                        total_search_time_limit_seconds=None if goal_mode else args.time_limit,
                        passenger_lp_time_limit_seconds=args.time_limit if goal_mode else None,
                        proposal_limit=(config.annealing_iterations
                                        if config.solver_method == 'annealing' and not goal_mode else None))
    output = Path(args.output) if args.output is not None else Path('runs') / (args.profile or 'netherlands')
    if output.exists() and not output.is_dir():
        parser.error('--output must be a directory')
    if args.overwrite:
        resolved_output = output.resolve()
        if resolved_output == Path.cwd() or resolved_output in Path.cwd().parents:
            parser.error('--overwrite cannot delete the working directory or its ancestors')
        if output.is_symlink():
            parser.error('--overwrite requires an output directory that is not a symlink')
        if output.exists():
            try:
                shutil.rmtree(output)
            except OSError as error:
                parser.error(f'Cannot overwrite output directory: {error}')
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output directory is not empty; use --overwrite, a new directory or --resume its checkpoint')
    state, interrupted = LiveState(), threading.Event()
    reporter = Reporter(state, config, output, args.snapshot_every, args.gif_frame_ms)
    previous_handlers = {}
    def request_stop(signum, frame):
        interrupted.set()
        state.update(message='Stop requested; preserving the last feasible incumbent')
    for name in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[name] = signal.signal(name, request_stop)
    reporter.start()
    startup_started = time.monotonic()
    startup_done = threading.Event()
    def startup_heartbeat():
        while not startup_done.wait(15):
            _, _, _, phase, message, solver = state.snapshot()
            if solver['iterations']:
                return
            print(f'[{time.monotonic()-startup_started:7.1f}s] Still working ({phase}): {message}', flush=True)
    heartbeat = threading.Thread(target=startup_heartbeat, daemon=True)
    heartbeat.start()
    exit_code = 1
    try:
        def preparation_progress(message):
            state.update(message=message)
            print(f'[{time.monotonic()-startup_started:7.1f}s] {message}', flush=True)
            if interrupted.is_set():
                raise InterruptedError('Stopped during preparation')
            if reporter.failure:
                raise RuntimeError(f'Snapshot writer failed: {reporter.failure}')
        preparation_progress(f'Preparing {args.config} with {config.solver_method} solver')
        if goal_mode:
            preparation_progress(f'Goal mode: no total runtime/proposal cap; passenger LP limit {args.time_limit:g}s')
        else:
            preparation_progress(f'Total search limit {args.time_limit:g}s; proposal limit {run_settings["proposal_limit"] or "none"}')
        problem = prepare(config, preparation_progress, lambda problem: state.update(problem=problem))
        state.update(problem=problem)
        reporter.write()  # Publish fixed candidate demand before evaluating networks.
        if config.solver_method == 'annealing':
            state.update(problem=problem)
            signature = input_signature(problem)
            atomic_text(output/'manifest.json', json.dumps(dict(config=asdict(config), run_settings=run_settings,
                        input_signature=signature,
                        progress=dict(snapshot_every_iterations=args.snapshot_every,
                                      iteration_unit='annealing_proposal', gif_frame_ms=args.gif_frame_ms),
                        warnings=problem.warnings), indent=2))
            status = solve_annealing(problem, state, args.time_limit, interrupted, args.resume, reporter, signature)
            model = state.snapshot()[1]
        else:
            if config.solver_method == 'heuristic':
                preparation_progress('Selecting heuristic station options and corridors')
                problem = heuristic_problem(problem)
            state.update(problem=problem, phase='building', message=f'Building {config.solver_method} network model')
            preparation_progress(f'Building {config.solver_method} network model')
            reporter.requested.set()  # Publish the first complete map before solving.
            model = NetworkModel(problem, preparation_progress)
            state.update(model=model)
            preparation_progress('Fingerprinting model and input files; writing manifest')
            manifest = dict(config=asdict(config), run_settings=run_settings, model_fingerprint=model.fingerprint(),
                            progress=dict(snapshot_every_iterations=args.snapshot_every,
                                          iteration_unit=('heuristic_design_evaluation' if config.solver_method == 'heuristic'
                                                          else 'solver_progress_callback'), gif_frame_ms=args.gif_frame_ms),
                            variables=len(model.names), constraints=len(model.row_lower),
                            input_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                          for path in (config.population_file, config.cities_file, config.country_boundaries_file, config.terrain_file, config.land_file) if path and Path(path).exists()},
                            versions=dict(highs=highspy.Highs().version(), numpy=np.__version__),
                            warnings=problem.warnings)
            atomic_text(output/'manifest.json', json.dumps(manifest, indent=2))
            print(f"Model: {len(model.names):,} variables, {len(model.row_lower):,} constraints", flush=True)
            if interrupted.is_set():
                raise InterruptedError('Stopped before solve')
            solve_method = solve_heuristic if config.solver_method == 'heuristic' else solve
            preparation_progress('Initializing solver and first network evaluation (including presolve)')
            status = solve_method(model, state, args.time_limit, interrupted, args.resume, reporter)
        vector = state.snapshot()[2]
        feasible = model is not None and vector is not None and model.describe(vector).get('feasible')
        if interrupted.is_set():
            phase, exit_code = 'interrupted', 130
        elif feasible:
            phase, exit_code = ('heuristic_feasible' if config.solver_method != 'milp' else
                                'optimal' if status == 'Optimal' else 'time_limit_with_incumbent'), 0
        elif status == 'Infeasible':
            phase, exit_code = 'infeasible', 2
        else:
            phase, exit_code = ('search_exhausted' if config.solver_method != 'milp'
                                else 'no_feasible_solution'), 3
        message = (f'{status}. Heuristic results have no optimality guarantee; failure to find a network does not prove infeasibility.'
                   if config.solver_method != 'milp' else
                   f'HiGHS: {status}. Optimality applies only to the generated candidate model.')
        state.update(phase=phase, message=message)
    except InterruptedError as error:
        state.update(phase='interrupted', message=str(error))
        exit_code = 130
    except Exception as error:
        state.update(phase='error', message=f'{type(error).__name__}: {error}')
        print(f'Error: {error}', file=sys.stderr)
        exit_code = 1
    finally:
        startup_done.set()
        heartbeat.join()
        try:
            reporter.finish()
        finally:
            for name, handler in previous_handlers.items():
                signal.signal(name, handler)
    print(f'Current map: {(output/"latest.png").resolve()}\nProgress charts: {(output/"progress.png").resolve()}\nCost breakdown: {(output/"cost_breakdown.log").resolve()}', flush=True)
    return exit_code
