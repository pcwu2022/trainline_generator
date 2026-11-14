#!/usr/bin/env python3
"""Dispatch current generator commands and legacy optimizer commands."""
import argparse


def main(argv=None):
    routing = argparse.ArgumentParser(add_help=False)
    routing.add_argument('--solver')
    selected, remaining = routing.parse_known_args(argv)
    if selected.solver in ('pareto', 'incremental', 'nsga2', 'nsga3'):
        from rail_gen.__main__ import main as generate
        algorithm = 'incremental' if selected.solver == 'pareto' else selected.solver
        return generate(['run', '--algorithm', algorithm, *remaining])
    from train_optimizer.runner import main as legacy
    return legacy(argv)


if __name__ == '__main__':
    raise SystemExit(main())
