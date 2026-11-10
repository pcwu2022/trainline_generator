import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from test_heuristic import problem
from train_optimizer.annealing import RoutePool, canonical, evaluate, repair_origin, solve_annealing
from train_optimizer.consolidation import annual_cost, candidates, materialize, service_protection
from train_optimizer.outputs import LiveState, Reporter


def corridor_problem():
    p = problem()
    p.config.minimum_city_served_fraction = 1
    p.cities.append(dict(id=2, name='Groningen'))
    p.locations.append(dict(p.locations[1], id=2, city=2, coordinate=(2, 51)))
    p.stations.append(dict(p.stations[1], id=2, city=2, location=2))
    edge = p.edges[0]
    p.edges = [dict(edge, id=0, a=0, b=2, km=180, track_capital=3e9),
               dict(edge, id=1, a=0, b=1, km=60, track_capital=.8e9),
               dict(edge, id=2, a=1, b=2, km=120, track_capital=1e9)]
    p.demand = [dict(id=0, origin=0, destination=2, volume=100, km=180)]
    design = canonical([((0, 2), (0,), 1), ((0, 1), (1,), 1)])
    return p, design


class ConsolidationTests(unittest.TestCase):
    def test_alignment_replacement_uses_the_cheaper_physical_segment(self):
        p = problem()
        p.edges.append(dict(p.edges[0], id=1, track_capital=.5e9))
        pool = RoutePool(p)
        design = (((0, 1), (0,), 1),)
        candidate = next(item for item in candidates(p, pool, design, [1]) if item[1] == 'alignment')
        changed = materialize(pool, design, candidate)
        self.assertEqual(changed[0][1], (1,))

    def test_invalid_consolidation_settings_are_rejected(self):
        from train_optimizer.config import Config
        for data in [dict(annealing_consolidation_coverage=0), dict(annealing_consolidation_coverage=1.1),
                     dict(annealing_coverage_metric='unknown'),
                     dict(annealing_retained_ridership=0), dict(annealing_retained_ridership=1.1),
                     dict(annealing_journey_time_allowance=-.1), dict(annealing_consolidation_candidates=0)]:
            with self.assertRaises(ValueError):
                Config.from_dict(data)

    def test_targeted_shared_trunk_replaces_direct_corridor(self):
        p, design = corridor_problem()
        pool = RoutePool(p)
        before = evaluate(p, pool, design, 1, lambda: 5, threading.Event(), final=True)
        floors = service_protection(*before[:2], p.config)
        ranked = candidates(p, pool, design, [1])
        candidate = next(item for item in ranked if item[1] == 'hub' and item[2][2] == 1)
        shared = materialize(pool, design, candidate)
        self.assertEqual(sum(1 in ids for stops, ids, f in shared), 2)
        self.assertNotIn(0, {e for stops, ids, f in shared for e in ids})
        after = evaluate(p, pool, shared, 1, lambda: 5, threading.Event(), final=True, protection=floors)
        self.assertIsNotNone(after)
        self.assertLess(annual_cost(*after[:2]), annual_cost(*before[:2]))
        self.assertTrue(after[0].validation(after[1])['valid'])
        self.assertGreaterEqual(sum(after[1][i] for i in after[0].q.values()), floors['minimum_trips'])

    def test_existing_multi_segment_path_is_a_deterministic_candidate(self):
        p, design = corridor_problem()
        design = canonical(list(design)+[((1, 2), (2,), 1)])
        ranked = candidates(p, RoutePool(p), design, [1])
        candidate = next(item for item in ranked if item[1] == 'reuse')
        shared = materialize(RoutePool(p), design, candidate)
        self.assertNotIn(0, {e for stops, ids, f in shared for e in ids})

    def test_reducing_frequency_cannot_violate_the_fixed_ridership_floor(self):
        p = problem()
        p.config.frequencies_per_hour = [1, 2]
        p.config.train_seats = 50
        pool = RoutePool(p)
        before = evaluate(p, pool, (((0, 1), (0,), 2),), 1, lambda: 5, threading.Event(), final=True)
        floors = service_protection(*before[:2], p.config)
        self.assertAlmostEqual(floors['minimum_by_origin']['0'], 98)
        after = evaluate(p, pool, (((0, 1), (0,), 1),), 1, lambda: 5, threading.Event(), final=True, protection=floors)
        self.assertIsNone(after)

    def test_intermediate_station_creates_one_shared_trunk_for_two_branches(self):
        p = problem()
        p.cities.extend([dict(id=2, name='Branch'), dict(id=3, name='New interchange')])
        p.locations[1]['coordinate'] = (2, 51.1)
        p.locations.extend([dict(p.locations[1], id=2, city=2, coordinate=(2, 50.9)),
                            dict(p.locations[1], id=3, city=3, coordinate=(1, 51))])
        p.stations.extend([dict(p.stations[1], id=2, city=2, location=2),
                           dict(p.stations[1], id=3, city=3, location=3)])
        edge = p.edges[0]
        p.edges = [dict(edge, id=i, a=a, b=b, km=km, track_capital=cost)
                   for i, (a, b, km, cost) in enumerate([(0, 1, 150, 2e9), (0, 2, 150, 2e9),
                         (0, 3, 70, .6e9), (3, 1, 80, .6e9), (3, 2, 80, .6e9)])]
        p.demand = [dict(id=i, origin=0, destination=b, volume=100, km=150) for i, b in enumerate((1, 2))]
        design = canonical([((0, 1), (0,), 1), ((0, 2), (1,), 1)])
        pool = RoutePool(p)
        candidate = next(item for item in candidates(p, pool, design, [1]) if item[1] == 'junction_station' and item[2][-1] == 3)
        shared = materialize(pool, design, candidate)
        self.assertEqual(sum(2 in ids for stops, ids, f in shared), 2)
        after = evaluate(p, pool, shared, 1, lambda: 5, threading.Event(), final=True)
        self.assertIsNotNone(after)
        self.assertEqual(len(after[0].y), 3)
        self.assertEqual(len(after[0].x), 4)

    def test_cost_phase_runs_and_protected_checkpoint_resumes(self):
        p, design = corridor_problem()
        p.config.annealing_iterations = 40
        state = LiveState()
        with patch('train_optimizer.annealing.propose', return_value=(design, 'add')):
            status = solve_annealing(p, state, 5, threading.Event())
        self.assertEqual(status, 'Annealing feasible')
        self.assertGreater(state.snapshot()[-1]['cost_improvements'], 0)
        model, vector = state.snapshot()[1:3]
        metadata = json.loads(model.annealing_checkpoint)
        self.assertIsNotNone(metadata['protection'])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'checkpoint.npz'
            np.savez(checkpoint, vector=vector, fingerprint=model.fingerprint(), annealing=model.annealing_checkpoint)
            restored = LiveState()
            stopped = threading.Event()
            stopped.set()
            self.assertEqual(solve_annealing(p, restored, 5, stopped, checkpoint), 'Annealing feasible')
            self.assertTrue(restored.snapshot()[1].validation(restored.snapshot()[2])['valid'])

    def test_cheap_detour_cannot_violate_the_journey_time_floor(self):
        p, design = corridor_problem()
        pool = RoutePool(p)
        before = evaluate(p, pool, design, 1, lambda: 5, threading.Event(), final=True)
        floors = service_protection(*before[:2], p.config)
        pool.edges[2] = dict(pool.edges[2], km=300, track_capital=1e6)
        candidate = next(item for item in candidates(p, pool, design, [1]) if item[1] == 'hub' and item[2][2] == 1)
        detour = materialize(pool, design, candidate)
        self.assertIsNone(evaluate(p, pool, detour, 1, lambda: 5, threading.Event(), final=True, protection=floors))

    def test_growth_repairs_underserved_origins_and_skips_rejected_repairs(self):
        p = problem()
        p.config.minimum_selected_origin_served_fraction = 1
        p.config.train_seats = 50
        p.config.frequencies_per_hour = [1, 2]
        pool = RoutePool(p)
        design = (((0, 1), (0,), 1),)
        record = evaluate(p, pool, design, .1, lambda: 5, threading.Event())[:2]
        repaired = repair_origin(p, pool, design, record, [1, 2])
        self.assertEqual(repaired[0][2], 2)
        self.assertEqual(repair_origin(p, pool, design, record, [1, 2], {repaired}), design)

    def test_preview_shows_one_route_without_creating_a_feasible_checkpoint(self):
        p, design = corridor_problem()
        pool = RoutePool(p)
        partial = evaluate(p, pool, (((0, 1), (1,), 1),), .1, lambda: 5, threading.Event())
        self.assertGreater(partial[4], 0)
        state = LiveState()
        state.update(problem=p, phase='solving', preview=(partial[0].problem, *partial[:2], partial[4]))
        def draw(path, *args, **kwargs):
            Image.new('RGB', (32, 32), 'white').save(path)
        with tempfile.TemporaryDirectory() as folder, patch('train_optimizer.outputs.draw_map', side_effect=draw), patch('train_optimizer.outputs.map_viewport', return_value=(0, 1, 0, 1)):
            reporter = Reporter(state, p.config, folder)
            reporter.capture_growth()
            # A later view must not overwrite the queued first-route frame.
            state.update(preview=None)
            reporter.write()
            result = json.loads((Path(folder)/'latest.json').read_text())
            self.assertTrue(result['exploratory'])
            self.assertFalse(result['feasible'])
            self.assertEqual(len(result['lines']), 1)
            self.assertFalse((Path(folder)/'checkpoint.npz').exists())

    def test_exploratory_view_keeps_the_validated_checkpoint(self):
        p, design = corridor_problem()
        pool = RoutePool(p)
        full = evaluate(p, pool, design, 1, lambda: 5, threading.Event(), final=True)
        partial = evaluate(p, pool, (((0, 1), (1,), 1),), .1, lambda: 5, threading.Event())
        state = LiveState()
        state.update(problem=full[0].problem, model=full[0], vector=full[1], phase='solving',
                     preview=(partial[0].problem, *partial[:2], partial[4]))
        def draw(path, *args, **kwargs):
            Image.new('RGB', (32, 32), 'white').save(path)
        with tempfile.TemporaryDirectory() as folder, patch('train_optimizer.outputs.draw_map', side_effect=draw), patch('train_optimizer.outputs.map_viewport', return_value=(0, 1, 0, 1)):
            reporter = Reporter(state, p.config, folder)
            reporter.write()
            view = json.loads((Path(folder)/'latest.json').read_text())
            best = json.loads((Path(folder)/'best_feasible.json').read_text())
            self.assertTrue(view['exploratory'])
            self.assertFalse(view['feasible'])
            self.assertTrue(best['feasible'])
            with np.load(Path(folder)/'checkpoint.npz', allow_pickle=False) as checkpoint:
                self.assertEqual(str(checkpoint['fingerprint']), full[0].fingerprint())
                self.assertTrue(full[0].validation(checkpoint['vector'])['valid'])
