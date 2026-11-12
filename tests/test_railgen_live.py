"""Persistent network lineage and real live artifact integration checks."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from test_railgen import config, graph_fixture
from rail_gen.config import load_config
from rail_gen.evaluator import Evaluator
from rail_gen.incremental import consolidation_choice, integration_choice, topology
from rail_gen.live import LiveReporter
from rail_gen.search import search


def incremental_config():
    c = config()
    c['search'].update(algorithm='incremental', generations=12)
    c['boundaries_file'] = ''
    c['reporting']['snapshot_every'] = 2
    c['reporting']['gif_frame_ms'] = 300
    return c


class IncrementalSearchTests(unittest.TestCase):
    def test_updates_inherit_the_preceding_network_and_are_reproducible(self):
        c, g = incremental_config(), graph_fixture()
        updates = []
        first = search(g, c, progress=None, observer=updates.append)
        second = search(g, c, progress=None)
        np.testing.assert_array_equal(first.current_genome, second.current_genome)
        np.testing.assert_array_equal(first.genomes, second.genomes)
        self.assertGreater(len(updates), 2)
        np.testing.assert_array_equal(updates[0].result.genome, [0, 0, 0])
        self.assertTrue(any(update.accepted for update in updates))
        for previous, current in zip(updates, updates[1:]):
            np.testing.assert_array_equal(current.parent_genome, previous.result.genome)
            expected = [(i, int(previous.result.genome[i]), int(current.result.genome[i]))
                        for i in np.flatnonzero(previous.result.genome != current.result.genome)]
            self.assertEqual(list(current.changes), expected)
            self.assertEqual(current.accepted, bool(expected))
            self.assertTrue(current.result.feasible)
            if current.phase == 'growth':
                self.assertGreaterEqual(current.result.metrics['passengers'] + 1e-6,
                                        previous.result.metrics['passengers'])
            if current.phase == previous.phase == 'consolidation':
                self.assertLessEqual(current.result.objectives[0], previous.result.objectives[0] + .01)
        np.testing.assert_array_equal(first.current_genome, updates[-1].result.genome)
        for row in first.history:
            if 'growth_baseline' in row:
                self.assertGreaterEqual(row['metrics']['passengers'] + 1e-6,
                    row['growth_baseline']['passengers'] * c['search']['retained_ridership'])

    def test_rejected_proposals_keep_current_and_respect_budget(self):
        c = incremental_config()
        c['constraints']['upfront_budget_eur'] = 1
        updates = []
        run = search(graph_fixture(), c, progress=None, observer=updates.append)
        self.assertTrue(all(not update.accepted for update in updates))
        for update in updates:
            np.testing.assert_array_equal(update.result.genome, [0, 0, 0])
        self.assertEqual(run.status, 'coverage_incomplete')
        self.assertEqual(len(run.genomes), 1)

    def test_interruption_and_time_limit_preserve_last_accepted_network(self):
        c, g = incremental_config(), graph_fixture()
        stopped = threading.Event()
        seen = []
        def observe(update):
            seen.append(update)
            if update.iteration == 2:
                stopped.set()
        run = search(g, c, progress=None, observer=observe, interrupted=stopped)
        self.assertEqual(run.status, 'interrupted')
        self.assertEqual(seen[-1].iteration, 2)
        np.testing.assert_array_equal(run.current_genome, seen[-1].result.genome)
        c['search']['time_limit_seconds'] = 1e-12
        timed = search(g, c, progress=None)
        self.assertEqual(timed.status, 'time_limit')
        np.testing.assert_array_equal(timed.current_genome, [0, 0, 0])

    def test_service_retention_uses_fixed_baseline(self):
        c = incremental_config()
        e = Evaluator(graph_fixture(), c)
        current = e.evaluate([1, 1, 0])
        candidate = deepcopy(current)
        baseline = current.metrics.copy()
        baseline['passengers'] = 100
        current.metrics['passengers'] = 98
        candidate.metrics['passengers'] = 96.1
        candidate.objectives[0] -= 1e6
        candidate.metrics['mean_generalized_hours'] = baseline['mean_generalized_hours']
        self.assertIsNone(consolidation_choice(current, [candidate], baseline, c))
        candidate.metrics['passengers'] = 98
        self.assertIs(consolidation_choice(current, [candidate], baseline, c), candidate)

    def test_demand_fulfillment_includes_reference_for_unserved_pairs(self):
        e = Evaluator(graph_fixture(), incremental_config())
        empty = e.evaluate([0, 0, 0])
        partial = e.evaluate([1, 0, 0])
        connected = e.evaluate([1, 1, 0])
        denominator = float(e.reference_trips.sum())
        self.assertGreater(denominator, 0)
        self.assertEqual(empty.metrics['need_satisfaction_ratio'], 0)
        for result in [empty, partial, connected]:
            eligible = np.where(np.isfinite(result.journey_time), result.potential_trips, e.reference_trips)
            denominator = float(eligible.sum())
            self.assertEqual(result.metrics['eligible_need_trips'], denominator)
            self.assertAlmostEqual(result.metrics['need_satisfaction_ratio'],
                                   result.metrics['passengers'] / denominator)
        e.base[:] = 0
        self.assertEqual(e.evaluate([0, 0, 0]).metrics['need_satisfaction_ratio'], 0)

    def test_integration_accepts_costly_connections_and_missing_cities(self):
        c = incremental_config()
        e = Evaluator(graph_fixture(), c)
        current = e.evaluate([1, 0, 0])
        connected = e.evaluate([1, 1, 0])
        # Integration must not require an immediate financial or ridership gain.
        connected.objectives[0] = current.objectives[0] + 1e9
        connected.metrics['passengers'] = current.metrics['passengers'] * .99
        self.assertEqual(topology(current.genome, e), (1, 0))
        self.assertEqual(topology(connected.genome, e), (0, 0))
        self.assertIs(integration_choice(current, [connected], e), connected)

    def test_integration_joins_disconnected_served_components(self):
        c = incremental_config()
        base = graph_fixture()
        cities = base.cities + [replace(base.cities[-1], id='d', name='D', lon=7)]
        edges = [replace(edge, a=a, b=b) for edge, (a, b) in
                 zip(base.edges, [(0, 1), (2, 3), (1, 2)])]
        g = graph_fixture(cities=cities, edges=edges)
        e = Evaluator(g, c)
        current = e.evaluate([1, 1, 0])
        connected = e.evaluate([1, 1, 1])
        self.assertEqual(topology(current.genome, e), (0, 1))
        self.assertEqual(topology(connected.genome, e), (0, 0))
        self.assertIs(integration_choice(current, [connected], e), connected)
        # A cheaper disconnected proposal cannot reduce the topology deficit.
        self.assertIsNone(integration_choice(connected, [current], e))

    def test_consolidation_updates_keep_all_candidate_cities_connected(self):
        c = incremental_config()
        c['search']['generations'] = 30
        c['search']['growth_fraction'] = .1
        updates = []
        g = graph_fixture()
        e = Evaluator(g, c)
        search(g, c, progress=None, observer=updates.append)
        consolidation = [u for u in updates if u.phase == 'consolidation']
        self.assertTrue(consolidation)
        for update in consolidation:
            self.assertEqual(topology(update.result.genome, e), (0, 0))

    def test_parallel_neighbourhood_scoring_is_deterministic(self):
        c = incremental_config()
        single = search(graph_fixture(), c, progress=None)
        c['search']['workers'] = 2
        parallel = search(graph_fixture(), c, progress=None)
        np.testing.assert_array_equal(single.current_genome, parallel.current_genome)
        np.testing.assert_array_equal(single.genomes, parallel.genomes)
        np.testing.assert_allclose(single.objectives, parallel.objectives)


class LiveOutputTests(unittest.TestCase):
    def test_live_map_gif_cadence_and_final_snapshot_match_current_network(self):
        c, g = incremental_config(), graph_fixture()
        c['search']['generations'] = 5
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reporter = LiveReporter(g, c, directory)
            snapshots = []
            def observe(update):
                reporter(update)
                if update.iteration == 0 or update.iteration % 2 == 0:
                    data = json.loads((directory / 'latest.json').read_text())
                    self.assertEqual(data['iteration'], update.iteration)
                    np.testing.assert_array_equal(data['genome'], update.result.genome)
                    with Image.open(directory / 'progress.gif') as animation:
                        self.assertEqual(animation.n_frames, data['animation_frames'])
                    snapshots.append(update.iteration)
            run = search(g, c, progress=None, observer=observe)
            reporter.finish(run)
            self.assertEqual(snapshots, [0, 2, 4])
            latest = json.loads((directory / 'latest.json').read_text())
            self.assertEqual(latest['iteration'], run.history[-1]['iteration'])
            self.assertEqual(latest['phase'], run.status)
            np.testing.assert_array_equal(latest['genome'], run.current_genome)
            expected = Evaluator(g, c).evaluate(run.current_genome, details=True)
            self.assertEqual(latest['metrics'], expected.metrics)
            analysis = json.loads((directory / 'analysis.json').read_text())
            self.assertEqual(analysis['genome'], latest['genome'])
            self.assertEqual(analysis['metrics'], latest['metrics'])
            self.assertAlmostEqual(sum(r['annual_cost_eur'] for r in analysis['corridors']) / 1e6,
                                   latest['metrics']['annual_cost_eur'] / 1e6)
            self.assertGreater(expected.metrics['passengers'], 0)
            with Image.open(directory / 'progress.gif') as animation:
                self.assertEqual(animation.n_frames, 4)
                self.assertEqual(animation.info['loop'], 0)
                self.assertEqual(animation.info['duration'], 300)
                first = animation.convert('RGB').copy()
                animation.seek(animation.n_frames - 1)
                self.assertNotEqual(first.tobytes(), animation.convert('RGB').tobytes())
            self.assertEqual({p.name for p in directory.glob('*.png')}, {'latest.png', 'progress.png', 'latest_revenue_cost.png', 'analysis.png'})
            self.assertEqual(list(directory.glob('*.tmp*')), [])
            self.assertEqual(list(directory.glob('.*.tmp*')), [])
            rows = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
            for previous, current in zip(rows, rows[1:]):
                self.assertEqual(current['parent_genome'], previous['genome'])
            with np.load(directory / 'checkpoint.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['genome'], run.current_genome)
                restored = Evaluator(g, c).evaluate(saved['genome'],
                    services=json.loads(str(saved['services_json'])), track_counts=saved['track_counts'])
                self.assertEqual(restored.services, expected.services)
                np.testing.assert_array_equal(restored.objectives, expected.objectives)
                self.assertEqual(str(saved['model_version']), latest['model_version'])
            self.assertIn('Annual net:', (directory / 'cost_breakdown.log').read_text())
            self.assertIn('latest.png', (directory / 'report.md').read_text())
            self.assertIn('pareto/report.md', (directory / 'report.md').read_text())
            geo = json.loads((directory / 'latest.geojson').read_text())
            self.assertEqual(sum(f['geometry']['type'] == 'LineString' for f in geo['features']),
                             np.count_nonzero(run.current_genome))

    def test_legacy_entry_point_routes_the_new_solver_and_live_flags(self):
        from train_optimizer.runner import main
        with patch('rail_gen.__main__.main', return_value=0) as generate:
            self.assertEqual(main(['--profile', 'smoke', '--solver', 'pareto',
                '--output', 'runs/example', '--snapshot-every', '2', '--gif-frame-ms', '300']), 0)
        generate.assert_called_once_with(['run', '--algorithm', 'incremental',
            '--profile', 'smoke', '--output', 'runs/example', '--snapshot-every', '2', '--gif-frame-ms', '300'])

    def test_invalid_reporting_parameters_are_rejected(self):
        for override in [{'reporting': {'snapshot_every': 0}},
                         {'reporting': {'gif_frame_ms': 17}},
                         {'reporting': {'gif_frame_ms': 0}},
                         {'search': {'growth_fraction': 0}},
                         {'search': {'retained_ridership': 1.1}}]:
            with self.assertRaises(ValueError):
                load_config(overrides=override)


if __name__ == '__main__':
    unittest.main()
