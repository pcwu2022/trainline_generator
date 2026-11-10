"""Regression checks for stable map framing and complete animation frames."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h3
import matplotlib
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from train_optimizer.config import Config
from train_optimizer.outputs import LiveState, Reporter, draw_map, map_figure_size, map_projector, map_viewport, segment_satisfaction, demand_connections


def example_map():
    config = Config.from_dict({'bbox': [4.0, 50.8, 4.8, 51.5]})
    index = h3.latlng_to_cell(51.49, 4.79, 4)
    problem = SimpleNamespace(
        cells=[dict(h3=index, density=150)],
        cities=[dict(center=(4.4, 51.1), name='Example')],
        locations=[dict(coordinate=(4.7, 51.4))],
        stations=[dict(location=0, radius_km=30)],
        edges=[dict(geometry=[(4.2, 50.9), (5.0, 51.2)])], warnings=[])
    result = dict(phase='solving', elapsed_seconds=10, iterations=5, snapshot=2, feasible=False)
    return config, problem, result


class MapOutputTests(unittest.TestCase):
    def test_transfer_flows_use_full_demand_on_each_leg(self):
        problem = SimpleNamespace(demand=[dict(id=0, volume=3000)])
        result = dict(lines=[dict(id=1), dict(id=2)], passenger_flows=[
            dict(od=0, line=line, board_stop=0, alight_stop=1, trips_per_hour=2000)
            for line in (1, 2)])
        self.assertEqual(segment_satisfaction(problem, result), {(1, 0): 2/3, (2, 0): 2/3})

    def test_demand_filter_combines_directions_and_omits_complete_graph(self):
        problem = SimpleNamespace(demand=[
            dict(origin=0, destination=1, volume=60),
            dict(origin=1, destination=0, volume=60),
            dict(origin=0, destination=2, volume=100),
            dict(origin=1, destination=2, volume=10)])
        config = Config.from_dict(dict(demand_map_top_percent=50))
        self.assertEqual(demand_connections(problem, config), [((0, 1), 120), ((0, 2), 100)])
        config.demand_map_top_percent = 100
        self.assertEqual(len(demand_connections(problem, config)), 2)
        config.demand_map_min_trips_per_hour = 110
        self.assertEqual(demand_connections(problem, config), [((0, 1), 120)])

    def test_canvas_scales_with_square_root_of_physical_span(self):
        self.assertEqual(map_figure_size((0, 1000, 0, 500)), (10, 10))
        self.assertEqual(map_figure_size((0, 4000, 0, 2000)), (20, 20))
        self.assertEqual(map_figure_size((0, 500, 0, 1000)), (10, 10))
        self.assertEqual(map_figure_size((0, 10, 0, 20)), (8, 8))

    def test_viewport_contains_complete_cells_corridors_and_catchments(self):
        config, problem, _ = example_map()
        xmin, xmax, ymin, ymax = map_viewport(config, problem)
        project = map_projector(config)
        coordinates = [(lon, lat) for lat, lon in h3.cell_to_boundary(problem.cells[0]['h3'])]
        coordinates.extend(problem.edges[0]['geometry'])
        for x, y in project(coordinates):
            self.assertTrue(xmin < x < xmax)
            self.assertTrue(ymin < y < ymax)
        x, y = project([problem.locations[0]['coordinate']])[0]
        self.assertLess(xmin, x-30)
        self.assertGreater(xmax, x+30)
        self.assertLess(ymin, y-30)
        self.assertGreater(ymax, y+30)

    def test_map_axes_and_canvas_do_not_shift_with_colorbar_or_solution(self):
        config, problem, result = example_map()
        viewport = map_viewport(config, problem)
        captures = []
        savefig = Figure.savefig
        def record(figure, path, **kwargs):
            savefig(figure, path, **kwargs)
            ax = figure.axes[0]
            captures.append((ax.get_position().bounds, ax.get_xlim(), ax.get_ylim()))
        solution = dict(result, feasible=True,
                        lines=[dict(id=0, geometry=problem.edges[0]['geometry'], frequency_per_hour_per_direction=1)],
                        stations=[dict(coordinate=problem.locations[0]['coordinate'], radius_km=30)],
                        metrics=dict(served_trips_per_hour=100, served_fraction=0.1,
                                     total_track_km=50, horizon_profit=1e6))
        with tempfile.TemporaryDirectory() as directory:
            # Explicit layout must override a user's matplotlibrc cropping/layout.
            with matplotlib.rc_context({'savefig.bbox': 'tight', 'figure.autolayout': True}):
                with patch.object(Figure, 'savefig', autospec=True, side_effect=record):
                    for i, (data, state) in enumerate([(None, result), (problem, result), (problem, solution)]):
                        path = Path(directory)/f'frame{i}.png'
                        draw_map(path, config, data, state, viewport=viewport)
                        with Image.open(path) as image:
                            side = int(map_figure_size(viewport)[0] * 130)
                            self.assertEqual(image.size, (side, side))
            for actual in captures[1:]:
                for a, b in zip(captures[0], actual):
                    np.testing.assert_allclose(a, b, atol=1e-12)

    def test_first_image_waits_for_population_data(self):
        config, problem, _ = example_map()
        state = LiveState()
        def render(path, *args, **kwargs):
            Image.new('RGB', (32, 32), color='white').save(path)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch('train_optimizer.outputs.draw_map', side_effect=render):
                reporter = Reporter(state, config, folder)
                reporter.start()
                try:
                    self.assertFalse((folder/'latest.png').exists())
                    self.assertFalse((folder/'progress.gif').exists())
                    state.update(problem=problem)
                    reporter.requested.set()
                    deadline = time.monotonic()+2
                    while time.monotonic() < deadline:
                        latest = folder/'latest.json'
                        if latest.exists() and json.loads(latest.read_text())['animation_frames'] == 1:
                            break
                        time.sleep(0.01)
                    else:
                        self.fail('First complete map frame was not published')
                finally:
                    reporter.finish()
            with Image.open(folder/'progress.gif') as animation:
                self.assertEqual(animation.n_frames, 2)  # First complete map and final state.
            self.assertFalse(list(folder.glob('snapshot_*')))


if __name__ == '__main__':
    unittest.main()
