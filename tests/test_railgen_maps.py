"""Density-source, transfer-demand color and original-style map regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h3
import matplotlib
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from test_railgen import config, graph_fixture
from rail_gen.demand import gravity_matrix
from rail_gen.evaluator import Evaluator
from rail_gen.maps import MapBackground
from train_optimizer.outputs import draw_map


class PopulationMapTests(unittest.TestCase):
    def prepare(self, directory):
        c, graph = config(), graph_fixture()
        c.update(bbox=[1, 45, 7, 47], countries=[], boundaries_file='')
        cells, features = [], []
        for city, population in zip(graph.cities, [1000, 5000, 10000]):
            index = h3.latlng_to_cell(city.lat, city.lon, 5)
            features.append(dict(type='Feature', geometry=dict(type='Point', coordinates=[city.lon, city.lat]),
                properties=dict(population=population, h3Index=index)))
            cells.append(dict(h3=index, density=population / h3.cell_area(index, unit='km^2')))
        path = Path(directory) / 'population.geojson'
        path.write_text(json.dumps(dict(type='FeatureCollection', features=features)))
        c['population_file'] = str(path)
        return c, graph, cells

    def test_background_uses_population_per_cell_area_and_is_prepared_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            c, graph, cells = self.prepare(temporary)
            background = MapBackground(graph, c)
            np.testing.assert_allclose(background.densities, [cell['density'] for cell in cells])
            self.assertEqual(len(background.polygons), 3)
            # Drawing later snapshots must not reload changed/missing source files.
            Path(c['population_file']).unlink()
            empty = Evaluator(graph, c).evaluate([0, 0, 0])
            background.draw(empty, Path(temporary) / 'map.png', 'Empty network')
            self.assertTrue((Path(temporary) / 'map.png').exists())

    def test_transfer_colors_use_fixed_two_way_od_markets_on_both_legs(self):
        with tempfile.TemporaryDirectory() as temporary:
            c, graph, _ = self.prepare(temporary)
            background = MapBackground(graph, c)
            e = Evaluator(graph, c)
            result = e.evaluate([1, 1, 0])
            demand = 2 * gravity_matrix(graph.cities, c)
            trips = dict(zip(zip(e.origins, e.destinations), result.trips))
            expected = [(trips[0, 1] + trips[0, 2]) / (demand[0, 1] + demand[0, 2]),
                        (trips[1, 2] + trips[0, 2]) / (demand[1, 2] + demand[0, 2]), 0]
            ratios = background.satisfaction(result)
            np.testing.assert_allclose(ratios, expected)
            reduced = deepcopy(result)
            reduced.trips *= .5
            reduced.directional_load *= .5
            np.testing.assert_allclose(background.satisfaction(reduced), ratios * .5)
            np.testing.assert_array_equal(background.satisfaction(e.evaluate([0, 0, 0])), np.zeros(3))

    def test_rendered_tracks_match_original_and_animation_framing_is_fixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            c, graph, cells = self.prepare(temporary)
            background = MapBackground(graph, c)
            e = Evaluator(graph, c)
            result = e.evaluate([1, 2, 0])
            result.frequency[:2] = [1, 4]
            ratios = background.satisfaction(result)
            problem = SimpleNamespace(cells=cells,
                cities=[dict(center=(city.lon, city.lat), name=city.name, population=city.population) for city in graph.cities],
                locations=[dict(coordinate=(city.lon, city.lat)) for city in graph.cities],
                stations=[dict(location=i, radius_km=c['prep']['catchment_km']) for i in range(3)],
                edges=[dict(geometry=edge.alignments['CR'].geometry) for edge in graph.edges], demand=[])
            legacy_config = SimpleNamespace(bbox=c['bbox'], terrain_file='', land_file='', accounting_years=10)
            legacy_result = dict(phase='solving', iterations=1, elapsed_seconds=1, snapshot=1, feasible=True,
                lines=[dict(id=i, edges=[i], train_type=track, frequency_per_hour_per_direction=frequency)
                       for i, (track, frequency) in enumerate([('CR', 1), ('HSR', 4)])],
                od_pairs=[dict(id=i, rail_eligible_needs=100) for i in range(2)],
                passenger_flows=[dict(od=i, line=i, board_stop=0, alight_stop=1, trips_per_hour=100 * ratios[i]) for i in range(2)],
                stations=[dict(coordinate=(city.lon, city.lat), radius_km=c['prep']['catchment_km']) for city in graph.cities],
                metrics=dict(served_trips_per_hour=1, total_track_km=1, horizon_profit=0))
            captured = []
            original_save = Figure.savefig
            def capture(figure, path, **kwargs):
                original_save(figure, path, **kwargs)
                axis = figure.axes[0]
                captured.append(dict(position=axis.get_position().bounds, xlim=axis.get_xlim(), ylim=axis.get_ylim(),
                    tracks=[(line.get_color(), line.get_linewidth(), line.get_linestyle()) for line in axis.lines if line.get_zorder() == 4],
                    palette=axis.collections[0].cmap.name, values=axis.collections[0].get_array().copy(),
                    norm=(axis.collections[0].norm.vmin, axis.collections[0].norm.vmax)))
            with matplotlib.rc_context({'savefig.bbox': 'tight', 'figure.autolayout': True}):
                with patch.object(Figure, 'savefig', autospec=True, side_effect=capture):
                    draw_map(Path(temporary) / 'original.png', legacy_config, problem, legacy_result,
                             viewport=background.viewport)
                    background.draw(result, Path(temporary) / 'populated.png', 'Populated')
                    background.draw(e.evaluate([0, 0, 0]), Path(temporary) / 'empty.png', 'Empty')
                    background.draw(result, Path(temporary) / 'financial.png', 'Financial', color_by='revenue_cost')
            original, populated, empty, financial = captured
            self.assertEqual(populated['palette'], original['palette'])
            self.assertEqual(populated['norm'], original['norm'])
            np.testing.assert_allclose(populated['values'], original['values'])
            for actual, expected in zip(populated['tracks'], original['tracks']):
                np.testing.assert_allclose(actual[0], expected[0])
                self.assertEqual(actual[1:], expected[1:])
            self.assertEqual(len(populated['tracks']), 2)
            self.assertEqual(len(financial['tracks']), 2)
            for actual, expected in zip(financial['tracks'], populated['tracks']):
                self.assertEqual(actual[1:], expected[1:])
                self.assertFalse(np.allclose(actual[0], expected[0]))
            for key in ('position', 'xlim', 'ylim'):
                np.testing.assert_allclose(empty[key], populated[key], atol=1e-12)
                np.testing.assert_allclose(financial[key], populated[key], atol=1e-12)
            with Image.open(Path(temporary) / 'populated.png') as full, Image.open(Path(temporary) / 'empty.png') as blank:
                self.assertEqual(full.size, blank.size)


if __name__ == '__main__':
    unittest.main()
