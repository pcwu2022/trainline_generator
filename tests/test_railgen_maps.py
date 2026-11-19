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
from rail_gen.map_style import track_style


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
        for city, radius in zip(graph.cities, [4, 8, 12]):
            city.radius_km = radius
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

    def test_transfer_colors_use_potential_od_demand_on_both_legs(self):
        with tempfile.TemporaryDirectory() as temporary:
            c, graph, _ = self.prepare(temporary)
            background = MapBackground(graph, c)
            e = Evaluator(graph, c)
            result = e.evaluate([1, 1, 0])
            demand = np.zeros((e.n, e.n))
            demand[e.origins, e.destinations] = result.potential_trips
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
            captured = []
            original_save = Figure.savefig
            def capture(figure, path, **kwargs):
                original_save(figure, path, **kwargs)
                axis = figure.axes[0]
                renderer = figure.canvas.get_renderer()
                panel = figure.artists[0].get_window_extent(renderer)
                legend = figure.legends[0].get_window_extent(renderer)
                self.assertGreater(legend.y0, panel.y0)
                self.assertLess(legend.y1, panel.y1)
                for scale in figure.axes[1:]:
                    bounds = scale.get_tightbbox(renderer)
                    self.assertGreater(bounds.y0, panel.y0)
                    self.assertLess(bounds.y1, panel.y1)
                captured.append(dict(position=axis.get_position().bounds, xlim=axis.get_xlim(), ylim=axis.get_ylim(),
                    axis_on=axis.axison, title=axis.get_title(),
                    tracks=[(line.get_color(), line.get_linewidth(), line.get_linestyle()) for line in axis.lines if line.get_zorder() == 4],
                    radii=[patch.radius for patch in axis.patches if hasattr(patch, 'radius')],
                    palette=axis.collections[0].cmap.name, values=axis.collections[0].get_array().copy(),
                    norm=(axis.collections[0].norm.vmin, axis.collections[0].norm.vmax)))
            with matplotlib.rc_context({'savefig.bbox': 'tight', 'figure.autolayout': True}):
                with patch.object(Figure, 'savefig', autospec=True, side_effect=capture):
                    background.draw(result, Path(temporary) / 'populated.png', 'Populated')
                    background.draw(e.evaluate([0, 0, 0]), Path(temporary) / 'empty.png', 'Empty')
                    background.draw(result, Path(temporary) / 'financial.png', 'Financial', color_by='revenue_cost')
            populated, empty, financial = captured
            self.assertFalse(populated['axis_on'])
            self.assertEqual(populated['title'], '')
            np.testing.assert_allclose(populated['position'], (0, 0, 1, 1))
            self.assertAlmostEqual(np.ptp(populated['xlim']), np.ptp(populated['ylim']))
            self.assertEqual(populated['radii'], [4, 8, 12])
            self.assertEqual(financial['radii'], [4, 8, 12])
            self.assertEqual(populated['palette'], 'YlOrRd')
            for actual, (frequency, kind, satisfaction) in zip(populated['tracks'], [(1, 'CR', ratios[0]), (4, 'HSR', ratios[1])]):
                expected = track_style(frequency, kind, satisfaction)
                np.testing.assert_allclose(matplotlib.colors.to_rgba(actual[0]), expected['color'])
                self.assertEqual(actual[1:], (expected['linewidth'], expected['linestyle']))
            self.assertEqual(len(populated['tracks']), 2)
            self.assertEqual(len(financial['tracks']), 2)
            for actual, expected in zip(financial['tracks'], populated['tracks']):
                self.assertEqual(actual[1:], expected[1:])
                self.assertFalse(np.allclose(matplotlib.colors.to_rgba(actual[0]), matplotlib.colors.to_rgba(expected[0])))
            for key in ('position', 'xlim', 'ylim'):
                np.testing.assert_allclose(empty[key], populated[key], atol=1e-12)
                np.testing.assert_allclose(financial[key], populated[key], atol=1e-12)
            with Image.open(Path(temporary) / 'populated.png') as full, Image.open(Path(temporary) / 'empty.png') as blank:
                self.assertEqual(full.size, blank.size)


if __name__ == '__main__':
    unittest.main()
