"""Rendered route symbols retain the map between HSR's two thin rails."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_optimizer.map_style import track_style


class TrackSymbolTests(unittest.TestCase):
    def test_hsr_has_two_visible_rails_with_transparent_center(self):
        for track, expected_strokes in [('CR', 1), ('HSR', 2)]:
            with self.subTest(track=track):
                figure = plt.figure(figsize=(1, 1), dpi=100)
                try:
                    ax = figure.add_axes([0, 0, 1, 1], facecolor='#c69c72')
                    ax.set(xlim=(0, 1), ylim=(0, 1))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    style = track_style(20, track, 1)
                    expected_width = (1.7 + 20 * .6) * .25 * (3 if track == 'HSR' else 1)
                    self.assertAlmostEqual(style['linewidth'], expected_width)
                    ax.plot([.1, .9], [.5, .5], **style)
                    figure.canvas.draw()
                    pixels = np.asarray(figure.canvas.buffer_rgba())
                    # Count separated ink runs across the route at its midpoint.
                    background = pixels[25, 50, :3].astype(float)
                    ink = np.max(np.abs(pixels[35:65, 50, :3].astype(float) - background), axis=1) > 30
                    runs = np.count_nonzero(np.diff(np.r_[False, ink].astype(int)) == 1)
                    self.assertEqual(runs, expected_strokes)
                    if track == 'HSR':
                        np.testing.assert_array_equal(pixels[50, 50, :3], background)
                finally:
                    plt.close(figure)
