"""HSR uses a normal-width colored and white railway symbol."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_optimizer.map_style import track_style


class TrackSymbolTests(unittest.TestCase):
    def test_hsr_uses_normal_width_colored_and_white_track(self):
        figure = plt.figure(figsize=(2, 1), dpi=150)
        try:
            ax = figure.add_axes([0, 0, 1, 1], facecolor='#c69c72')
            ax.set(xlim=(0, 1), ylim=(0, 1))
            style = track_style(20, 'HSR', 1)
            self.assertEqual(style['linewidth'], track_style(20, 'CR', 1)['linewidth'])
            self.assertEqual(style['color'], track_style(20, 'CR', 1)['color'])
            # Financial maps may replace the color; the railway symbol preserves it.
            style['color'] = 'red'
            ax.plot([.1, .9], [.5, .5], **style)
            figure.canvas.draw()
            pixels = np.asarray(figure.canvas.buffer_rgba())[75, 40:260, :3]
            self.assertTrue(np.any((pixels[:, 0] > 235) & (pixels[:, 1] < 20) & (pixels[:, 2] < 20)))
            self.assertTrue(np.any(np.all(pixels > 235, axis=1)))
        finally:
            plt.close(figure)
