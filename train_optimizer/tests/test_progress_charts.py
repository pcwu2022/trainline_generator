"""Progress history preserves physical units and supports older snapshots."""
import tempfile
from pathlib import Path
import unittest
from PIL import Image
from train_optimizer.outputs import draw_progress, progress_metrics


class ProgressChartTests(unittest.TestCase):
    def test_units_and_missing_metrics(self):
        result = dict(stations=[{}], service_lines=2, solver=dict(coverage_fraction=.25),
                      metrics=dict(capital_cost=3e6, annual_operating_cost=2e6,
                                   lifecycle=dict(pv_cost=20e6, annuity_factor=5)))
        values = progress_metrics(result)
        self.assertEqual(values['stations'], 1)
        self.assertEqual(values['coverage_percent'], 25)
        self.assertEqual(values['capital_millions'], 3)
        self.assertEqual(values['annual_cost_millions'], 4)
        self.assertIsNone(values['served_trips_per_hour'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.png'
            draw_progress(path, [dict(iterations=0), dict(iterations=5, plot_metrics=values)])
            with Image.open(path) as image:
                self.assertEqual(image.size, (1680, 1320))
            self.assertFalse((Path(folder)/'progress.tmp.png').exists())
