import json
from pathlib import Path
import tempfile
import unittest

from train_optimizer.config import Config
from train_optimizer.seeds import population_seeds, label_markets, market_membership


class PopulationSeedTests(unittest.TestCase):
    def test_population_scores_spacing_and_order(self):
        config = Config(bbox=[0, 49, 3, 52], minimum_city_population=100,
                        density_threshold=10, max_cities=3)
        cells = [dict(h3='a', lon=0, lat=50, population=600, density=20),
                 dict(h3='b', lon=.1, lat=50, population=500, density=20),
                 dict(h3='c', lon=1, lat=50, population=900, density=20),
                 dict(h3='d', lon=2, lat=50, population=10000, density=1)]
        seeds = population_seeds(cells, config)
        self.assertEqual([s['seed_h3'] for s in seeds], ['a', 'c'])
        self.assertEqual(seeds[0]['seed_population'], 1100)
        self.assertEqual(seeds, population_seeds(list(reversed(cells)), config))

    def test_labels_are_optional_and_do_not_move_markets(self):
        cities = [dict(id=0, name='Market 1', country='', center=(4.3, 50.8),
                       coordinate=(4.31, 50.81), seed_population=1000)]
        original = dict(cities[0])
        label_markets(cities, '')
        label_markets(cities, '/missing/labels.json')
        self.assertEqual(cities[0], original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cities.json'
            path.write_text(json.dumps([dict(name='Example', lon=4.3, lat=50.8, country='BE')]))
            label_markets(cities, path)
        self.assertEqual(cities[0]['name'], 'Example')
        for key in ('coordinate', 'center', 'seed_population'):
            self.assertEqual(cities[0][key], original[key])

    def test_markets_stop_at_25km_and_overlap_has_one_owner(self):
        import numpy as np
        config = Config(density_threshold=100)
        owner, eligible = market_membership(
            np.array([[0, 0], [25, 0], [25.1, 0], [40, 0], [90, 0], [1, 0]]),
            np.array([[0, 0], [60, 0]]), [100, 100, 100, 100, 100, 99], config)
        self.assertEqual(eligible.tolist(), [True, True, False, True, False, False])
        self.assertEqual(owner[3], 1)
        owner, eligible = market_membership([[20, 0]], [[0, 0], [35, 0]], [100], config)
        self.assertTrue(eligible[0])
        self.assertEqual(owner[0], 1)

    def test_radius_must_be_positive(self):
        with self.assertRaises(ValueError):
            Config(market_radius_km=0).validate()


if __name__ == '__main__':
    unittest.main()
