import json
from pathlib import Path
import tempfile
import unittest

from train_optimizer.settings import Config
from train_optimizer.seeds import population_seeds, label_markets, market_membership


class PopulationSeedTests(unittest.TestCase):
    def test_population_scores_spacing_and_order(self):
        config = Config(bbox=[0, 49, 3, 52], market_radius_km=15, minimum_city_population=100,
                        density_threshold=10, max_cities=3)
        cells = [dict(h3='a', lon=0, lat=50, population=600, density=20),
                 dict(h3='b', lon=.1, lat=50, population=500, density=20),
                 dict(h3='c', lon=1, lat=50, population=900, density=20),
                 dict(h3='d', lon=2, lat=50, population=10000, density=1)]
        seeds = population_seeds(cells, config)
        self.assertEqual([s['seed_h3'] for s in seeds], ['c', 'a'])
        self.assertEqual(seeds[1]['seed_population'], 1100)
        self.assertEqual(seeds, population_seeds(list(reversed(cells)), config))

    def test_country_polygon_filter_excludes_germany(self):
        from shapely.geometry import Point
        from train_optimizer.geography import country_mask
        region = country_mask(Config(countries=['NL', 'BE']))
        self.assertTrue(region.covers(Point(4.9, 52.37)))  # Amsterdam
        self.assertTrue(region.covers(Point(4.35, 50.85)))  # Brussels
        self.assertFalse(region.covers(Point(7.0, 51.45)))  # Essen
        self.assertIsNone(country_mask(Config(countries=[])))
        with self.assertRaisesRegex(ValueError, 'missing codes'):
            country_mask(Config(countries=['XX']))

    def test_city_catalogue_cannot_affect_seed_selection(self):
        cells = [dict(h3='a', lon=4, lat=51, population=1000, density=100)]
        first = population_seeds(cells, Config(cities_file='/missing/catalogue'))
        second = population_seeds(cells, Config(cities_file=''))
        self.assertEqual(first, second)

    def test_marginal_scoring_prefers_unrepresented_population(self):
        config = Config(bbox=[0, 49, 3, 52], market_radius_km=25,
                        minimum_seed_spacing_km=1, minimum_city_population=100,
                        density_threshold=10, max_cities=2)
        cells = [dict(h3='a', lon=0, lat=50, population=800, density=20),
                 dict(h3='b', lon=.1, lat=50, population=700, density=20),
                 dict(h3='c', lon=1, lat=50, population=900, density=20)]
        self.assertEqual([s['seed_h3'] for s in population_seeds(cells, config)], ['a', 'c'])

    def test_compact_city_beats_larger_diffuse_population(self):
        config = Config(bbox=[-1, 49, 3, 52], market_radius_km=25,
                        minimum_city_population=100, density_threshold=10, max_cities=1)
        compact = [dict(h3=f'a{i}', lon=i*.01, lat=50, population=200, density=100)
                   for i in range(3)]
        diffuse = [dict(h3=f'b{i}', lon=1+i*.06, lat=50, population=100, density=100)
                   for i in range(-4, 4)]
        # The diffuse region has more people, but the city has a stronger center.
        seed = population_seeds(compact+diffuse, config)[0]
        self.assertLess(seed['coordinate'][0], .1)
        self.assertEqual(seed['seed_population'], 600)

    def test_radius_proportional_to_square_root_density_without_limits(self):
        config = Config(max_cities=1, market_radius_km=1, center_coefficient=2,
                        minimum_city_population=1, density_threshold=0)
        def radius(density, population):
            cells = [dict(h3='a', lon=4.35, lat=50.85,
                          population=population, density=density)]
            return population_seeds(cells, config)[0]['radius_km']
        self.assertEqual(radius(100, 100), 20)  # Exceeds scoring radius.
        self.assertEqual(radius(400, 100), 40)
        self.assertEqual(radius(100, 1000000), 20)  # No population/area floor.
        self.assertEqual(radius(0, 100), 0)

    def test_membership_uses_owner_radius(self):
        owner, eligible = market_membership([[3, 0], [7, 0], [19, 0]],
                                            [[0, 0], [20, 0]], [100]*3,
                                            Config(density_threshold=100), radii=[5, 2])
        self.assertEqual(eligible.tolist(), [True, False, True])

    def test_brussels_density_calibrates_to_twenty_km(self):
        cells = [dict(h3='a', lon=4.35551, lat=50.8476, population=477544,
                      density=15362.67057748953)]
        self.assertAlmostEqual(population_seeds(cells, Config(max_cities=1))[0]['radius_km'],
                               20, places=3)

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
        config = Config(density_threshold=100, market_radius_km=25)
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
