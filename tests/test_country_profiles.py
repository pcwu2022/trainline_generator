"""Country profiles must agree across optimizers and never include a neighbor."""
import json
from pathlib import Path
import unittest

from rail_gen.config import load_config as load_fast
from train_optimizer.settings import load_config as load_optimizer

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CODES = set('AL AD AM AT AZ BY BE BA BG HR CY CZ DK EE FI FR GE DE GR HU IS IE IT KZ XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH TR UA GB VA'.split())


class CountryProfileTests(unittest.TestCase):
    def test_complete_single_country_profiles_in_both_optimizers(self):
        catalogs = []
        for filename, loader in [('rail_gen/config.json', load_fast),
                                 ('train_optimizer/config.json', load_optimizer)]:
            profiles = json.loads((ROOT / filename).read_text())['profiles']
            countries = {name: p for name, p in profiles.items()
                         if name not in {'europe', 'exploratory', 'smoke'}}
            self.assertEqual(len(countries), len(EXPECTED_CODES))
            self.assertEqual({p['countries'][0] for p in countries.values()}, EXPECTED_CODES)
            for name, profile in countries.items():
                with self.subTest(file=filename, country=name):
                    self.assertEqual(len(profile['countries']), 1)
                    west, south, east, north = profile['bbox']
                    self.assertTrue(-180 <= west < east <= 180)
                    self.assertTrue(-90 <= south < north <= 90)
                    loader(profile=name)
                    limit = profile.get('max_cities', profile.get('prep', {}).get('max_cities'))
                    self.assertIsInstance(limit, int)
                    self.assertGreaterEqual(limit, 2)
                    limit = profile.get('max_cities', profile.get('prep', {}).get('max_cities'))
                    self.assertIsInstance(limit, int)
                    self.assertGreaterEqual(limit, 2)
            self.assertEqual(countries['netherlands']['countries'], ['NL'])
            self.assertEqual(countries['belgium']['countries'], ['BE'])
            catalogs.append({name: (p['countries'], p['bbox'], p.get('max_cities', p.get('prep', {}).get('max_cities'))) for name, p in countries.items()})
        self.assertEqual(*catalogs)
