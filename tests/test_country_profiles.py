"""Indexed country profiles and override precedence."""
import json
from pathlib import Path
import tempfile
import unittest
from rail_gen.config import DEFAULT_CONFIG, PROFILE_LOCATIONS, load_config

class CountryProfileTests(unittest.TestCase):
    def test_indexed_profiles_are_complete_and_valid(self):
        locations = json.loads(PROFILE_LOCATIONS.read_text())
        self.assertNotIn('profiles', json.loads(DEFAULT_CONFIG.read_text()))
        self.assertEqual(set(locations.values()), {
            str(p.relative_to(PROFILE_LOCATIONS.parent))
            for p in (PROFILE_LOCATIONS.parent / 'profiles').glob('*.json')})
        countries = set()
        for name, location in locations.items():
            with self.subTest(profile=name):
                profile = json.loads((PROFILE_LOCATIONS.parent / location).read_text())
                config = load_config(profile=name)
                if name not in {'europe', 'smoke'}:
                    self.assertEqual(len(profile['countries']), 1)
                    countries.update(profile['countries'])
                    self.assertEqual(config['countries'], profile['countries'])
        self.assertEqual(countries, set('AL AD AM AT AZ BY BE BA BG HR CY CZ DK EE FI FR GE DE GR HU IS IE IT KZ XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH TR UA GB VA JP TW'.split()))

    def test_custom_profile_and_final_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.json'
            path.write_text(json.dumps({'defaults': {'prep': {'max_cities': 9}},
                'profiles': {'france': {'prep': {'max_cities': 7}}}}))
            self.assertEqual(load_config(path)['prep']['max_cities'], 7)
            self.assertEqual(load_config(path, overrides={'prep': {'max_cities': 4}})['prep']['max_cities'], 4)
        with self.assertRaisesRegex(ValueError, 'Unknown profile'):
            load_config(profile='missing')
