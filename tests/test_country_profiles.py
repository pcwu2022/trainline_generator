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
                if name not in {'europe', 'smoke', 'benelux', 'scandinavia'}:
                    self.assertEqual(len(profile['countries']), 1)
                    countries.update(profile['countries'])
                    self.assertEqual(config['countries'], profile['countries'])
                elif name in {'benelux','scandinavia'}:
                    self.assertEqual(config['countries'], profile['countries'])
                    countries.update(profile['countries'])
        self.assertEqual(countries, set('AL AD AM AT AZ BY BE BA BG HR CY CZ DK EE FI FR GE DE GR HU IS IE IT KZ XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH TR UA GB VA JP TW US IN'.split()))

    def test_custom_profile_and_final_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.json'
            path.write_text(json.dumps({'defaults': {'prep': {'max_cities': 9}},
                'profiles': {'france': {'prep': {'max_cities': 7}}}}))
            self.assertEqual(load_config(path)['prep']['max_cities'], 7)
            self.assertEqual(load_config(path, overrides={'prep': {'max_cities': 4}})['prep']['max_cities'], 4)
        with self.assertRaisesRegex(ValueError, 'Unknown profile'):
            load_config(profile='missing')

    def test_regional_profiles_use_the_correct_population_and_country_filters(self):
        for name, expected in {'benelux': ['BE','NL','LU'], 'scandinavia': ['DK','NO','SE']}.items():
            with self.subTest(profile=name):
                c=load_config(profile=name)
                self.assertEqual(c['countries'],expected)
                self.assertEqual(c['population_file'],'data/europe_3km.geojson')
        self.assertEqual(load_config(profile='india')['population_file'],'data/asia.geojson')
        self.assertEqual(load_config(profile='usa_northeast')['cities_file'],'data/usa_northeast/cities.json')

    def test_northeast_data_is_clipped_to_the_nine_states(self):
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union
        root=DEFAULT_CONFIG.parents[1]
        states=json.loads((root/'data/usa_northeast/states.geojson').read_text())['features']
        self.assertEqual({s['properties']['state'] for s in states},{'CT','ME','MA','NH','RI','VT','NJ','NY','PA'})
        region=unary_union([shape(s['geometry']) for s in states])
        cities=json.loads((root/'data/usa_northeast/cities.json').read_text())
        names={c['name'] for c in cities}
        self.assertTrue({'New York City','Boston','Philadelphia','Pittsburgh'}<=names)
        self.assertFalse({'Washington','Baltimore','Cleveland','Toronto'}&names)
        self.assertTrue(all(c['country']=='US' and region.covers(Point(float(c['lon']),float(c['lat']))) for c in cities))
        population=json.loads((root/'data/usa_northeast/population.geojson').read_text())['features']
        self.assertTrue(all(region.covers(shape(f['geometry']).centroid) for f in population))
