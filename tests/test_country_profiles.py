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
        self.assertEqual(countries, set('AL AD AM AT AZ BY BE BA BG HR CY CZ DK EE FI FR GE DE GR HU IS IE IT KZ XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH TR UA GB VA JP TW US IN KR MX CA BR AR AU NZ TH VN PH ID CN'.split()))

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

    def test_world_profiles_use_country_boundaries_and_population(self):
        from shapely.geometry import Point, box, shape
        from tools.prepare_country_profiles import PROFILES
        root = DEFAULT_CONFIG.parents[1]
        boundaries = json.loads((root / 'data/country_boundaries.geojson').read_text())['features']
        for name, (code, continent) in PROFILES.items():
            with self.subTest(profile=name):
                config = load_config(profile=name)
                self.assertEqual(config['countries'], [code])
                features = json.loads((root / config['boundaries_file']).read_text())['features']
                self.assertEqual(len(features), 1)
                region = shape(features[0]['geometry'])
                source = next(f for f in boundaries if f['properties']['iso_a2'] == code)
                self.assertTrue(region.equals(shape(source['geometry']).intersection(box(*config['bbox']))))
                cities = json.loads((root / config['cities_file']).read_text())
                self.assertGreaterEqual(len(cities), 4)
                self.assertEqual(len({c['id'] for c in cities}), len(cities))
                self.assertTrue(all(c['country'] == code and region.covers(Point(float(c['lon']), float(c['lat']))) for c in cities))
                cells = json.loads((root / config['population_file']).read_text())['features']
                self.assertTrue(cells)
                self.assertTrue(all(region.covers(shape(f['geometry']).centroid) for f in cells))
                if name == 'indonesia':
                    self.assertGreater(len(cells), 1800)
                    self.assertEqual(len({f['properties']['h3Index'] for f in cells}), len(cells))

    def test_south_korea_uses_only_the_mainland_boundary(self):
        from shapely.geometry import Point, shape
        root = DEFAULT_CONFIG.parents[1]
        config = load_config(profile='south_korea')
        features = json.loads((root / config['boundaries_file']).read_text())['features']
        self.assertEqual(len(features), 1)
        mainland = shape(features[0]['geometry'])
        self.assertEqual(mainland.geom_type, 'Polygon')
        source = json.loads((root / 'data/country_boundaries.geojson').read_text())['features']
        country = next(f for f in source if f['properties']['iso_a2'] == 'KR')
        self.assertTrue(mainland.equals(max(shape(country['geometry']).geoms, key=lambda p: p.area)))
        self.assertTrue(mainland.covers(Point(126.9784, 37.566)))  # Seoul
        for lon, lat in [(126.53, 33.50), (128.62, 34.88), (130.90, 37.49), (125.75, 39.03)]:
            self.assertFalse(mainland.covers(Point(lon, lat)))  # Jeju, Geoje, Ulleungdo, Pyongyang
        cities = json.loads((root / config['cities_file']).read_text())
        self.assertTrue({'Seoul', 'Daejeon', 'Daegu'} <= {c['name'] for c in cities})
        self.assertTrue(all(c['country'] == 'KR' and mainland.covers(Point(float(c['lon']), float(c['lat']))) for c in cities))
        population = json.loads((root / config['population_file']).read_text())['features']
        self.assertTrue(population)
        self.assertTrue(all(mainland.covers(shape(f['geometry']).centroid) for f in population))

    def test_state_profiles_are_clipped_to_their_state(self):
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union
        root = DEFAULT_CONFIG.parents[1]
        for name, state, included, excluded in [
            ('california', 'CA', {'Los Angeles', 'San Diego', 'San Francisco'}, {'Las Vegas', 'Phoenix'}),
            ('texas', 'TX', {'Houston', 'Dallas', 'Austin'}, {'Oklahoma City', 'New Orleans'}),
        ]:
            with self.subTest(profile=name):
                config = load_config(profile=name)
                states = json.loads((root / 'data' / name / 'states.geojson').read_text())['features']
                self.assertEqual({s['properties']['state'] for s in states}, {state})
                region = unary_union([shape(s['geometry']) for s in states])
                cities = json.loads((root / config['cities_file']).read_text())
                self.assertTrue(included <= {c['name'] for c in cities})
                self.assertFalse(excluded & {c['name'] for c in cities})
                self.assertTrue(all(c['country'] == 'US' and region.covers(Point(float(c['lon']), float(c['lat']))) for c in cities))
                cells = json.loads((root / config['population_file']).read_text())['features']
                self.assertTrue(cells)
                self.assertTrue(all(region.covers(shape(f['geometry']).centroid) for f in cells))

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
