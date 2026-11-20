"""Clip bundled country inputs for the additional worldwide profiles."""
import argparse
import json
from pathlib import Path
from shapely.geometry import Point, box, mapping, shape
from shapely.prepared import prep

PROFILES = {
    'mexico': ('MX', 'north-america'), 'canada': ('CA', 'north-america'),
    'brazil': ('BR', 'south-america'), 'argentina': ('AR', 'south-america'),
    'australia': ('AU', 'oceania'), 'new_zealand': ('NZ', 'oceania'),
    'thailand': ('TH', 'asia'), 'vietnam': ('VN', 'asia'),
    'philippines': ('PH', 'asia'), 'indonesia': ('ID', 'asia'), 'china': ('CN', 'asia'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    args = parser.parse_args()
    boundaries = json.loads(Path('data/country_boundaries.geojson').read_text())['features']
    cities = json.loads(Path('data/cities50000.json').read_text())
    populations = {}
    for name in args.profiles:
        code, continent = PROFILES[name]
        config = json.loads(Path(f'config/profiles/{name}.json').read_text())
        country = next(f for f in boundaries if f['properties']['iso_a2'] == code)
        geometry = shape(country['geometry']).intersection(box(*config['bbox']))
        region = prep(geometry)
        selected = [dict(c, id=str(i)) for i, c in enumerate(cities)
                    if c['country'] == code and region.covers(Point(float(c['lon']), float(c['lat'])))]
        sources = ['asia', 'oceania'] if name == 'indonesia' else [continent]
        cells_by_id = {}
        for source in sources:
            if source not in populations:
                populations[source] = json.loads(Path(f'data/{source}.geojson').read_text())['features']
            for feature in populations[source]:
                if region.covers(shape(feature['geometry']).centroid):
                    key = feature['properties'].get('h3Index') or json.dumps(feature['geometry'], sort_keys=True)
                    cells_by_id[key] = feature
        cells = list(cells_by_id.values())
        folder = Path('data') / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'cities.json').write_text(json.dumps(selected, indent=2) + '\n')
        for filename, features in [('boundaries', [dict(type='Feature', properties=country['properties'], geometry=mapping(geometry))]), ('population', cells)]:
            (folder / f'{filename}.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=features), separators=(',', ':')) + '\n')
        print(f'{name}: {len(selected)} cities, {len(cells)} population cells', flush=True)


if __name__ == '__main__':
    main()
