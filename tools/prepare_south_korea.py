"""Prepare South Korea's mainland using the bundled Natural Earth boundaries."""
import json
from pathlib import Path
from shapely.geometry import Point, mapping, shape
from shapely.prepared import prep


def main():
    boundary_source = Path('data/country_boundaries.geojson')
    country = next(f for f in json.loads(boundary_source.read_text())['features']
                   if f['properties']['iso_a2'] == 'KR')
    # The largest component is the peninsula; discard Jeju and offshore islands.
    geometry = max(shape(country['geometry']).geoms, key=lambda polygon: polygon.area)
    mainland = prep(geometry)
    cities = [c for c in json.loads(Path('data/cities50000.json').read_text())
              if c['country'] == 'KR' and mainland.covers(Point(float(c['lon']), float(c['lat'])))]
    population = [f for f in json.loads(Path('data/asia.geojson').read_text())['features']
                  if mainland.covers(shape(f['geometry']).centroid)]
    output = Path('data/south_korea')
    output.mkdir(parents=True, exist_ok=True)
    boundary = dict(type='Feature', properties=country['properties'], geometry=mapping(geometry))
    for name, features in [('boundaries', [boundary]), ('population', population)]:
        (output / f'{name}.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=features), separators=(',', ':')) + '\n')
    (output / 'cities.json').write_text(json.dumps(cities, indent=2) + '\n')
    print(f'Prepared {len(cities)} cities and {len(population)} mainland population cells')


if __name__ == '__main__':
    main()
