"""Clip existing city/population inputs to the Census Bureau's nine Northeast states.

Run before exporting the usa_northeast browser profile. No additional dependencies.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile
from shapely.geometry import Point, Polygon, MultiPolygon, mapping, shape
from shapely.ops import unary_union
from shapely.prepared import prep

SOURCE = 'https://www2.census.gov/geo/tiger/GENZ2024/kml/cb_2024_us_state_500k.zip'
STATES = {'CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'}
NS = {'k': 'http://www.opengis.net/kml/2.2'}


def prepare_states(raw, output, states=None):
    states = STATES if states is None else set(states)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        root = ET.fromstring(archive.read('cb_2024_us_state_500k.kml'))
    features = []
    for place in root.findall('.//k:Placemark', NS):
        props = {v.attrib['name']: v.text for v in place.findall('.//k:SimpleData', NS)}
        if props.get('STUSPS') not in states:
            continue
        polygons = []
        for item in place.findall('.//k:Polygon', NS):
            def ring(tag):
                text = tag.find('.//k:coordinates', NS).text
                return [tuple(map(float, v.split(',')[:2])) for v in text.split()]
            shell = ring(item.find('k:outerBoundaryIs', NS))
            holes = [ring(v) for v in item.findall('k:innerBoundaryIs', NS)]
            polygons.append(Polygon(shell, holes))
        geometry = MultiPolygon(polygons)
        features.append(dict(type='Feature', properties=dict(state=props['STUSPS'], name=props['NAME']), geometry=mapping(geometry)))
    if {f['properties']['state'] for f in features} != states:
        raise ValueError('Census KML does not contain exactly the requested states')
    region = prep(unary_union([shape(f['geometry']) for f in features]))
    cities = [c for c in json.loads(Path('data/cities50000.json').read_text())
              if c['country'] == 'US' and region.covers(Point(float(c['lon']), float(c['lat'])))]
    population = json.loads(Path('data/north-america.geojson').read_text())
    cells = [f for f in population['features'] if region.covers(shape(f['geometry']).centroid)]
    output.mkdir(parents=True, exist_ok=True)
    (output/'states.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=features), separators=(',', ':'))+'\n')
    (output/'cities.json').write_text(json.dumps(cities, indent=2)+'\n')
    (output/'population.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=cells), separators=(',', ':'))+'\n')
    (output/'source.json').write_text(json.dumps(dict(source=SOURCE, sha256=hashlib.sha256(raw).hexdigest(),
        states=sorted(states), city_source='data/cities50000.json', population_source='data/north-america.geojson',
        cities=len(cities), population_cells=len(cells), selection='City coordinates and population-cell centroids covered by state polygons'), indent=2)+'\n')
    print(f'Prepared {len(cities)} cities and {len(cells)} population cells in {output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states-kml', type=Path, help='Previously downloaded official Census KML zip')
    p.add_argument('--output', type=Path, default=Path('data/usa_northeast'))
    args = p.parse_args()
    raw = args.states_kml.read_bytes() if args.states_kml else urlopen(SOURCE, timeout=60).read()
    prepare_states(raw, args.output)


if __name__ == '__main__':
    main()
