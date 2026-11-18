"""Regional coordinates stay consistent across terrain, population, routes and caches."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pyproj import Geod, Proj, CRS

from rail_gen.alignment import cost_surfaces, route, save_graph, load_graph
from rail_gen.config import load_config
from rail_gen.map_utils import map_projector
from rail_gen.prep import City, population_samples
from rail_gen.projection import regional_projection, project_cities
from rail_gen.water import prepare_crossings
from types import SimpleNamespace


class RegionalProjectionTests(unittest.TestCase):
    def test_region_centers_and_radial_distances(self):
        geod = Geod(ellps='WGS84')
        for name in json.loads((Path(__file__).resolve().parents[1] / 'config/profile_locations.json').read_text()):
            c = load_config(profile=name)
            w, s, e, n = c['bbox']
            center = ((w+e)/2, (s+n)/2)
            projection = regional_projection(c)
            with self.subTest(region=name):
                np.testing.assert_allclose(projection.to_metric.transform(*center), [0, 0], atol=1e-6)
                factors = Proj(projection.crs).get_factors(*center)
                self.assertAlmostEqual(factors.meridional_scale, 1, places=7)
                self.assertAlmostEqual(factors.parallel_scale, 1, places=7)
                for bearing in (0, 90, 180, 270):
                    lon, lat, _ = geod.fwd(*center, bearing, 250000)
                    xy = projection.to_metric.transform(lon, lat)
                    self.assertAlmostEqual(np.linalg.norm(xy), 250000, places=5)
                    np.testing.assert_allclose(projection.to_wgs84.transform(*xy), [lon, lat], atol=1e-8)
                project = map_projector(SimpleNamespace(bbox=c['bbox']))
                np.testing.assert_allclose(project([center, [lon, lat]]),
                    [[0, 0], np.array(xy)/1000], atol=1e-8)

    def test_interleaved_regions_do_not_share_mutable_projection(self):
        taiwan, japan = load_config(profile='taiwan'), load_config(profile='japan')
        a = project_cities([City('a', 'Taipei', 121.56, 25.03, 1e6)], taiwan)[0]
        before = a.xy.copy()
        b = project_cities([City('b', 'Tokyo', 139.69, 35.68, 1e6)], japan)[0]
        np.testing.assert_array_equal(a.xy, before)
        self.assertNotEqual(a.metric_crs, b.metric_crs)
        with self.assertRaisesRegex(ValueError, 'requires a regional projection'):
            _ = City('c', 'Unprojected', 0, 0, 1).xy

    def test_population_samples_match_city_coordinates(self):
        c = load_config(profile='taiwan')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'population.json'
            path.write_text(json.dumps(dict(features=[dict(
                geometry=dict(type='Point', coordinates=[121.56, 25.03]),
                properties=dict(population=100, area_km2=1))])))
            c['population_file'] = str(path)
            xy, people, density = population_samples(c)
            city = project_cities([City('a', 'Taipei', 121.56, 25.03, 1e6)], c)[0]
            np.testing.assert_allclose(xy[0], city.xy)
            np.testing.assert_allclose([people[0], density[0]], [100, 100])

    def test_non_european_route_geometry_and_cache_round_trip(self):
        for bbox in ([121, 24, 122, 26], [139, 35, 140, 36]):
            c = load_config(profile=None, overrides=dict(bbox=bbox, countries=[],
                terrain_kind='flat', terrain_file='', boundaries_file='',
                alignment=dict(padding_km=6, max_capex_meur=0, smooth_cells=0)))
            w, s, e, n = bbox
            cities = project_cities([
                City('a', 'A', w+.4, (s+n)/2, 1e5),
                City('b', 'B', w+.6, (s+n)/2, 1e5)], c)
            samples = (np.array([city.xy for city in cities]), np.ones(2), np.zeros(2))
            graph = route(cities, [(0, 1)], c, samples, progress=None)
            projection = regional_projection(c)
            self.assertTrue(CRS.from_wkt(graph.metadata['metric_crs']).equals(projection.crs))
            for alignment in graph.edges[0].alignments.values():
                np.testing.assert_allclose(alignment.geometry[0], [cities[0].station_lon, cities[0].station_lat])
                np.testing.assert_allclose(alignment.geometry[-1], [cities[1].station_lon, cities[1].station_lat])
                coords = np.array(alignment.geometry)
                x, y = projection.to_metric.transform(coords[:, 0], coords[:, 1])
                length = np.linalg.norm(np.diff(np.column_stack([x, y]), axis=0), axis=1).sum()/1000
                self.assertAlmostEqual(alignment.length_km, length, places=6)
                self.assertTrue(10 < length < 30)
            with tempfile.TemporaryDirectory() as directory:
                save_graph(graph, directory)
                restored = load_graph(directory)
                np.testing.assert_allclose(restored.cities[0].station_xy, cities[0].station_xy)
                self.assertEqual(restored.metadata['metric_crs'], graph.metadata['metric_crs'])

    def test_crossing_portals_use_the_region_frame(self):
        c = load_config(profile='taiwan', overrides=dict(terrain_kind='flat', terrain_file='', boundaries_file=''))
        projection = regional_projection(c)
        from rasterio.transform import from_origin
        transform = from_origin(-4500, 4500, 3000, 3000)
        points = [projection.to_wgs84.transform(x, 0) for x in (-3000, 3000)]
        c['alignment']['water_crossings'] = [dict(id='local', geometry=points)]
        water = np.zeros((3, 3), dtype=bool)
        water[:, 1] = True
        rates = np.full((2, 3, 3), 6.)
        rates[:, water] = 120
        assets = prepare_crossings(c, rates, water, transform)
        self.assertEqual(len(assets), 1)
        self.assertAlmostEqual(assets[0]['length_km'], 6, places=7)
        self.assertAlmostEqual(assets[0]['water_km'], 3, places=7)

    def test_absent_local_boundaries_are_all_sea(self):
        c = load_config(profile='taiwan', overrides=dict(terrain_kind='flat', terrain_file=''))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.json'
            path.write_text(json.dumps(dict(features=[])))
            c['boundaries_file'] = str(path)
            costs, water, _, _ = cost_surfaces(c, (np.array([[0., 0.]]), None, np.zeros(1)))
            self.assertTrue(water.all())
            np.testing.assert_allclose(costs, 120.)
