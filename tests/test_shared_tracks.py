import unittest
from types import SimpleNamespace

from train_optimizer.settings import Config
from train_optimizer.model import NetworkModel


class SharedTrackTests(unittest.TestCase):
    def test_intermediate_station_enables_shared_trunk_paid_once(self):
        c = Config.from_dict(dict(train_types=['CR'], maximum_extra_lines=0, maximum_shared_track_lines=10,
            maximum_shared_track_detour=2, maximum_line_stops=3, frequencies_per_hour=[1],
            station_capital_cost=1000, station_annual_cost=0, train_capital_cost=0,
            train_cost_per_km=0, maintenance_fraction=0, cost_weight=1,
            profit_shortfall_weight=0, ticket_per_km=0))
        coordinates = [(0, 51), (1, 51), (2, 51.2), (2, 50.8)]
        locations = [dict(id=i, city=i, xy=(i*100, 0), coordinate=p, center_offset_km=0)
                     for i, p in enumerate(coordinates)]
        edges = []
        for a, b, km, capital in [(0, 1, 70, 1e6), (1, 2, 75, 1e6),
                                   (1, 3, 75, 1e6), (0, 2, 140, 1e10), (0, 3, 140, 1e10)]:
            edges.append(dict(id=len(edges), a=a, b=b, km=km, track_capital=capital,
                              geometry=[coordinates[a], coordinates[b]]))
        p = SimpleNamespace(config=c, cities=[dict(id=i, name=str(i)) for i in range(4)],
            locations=locations, stations=[dict(id=i, city=i, location=i, radius_km=1,
                capture_fraction=1, access_hours=0) for i in range(4)], edges=edges,
            demand=[dict(id=i, origin=0, destination=d, volume=100, km=140)
                    for i, d in enumerate([2, 3])])
        model = NetworkModel(p)
        self.assertTrue({(0, 1, 2), (0, 1, 3)} <= {tuple(l['stops']) for l in model.lines})
        solver = model.solver(c, 10)
        solver.run()
        self.assertEqual(solver.modelStatusToString(solver.getModelStatus()), 'Optimal')
        result = model.describe(solver.getSolution().col_value)
        self.assertEqual({t['id'] for t in result['tracks']}, {0, 1, 2})
        self.assertEqual({s['id'] for s in result['stations']}, {0, 1, 2, 3})
        self.assertAlmostEqual(result['metrics']['capital_cost'], 3e6 + 4000)
        trunk_lines = [l for l in result['lines'] if 0 in l['edges']]
        self.assertGreaterEqual(len(trunk_lines), 2)
        self.assertTrue({(0, 1, 2), (0, 1, 3)} <= {tuple(l['stops']) for l in trunk_lines})
        self.assertAlmostEqual(result['metrics']['served_trips_per_hour'], 200)
