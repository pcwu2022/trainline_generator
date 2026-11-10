"""Fixed regional rail-eligible passenger needs, independent of network selection."""
from .geography import haversine


def rail_need_bound(problem):
    """Optimistic capturable trips that could satisfy best-case journey limits.

    No built track or chosen service enters this calculation. Capacity and actual
    detours can reduce achievable satisfaction, so this is a bound, not a forecast.
    """
    c = problem.config
    frequencies = [f for f in c.frequencies_per_hour
                   if c.minimum_service_frequency <= f <= c.track_trains_per_hour_per_direction]
    if not frequencies:
        return 0.
    maximum_frequency = max(frequencies)
    minimum_frequency = min(f for f in c.frequencies_per_hour if f >= c.minimum_service_frequency)
    stations = {}
    for station in problem.stations:
        stations.setdefault(station['city'], []).append(station)
    total = 0.
    for od in problem.demand:
        capture = 0.
        for a in stations.get(od['origin'], []):
            for b in stations.get(od['destination'], []):
                access = a['access_hours']+b['access_hours']
                # Conservative lower bound: a 1% margin avoids excluding a trip
                # solely because spherical and ellipsoidal distances differ.
                direct = .99*haversine(problem.locations[a['location']]['coordinate'],
                                       problem.locations[b['location']]['coordinate'])
                optimistic_time = access+direct/c.train_speed_kmh+.5/maximum_frequency
                limit = c.maximum_journey_hours
                if c.maximum_journey_detour_ratio:
                    limit = min(limit, access+.5/minimum_frequency
                                +c.maximum_journey_detour_ratio*od['km']/c.train_speed_kmh)
                if optimistic_time <= limit+1e-9:
                    capture = max(capture, a['capture_fraction']*b['capture_fraction'])
        total += od['volume']*capture
    return float(total)
