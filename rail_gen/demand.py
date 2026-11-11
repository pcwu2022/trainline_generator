"""Annual one-way gravity markets and smooth generalized-time rail share."""
import numpy as np
from scipy.special import expit

from .prep import distances


def gravity_matrix(cities, config):
    p = config['demand']
    d = distances(cities)
    effective = np.asarray([c.population * c.kappa for c in cities]) ** p['a']
    result = p['k'] * np.outer(effective, effective) / np.maximum(d, p['minimum_distance_km']) ** p['beta']
    result[d > p['maximum_distance_km']] = 0
    np.fill_diagonal(result, 0)
    return result


def rail_share(generalized_hours, parameters):
    return expit((parameters['t0_hours'] - np.asarray(generalized_hours)) / parameters['sigma_hours'])


def fare(distance_km, path_yield, parameters):
    distance = np.asarray(distance_km)
    taper = (np.maximum(distance, parameters['minimum_distance_km']) / parameters['fare_reference_km']) ** (-parameters['fare_taper'])
    return distance * path_yield * taper
