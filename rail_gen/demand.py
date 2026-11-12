"""Gravity markets, fixed rail references and network journey-time elasticities."""
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


def reference_demand(market, distance_km, access_hours, parameters):
    """Synthetic conventional-rail reference; trips include both directions.

    The gravity market directly supplies the baseline, without logistic uptake.
    Its scale is absorbed into demand.k; reference time remains fixed for the run.
    """
    hours = (np.asarray(access_hours) + np.asarray(distance_km) *
             parameters['reference_detour_ratio'] / parameters['reference_speed_kmh'] +
             parameters['reference_wait_hours'] + parameters['reference_transfer_hours'])
    trips = 2 * np.asarray(market)
    return hours, trips


def elastic_demand(reference_trips, reference_hours, journey_hours, parameters):
    """Signed changes for all pairs; disconnected journeys have zero demand."""
    ratio = np.asarray(journey_hours) / np.asarray(reference_hours)
    shifted = np.asarray(reference_trips) * ratio ** parameters['shift_elasticity']
    total = np.asarray(reference_trips) * ratio ** parameters['total_elasticity']
    return total, shifted - reference_trips, total - shifted
