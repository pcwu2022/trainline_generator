"""Explicit, immutable metric projections centered on each region's bbox."""
from dataclasses import dataclass
from functools import lru_cache

from pyproj import CRS, Transformer


@dataclass(frozen=True)
class Projection:
    crs: CRS
    to_metric: Transformer
    to_wgs84: Transformer


@lru_cache(maxsize=128)
def from_crs(crs):
    metric = CRS.from_user_input(crs)
    return Projection(metric,
        Transformer.from_crs('EPSG:4326', metric, always_xy=True),
        Transformer.from_crs(metric, 'EPSG:4326', always_xy=True))


def regional_projection(config):
    """Azimuthal equidistant: true distances/directions from the region center."""
    west, south, east, north = config['bbox']
    crs = (f'+proj=aeqd +lat_0={(south+north)/2:.12g} '
           f'+lon_0={(west+east)/2:.12g} +datum=WGS84 +units=m +no_defs')
    return from_crs(crs)


def project_cities(cities, config):
    """Keep a city's coordinate frame explicit and serializable with the graph."""
    crs = regional_projection(config).crs.to_string()
    for city in cities:
        city.metric_crs = crs
    return cities
