"""Display-only mean elevations for land hexagons absent from population data."""
import json
from pathlib import Path

import h3
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from shapely.geometry import box, mapping, shape, Polygon
from shapely.ops import unary_union


def elevation_hexagons(config, project):
    path = config.get('terrain_dem_file') or config.get('terrain_file')
    if not path or config.get('terrain_kind', 'flat') == 'flat':
        return None
    import rasterio
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.warp import transform_geom
    features = json.loads(Path(config['population_file']).read_text())['features']
    occupied = {f['properties']['h3Index'] for f in features if f['properties'].get('h3Index')}
    if not occupied or not config.get('boundaries_file'):
        return None
    resolutions = {h3.get_resolution(index) for index in occupied}
    if len(resolutions) != 1:
        raise ValueError('Elevation background requires a consistent population hexagon resolution')
    resolution = resolutions.pop()
    boundaries = json.loads(Path(config['boundaries_file']).read_text())['features']
    land = [shape(f['geometry']) for f in boundaries
            if f['properties'].get('iso_a2') in config['countries']]
    if not land:
        return None
    land = unary_union(land).intersection(box(*config['bbox']))
    cells = set(h3.geo_to_cells(mapping(land), resolution)) - occupied
    polygons, elevations = [], []
    with rasterio.open(path) as source:
        if source.count != 1:
            return None  # RGB relief is not elevation data.
        for index in sorted(cells):
            hexagon = Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(index)])
            clipped = hexagon.intersection(land)
            if clipped.is_empty:
                continue
            geometry = transform_geom('EPSG:4326', source.crs, mapping(clipped))
            try:
                window = geometry_window(source, [geometry])
            except rasterio.errors.WindowError:
                continue
            data = source.read(1, window=window, masked=True)
            inside = geometry_mask([geometry], out_shape=data.shape,
                transform=source.window_transform(window), invert=True)
            values = np.ma.array(data, mask=np.ma.getmaskarray(data) | ~inside)
            if not values.count():
                lon, lat = clipped.representative_point().coords[0]
                from rasterio.warp import transform
                x, y = transform('EPSG:4326', source.crs, [lon], [lat])
                values = next(source.sample([(x[0], y[0])], masked=True))
            if not values.count():
                continue
            mean = float(values.mean())
            if not np.isfinite(mean):
                continue
            parts = clipped.geoms if clipped.geom_type == 'MultiPolygon' else [clipped]
            for part in parts:
                if part.geom_type == 'Polygon':
                    polygons.append(project(np.asarray(part.exterior.coords)))
                    elevations.append(mean)
    return polygons, elevations


def add_elevation(ax, elevation):
    if elevation is None or not elevation[0]:
        return None
    cmap = LinearSegmentedColormap.from_list('land_elevation',
        [(0, '#9ab960'), (300/4000, '#b4c67b'),
         (700/4000, '#d2c48c'), (1000/4000, '#bc995f'),
         (2000/4000, '#a17242'), (3000/4000, '#795436'), (1, '#503a29')], N=1024)
    collection = PolyCollection(elevation[0], array=np.asarray(elevation[1]),
        cmap=cmap, norm=Normalize(0, 4000, clip=True),
        edgecolors='white', linewidths=.2, zorder=.8)
    ax.add_collection(collection)
    return collection
