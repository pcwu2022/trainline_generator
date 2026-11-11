"""Fetch public Mapzen elevation tiles and create a 3 km European trial DEM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from urllib.request import urlretrieve

from fetch_terrain import build_mosaic
from generate_terrain import generate_terrain


def fetch_tiles(bbox, output_dir=Path('data/slope_tiles'), zoom=6):
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -85 < south < north < 85):
        raise ValueError('Expected non-wrapping WGS84 bounds within Web Mercator coverage')
    if isinstance(zoom, bool) or not isinstance(zoom, int) or not 0 <= zoom <= 12:
        raise ValueError('Expected integer zoom between 0 and 12')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    def tile(lon, lat):
        return (int((lon + 180) / 360 * 2**zoom),
                int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**zoom))
    xmin, ymin = tile(west, north)
    xmax, ymax = tile(east, south)
    def download(pair):
        x, y = pair
        path = output_dir / f'{zoom}_{x}_{y}.tif'
        if not path.exists():
            temporary = path.with_suffix('.part')
            try:
                urlretrieve(f'https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{x}/{y}.tif', temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path
    coordinates = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
    print(f'Fetching {len(coordinates)} elevation tiles at zoom {zoom}', flush=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        paths = list(executor.map(download, coordinates))
    mosaic = output_dir / 'mosaic.vrt'
    build_mosaic(paths, mosaic)
    return mosaic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bbox', nargs=4, type=float, default=[-25, 33, 46, 73])
    parser.add_argument('--zoom', type=int, default=6)
    parser.add_argument('--tile-dir', type=Path, default=Path('data/slope_tiles'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/slope_trial'))
    args = parser.parse_args()
    output = args.output_dir / 'europe_terrain_3km.tif'
    if output.exists():
        parser.exit(message=f'Already exists: {output}\n')
    mosaic = fetch_tiles(args.bbox, args.tile_dir, args.zoom)
    for path in generate_terrain(mosaic, args.output_dir, [3]):
        print(f'Generated {path}', flush=True)


if __name__ == '__main__':
    main()
