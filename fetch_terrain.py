"""Download Copernicus DEM tiles with EODAG and generate metric elevation grids."""
import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import rasterio

from generate_terrain import generate_terrain


def build_mosaic(paths, output):
    """Build a lazy, north-up VRT; never materialize the full-resolution mosaic."""
    tiles = []
    for path in paths:
        with rasterio.open(path) as src:
            t = src.transform
            if src.count != 1 or src.crs is None or t.b or t.d or t.a <= 0 or t.e >= 0:
                raise ValueError(f'Expected a north-up, georeferenced single-band DEM: {path}')
            tiles.append((Path(path).resolve(), src.crs, src.bounds, t,
                          src.width, src.height, src.nodata, src.units[0]))
    if not tiles:
        raise ValueError('No DEM tiles found')
    crs = tiles[0][1]
    if any(tile[1] != crs for tile in tiles):
        raise ValueError('DEM tiles must share a coordinate reference system')
    left = min(t[2].left for t in tiles)
    top = max(t[2].top for t in tiles)
    right = max(t[2].right for t in tiles)
    bottom = min(t[2].bottom for t in tiles)
    dx = min(t[3].a for t in tiles)
    dy = min(-t[3].e for t in tiles)
    root = ET.Element('VRTDataset', rasterXSize=str(math.ceil((right-left)/dx)),
                      rasterYSize=str(math.ceil((top-bottom)/dy)))
    ET.SubElement(root, 'SRS').text = crs.to_wkt()
    ET.SubElement(root, 'GeoTransform').text = f'{left}, {dx}, 0, {top}, 0, {-dy}'
    band = ET.SubElement(root, 'VRTRasterBand', dataType='Float32', band='1')
    ET.SubElement(band, 'NoDataValue').text = 'nan'
    if tiles[0][7]:
        ET.SubElement(band, 'UnitType').text = tiles[0][7]
    for path, _, bounds, _, width, height, nodata, _ in tiles:
        source = ET.SubElement(band, 'ComplexSource')
        ET.SubElement(source, 'SourceFilename', relativeToVRT='0').text = str(path)
        ET.SubElement(source, 'SourceBand').text = '1'
        ET.SubElement(source, 'SrcRect', xOff='0', yOff='0', xSize=str(width), ySize=str(height))
        ET.SubElement(source, 'DstRect', xOff=str((bounds.left-left)/dx),
                      yOff=str((top-bounds.top)/dy), xSize=str((bounds.right-bounds.left)/dx),
                      ySize=str((bounds.top-bounds.bottom)/dy))
        if nodata is not None:
            ET.SubElement(source, 'NODATA').text = str(nodata)
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)


def fetch_terrain(bbox, download_dir, output_dir, resolutions_km=(3, 22),
                  provider='creodias_s3', collection='COP_DEM_GLO30_DGED'):
    west, south, east, north = bbox
    if not all(math.isfinite(v) for v in bbox) or not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('Bounds must be WEST SOUTH EAST NORTH in WGS84')
    # Validate outputs before any potentially large downloads.
    resolutions_km = tuple(resolutions_km)
    if not resolutions_km or any(not math.isfinite(v) or v <= 0 for v in resolutions_km):
        raise ValueError('Resolutions must be finite positive distances in km')
    outputs = [Path(output_dir) / f'europe_terrain_{v:g}km.tif' for v in resolutions_km]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Duplicate output resolutions')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    try:
        from eodag import EODataAccessGateway
    except ImportError as exc:
        raise ValueError('Install download dependencies: pip install -r requirements-terrain.txt') from exc
    dag = EODataAccessGateway()
    print(f'Searching {collection} on {provider} for {bbox}...', flush=True)
    products = dag.search_all(provider=provider, collection=collection, geom=list(bbox))
    if not products:
        raise ValueError('No Copernicus DEM products found for these bounds')
    print(f'Downloading {len(products)} products...', flush=True)
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    paths = dag.download_all(products, output_dir=str(Path(download_dir).resolve()))
    if len(paths) != len(products) or any(not path for path in paths):
        raise ValueError('Some products failed to download; retry using the same download directory')
    tiles = set()
    for location in paths:
        path = Path(location)
        candidates = [path] if path.is_file() else path.rglob('*')
        found = {p.resolve() for p in candidates if p.suffix.lower() in {'.tif', '.tiff'} and 'DEM' in p.name.upper()}
        if not found:
            raise ValueError(f'No DEM GeoTIFF found in {path}; select a DGED collection')
        tiles.update(found)
    with TemporaryDirectory(prefix='terrain-mosaic-') as workspace:
        mosaic = Path(workspace) / 'mosaic.vrt'
        build_mosaic(sorted(tiles), mosaic)
        return generate_terrain(mosaic, output_dir, resolutions_km)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bbox', nargs=4, type=float, default=[-10, 35, 25, 65],
                        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'))
    parser.add_argument('--download-dir', type=Path, default=Path('data/copernicus_dem'))
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--resolutions-km', nargs='+', type=float, default=[3, 22])
    parser.add_argument('--provider', default='creodias_s3')
    parser.add_argument('--collection', choices=['COP_DEM_GLO30_DGED', 'COP_DEM_GLO90_DGED'],
                        default='COP_DEM_GLO30_DGED')
    args = parser.parse_args()
    try:
        outputs = fetch_terrain(args.bbox, args.download_dir, args.output_dir,
                                args.resolutions_km, args.provider, args.collection)
    except Exception as exc:
        parser.exit(1, f'Terrain download/generation failed: {exc}\n')
    for output in outputs:
        print(f'Generated {output}')


if __name__ == '__main__':
    main()
