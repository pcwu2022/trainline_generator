"""Map projection, labels, framing and incremental GIF output."""
import math
import shutil
from pathlib import Path
import numpy as np
from .projection import regional_projection
from PIL import GifImagePlugin, Image

def city_label_indices(cities):
    """Label the largest quarter of cities (rounded down), capped at fifty."""
    return set(sorted(range(len(cities)),
                      key=lambda i: -float(cities[i].get('population', cities[i].get('pop', 0))))[:min(50, len(cities) // 4)])


def map_projector(config):
    transformer = regional_projection({'bbox': config.bbox}).to_metric
    def project(coordinates):
        array = np.asarray(coordinates, dtype=float)
        if len(array) == 1:
            x, y = transformer.transform(float(array[0, 0]), float(array[0, 1]))
            return np.array([[x/1000, y/1000]])
        x, y = transformer.transform(array[:, 0], array[:, 1])
        return np.column_stack((x, y))/1000
    return project


def map_figure_size(viewport):
    """Square canvas growing with sqrt of the longest projected span in km.

    A 1,000 km span uses 10 inches; an 8-inch floor keeps labels legible.
    Use the fixed viewport so solution changes cannot resize animation frames.
    """
    xmin, xmax, ymin, ymax = viewport
    side = max(8.0, 10.0 * math.sqrt(max(xmax-xmin, ymax-ymin) / 1000.0))
    return side, side


class ProgressGif:
    """Append full frames using Pillow's documented GIF block encoder.

    Copy compressed bytes from the existing GIF, replace its trailer with the
    new frame and trailer, and atomically replace the file. Only the new image
    and small copy buffers are held in memory, not all historical frames. Local
    palettes preserve each frame's colors. No individual frame files are kept.
    """
    def __init__(self, path, duration_ms=500):
        if type(duration_ms) is not int or not 10 <= duration_ms <= 655350 or duration_ms % 10:
            raise ValueError('GIF frame duration must be a multiple of 10 ms between 10 and 655350')
        self.path, self.duration_ms = Path(path), duration_ms
        self.frames, self.size = 0, None
        if self.path.exists():
            with Image.open(self.path) as previous:
                if previous.format != 'GIF':
                    raise ValueError('progress.gif must be a valid GIF')
                self.frames, self.size = previous.n_frames, previous.size

    def append(self, png_path):
        with Image.open(png_path) as source:
            rgb = source.convert('RGB')
        if self.size and rgb.size != self.size:
            rgb = rgb.resize(self.size, Image.Resampling.LANCZOS)
        frame = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        temporary = self.path.with_name(self.path.name+'.tmp')
        try:
            with temporary.open('wb') as stream:
                if self.path.exists():
                    with self.path.open('rb') as previous:
                        previous.seek(-1, 2)
                        if previous.read(1) != b';':
                            raise ValueError('Cannot append to an incomplete progress GIF')
                        previous.seek(0)
                        shutil.copyfileobj(previous, stream, length=1024*1024)
                    stream.seek(-1, 2)  # Overwrite the previous GIF trailer.
                else:
                    header, _ = GifImagePlugin.getheader(frame, info=dict(loop=0, optimize=False))
                    for block in header:
                        stream.write(block)
                for block in GifImagePlugin.getdata(frame, duration=self.duration_ms, disposal=2,
                                                   include_color_table=True):
                    stream.write(block)
                stream.write(b';')
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
            frame.close()
            rgb.close()
        self.frames += 1
        self.size = self.size or frame.size

