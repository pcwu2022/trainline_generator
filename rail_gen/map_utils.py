"""Map projection, labels, framing and incremental MP4 output."""
from pathlib import Path
import numpy as np
from .projection import regional_projection
from PIL import Image

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


MAP_PIXELS = 1080
MAP_DPI = 90


def map_figure_size(viewport):
    """Fixed square export canvas; geographic extent only controls map framing."""
    side = MAP_PIXELS / MAP_DPI
    return side, side


class ProgressVideo:
    """Atomically append H.264 snapshot segments at a constant 30 fps.

    Stream-copy prior segments so history is neither decoded nor held in RAM.
    Each append leaves a playable MP4, including when a run is interrupted.
    """
    fps = 30

    def __init__(self, path, duration_ms=500):
        import imageio_ffmpeg
        if type(duration_ms) is not int or not 34 <= duration_ms <= 655350:
            raise ValueError('Snapshot duration must be between 34 and 655350 ms')
        self.path, self.duration_ms = Path(path), duration_ms
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.frames, self.size = 0, None
        self.repeat = max(1, round(duration_ms * self.fps / 1000))
        self.state_path = self.path.parent / 'progress.json'
        if self.path.exists():
            import json
            state = json.loads(self.state_path.read_text())
            self.frames, self.size = state['snapshots'], tuple(state['size'])

    def append(self, png_path):
        import json
        import subprocess
        import tempfile
        with Image.open(png_path) as source:
            width, height = self.size or source.size
        # yuv420p requires even dimensions; retain the square canvas.
        size = (width + width % 2, height + height % 2)
        with tempfile.TemporaryDirectory(prefix='.video-', dir=self.path.parent) as directory:
            directory = Path(directory)
            segment, merged = directory / 'segment.mp4', directory / 'merged.mp4'
            def run(arguments):
                subprocess.run([self.ffmpeg, '-v', 'error', '-y', *arguments],
                               check=True, capture_output=True)
            run(['-loop', '1', '-framerate', str(self.fps), '-i', str(png_path),
                 '-frames:v', str(self.repeat), '-vf', f'scale={size[0]}:{size[1]}',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                 '-pix_fmt', 'yuv420p', '-video_track_timescale', '15360',
                 '-movflags', '+faststart', str(segment)])
            if self.path.exists():
                # Temporary hard link avoids escaping arbitrary user output paths
                # in ffmpeg's concat-list syntax.
                import os
                os.link(self.path, directory / 'previous.mp4')
                listing = directory / 'concat.txt'
                listing.write_text("file 'previous.mp4'\nfile 'segment.mp4'\n")
                run(['-f', 'concat', '-safe', '0', '-i', str(listing),
                     '-c', 'copy', '-movflags', '+faststart', str(merged)])
            else:
                segment.rename(merged)
            merged.replace(self.path)
        self.frames += 1
        self.size = size
        temporary = self.state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(dict(snapshots=self.frames, size=size)))
        temporary.replace(self.state_path)
