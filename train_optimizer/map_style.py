"""Shared map styling for the original optimizer and the fast generator."""
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

CITY_STYLE = dict(marker='+', color='#5d6370', markersize=7, zorder=3)
STATION_STYLE = dict(marker='o', markerfacecolor='white', markeredgecolor='#124625',
                     markeredgewidth=2, markersize=7, zorder=5)
CATCHMENT_STYLE = dict(fill=False, color='#217c55', linewidth=1,
                       linestyle='--', alpha=0.6, zorder=3)


def population_collection(polygons, densities):
    values = np.maximum(.1, np.asarray(densities, dtype=float))
    return PolyCollection(polygons, array=values, cmap='YlOrRd',
        norm=LogNorm(vmin=1, vmax=max(10, float(values.max()) if len(values) else 10)),
        edgecolors='#ffffff', linewidths=.2, alpha=.65)


def satisfaction_scale():
    return (LinearSegmentedColormap.from_list('rail_satisfaction', ['#b2f2a5', '#124625']),
            Normalize(0, 1))


def track_style(frequency, train_type, satisfaction):
    cmap, norm = satisfaction_scale()
    width = (1.7 + frequency * .6) * .25
    if train_type == 'HSR':
        width *= 3
    return dict(color=cmap(norm(satisfaction)), linewidth=width,
                linestyle='-', zorder=4,
                path_effects=[DoubleRail()] if train_type == 'HSR' else [])


class DoubleRail(AbstractPathEffect):
    """Draw two rails separated in display points, with a transparent center.

    The complete pair spans the configured HSR width (three times CR at the
    same frequency). Offsets apply at draw time, so zoom/aspect never change
    the symbol width and the basemap stays visible between the rails.
    """
    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        path = affine.transform_path(tpath)
        vertices = path.vertices
        if len(vertices) < 2:
            return
        delta = np.diff(vertices, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        normals = np.divide(np.column_stack((-delta[:, 1], delta[:, 0])),
                            lengths[:, None], out=np.zeros_like(delta),
                            where=lengths[:, None] > 0)
        offsets = np.vstack((normals[0], normals[:-1] + normals[1:], normals[-1]))
        # Miter joins align the pair at bends; cap extreme hairpin miters.
        denominator = np.einsum('ij,ij->i', offsets, np.vstack((normals, normals[-1])))
        offsets = np.divide(offsets, denominator[:, None], out=np.zeros_like(offsets),
                            where=np.abs(denominator[:, None]) > 1e-8)
        magnitude = np.linalg.norm(offsets, axis=1)
        offsets *= np.minimum(1, np.divide(2, magnitude, out=np.ones_like(magnitude), where=magnitude > 0))[:, None]
        offsets *= renderer.points_to_pixels(gc.get_linewidth()) * .35
        rail_gc = renderer.new_gc()
        rail_gc.copy_properties(gc)
        rail_gc.set_linewidth(gc.get_linewidth() * .30)
        try:
            for sign in (-1, 1):
                renderer.draw_path(rail_gc, Path(vertices + sign * offsets, path.codes),
                                   IdentityTransform(), None)
        finally:
            rail_gc.restore()


def add_track_reference(figure):
    """Use the actual route styles for a fixed reference below the map."""
    handles = [Line2D([], [], label=f'CR {frequency:g}',
                      **track_style(frequency, 'CR', 1)) for frequency in (1, 2, 4, 8)]
    handles.append(Line2D([], [], label='HSR 4', **track_style(4, 'HSR', 1)))
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.45, .047),
                  ncol=5, frameon=False, fontsize=8, handlelength=3,
                  columnspacing=1.3, title='Frequency (trains/hour/direction)',
                  title_fontsize=8)
