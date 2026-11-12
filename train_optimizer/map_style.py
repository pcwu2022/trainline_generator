"""Shared map styling for the original optimizer and the fast generator."""
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

CITY_STYLE = dict(marker='+', color='#5d6370', markersize=7, zorder=3)
STATION_STYLE = dict(marker='o', markerfacecolor='white', markeredgecolor='#124625',
                     markeredgewidth=1, markersize=3.5, zorder=5)
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
    return dict(color=cmap(norm(satisfaction)),
                linewidth=width, linestyle='-', zorder=4,
                path_effects=[Railway()] if train_type == 'HSR' else [])


class Railway(AbstractPathEffect):
    """Colored track with white dashes, contained within the normal route width."""
    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        rail_gc = renderer.new_gc()
        rail_gc.copy_properties(gc)
        try:
            renderer.draw_path(rail_gc, tpath, affine, None)
            rail_gc.set_foreground('white')
            rail_gc.set_linewidth(gc.get_linewidth() * .65)
            rail_gc.set_dashes(0, (4, 4))
            rail_gc.set_capstyle('butt')
            renderer.draw_path(rail_gc, tpath, affine, None)
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
