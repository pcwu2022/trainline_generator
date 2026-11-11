"""Shared map styling for the original optimizer and the fast generator."""
import numpy as np
from matplotlib.collections import PolyCollection
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
    return (LinearSegmentedColormap.from_list('rail_satisfaction', ['#b2f2a5', '#00cc33']),
            Normalize(0, 1))


def track_style(frequency, train_type, satisfaction):
    cmap, norm = satisfaction_scale()
    return dict(color=cmap(norm(satisfaction)), linewidth=1.7 + frequency * .6,
                linestyle='-' if train_type == 'HSR' else '--', zorder=4)
