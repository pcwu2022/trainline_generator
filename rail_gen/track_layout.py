"""Infrastructure types and independent track counts (never train types)."""
import numpy as np

STATE_TRACK = (None, 'CR', 'HSR')
LAYOUT_BY_COUNT = {1: 'single', 2: 'double', 4: 'four'}
MODEL_VERSION = 'network-elasticity-v2'


def track_type(state):
    return STATE_TRACK[int(state)]


def migrate_legacy_genome(genome):
    """Explicit conversion of the short-lived 3/4/5 layout-state encoding."""
    raw = np.asarray(genome)
    if raw.ndim != 1 or not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all() or np.any((raw < 0) | (raw > 5)):
        raise ValueError('Legacy genome must contain integers in 0..5')
    x = raw.astype(int)
    return np.choose(x, [0, 1, 2, 1, 1, 2]), np.choose(x, [0, 2, 2, 1, 4, 4])
