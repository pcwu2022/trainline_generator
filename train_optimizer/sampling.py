"""Reproducible bounded sampling without constructing a full pair list."""
import math


def sample_indices(size, limit, rng):
    if not limit or size <= limit:
        return range(size)
    # Floyd's algorithm takes O(limit) time and memory even for huge populations.
    selected = set()
    for end in range(size-limit, size):
        pick = int(rng.integers(end+1))
        selected.add(end if pick in selected else pick)
    return sorted(selected)


def sample_actions(actions, fraction, rng, weights=None):
    actions = list(dict.fromkeys(actions))
    count = max(1, math.ceil(len(actions)*fraction))
    if count >= len(actions):
        return actions
    probabilities = None
    if weights is not None:
        total = sum(weights[a] for a in actions)
        probabilities = [weights[a]/total for a in actions]
    return [actions[i] for i in rng.choice(len(actions), size=count, replace=False, p=probabilities)]


def sample_pairs(items, limit, rng):
    """Map sampled triangular indices to unordered pairs in O(log N) each."""
    n = len(items)
    for index in sample_indices(n*(n-1)//2, limit, rng):
        low, high = 0, n-1
        while low+1 < high:
            mid = (low+high)//2
            if mid*(2*n-mid-1)//2 <= index:
                low = mid
            else:
                high = mid
        offset = index-low*(2*n-low-1)//2
        yield items[low], items[low+1+offset]


# Relative priorities apply both to eligibility and the actual move selection.
PHASE_ACTION_WEIGHTS = {
    'exploration': dict(seed_region=12, frontier=8, bridge=2, frequency=1,
                        through=1, extend=4, move=1),
    'integration': dict(seed_region=1, frontier=4, bridge=12, frequency=3,
                        through=5, extend=3, move=2),
    'consolidation': dict(alignment=8, reuse=10, hub=6, frequency=5,
                          remove=4, move=6, junction_station=4, through=3),
}


def choose_action(actions, phase, rng):
    weights = PHASE_ACTION_WEIGHTS[phase]
    total = sum(weights[a] for a in actions)
    return str(rng.choice(actions, p=[weights[a]/total for a in actions]))


def pop_cost_candidate(pending, fraction, rng):
    """Keep untested moves queued; sampling must not imply exhaustion."""
    weights = PHASE_ACTION_WEIGHTS['consolidation']
    allowed = sample_actions([item[1] for item in pending], fraction, rng, weights)
    family = choose_action(allowed, 'consolidation', rng)
    indices = [i for i, item in enumerate(pending) if item[1] == family]
    # Candidates are sorted by estimated saving. Favor higher savings while
    # retaining a chance to test lower-ranked alternatives within this family.
    ranks = [1/(rank+1) for rank in range(len(indices))]
    total = sum(ranks)
    selected = int(rng.choice(len(indices), p=[w/total for w in ranks]))
    return pending.pop(indices[selected])
