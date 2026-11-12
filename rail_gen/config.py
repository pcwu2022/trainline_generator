"""Single JSON parameter source, strict recursive overrides, and input validation."""
from copy import deepcopy
import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name('config.json')


def merge(base, overrides, prefix=''):
    if not isinstance(overrides, dict):
        raise ValueError('Configuration overrides must be an object')
    result = deepcopy(base)
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f'Unknown parameter: {prefix}{key}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{prefix}{key} must be an object')
            result[key] = merge(base[key], value, f'{prefix}{key}.')
        else:
            result[key] = value
    return result


def validate(c):
    schema = json.loads(DEFAULT_CONFIG.read_text())['defaults']
    integer_keys = {'prep.max_cities', 'prep.neighbors', 'prep.force_pairs',
        'operations.assignment_iterations', 'operations.maximum_train_units',
        'search.capacity_repair_steps', 'search.capacity_repair_candidates', 'search.population', 'search.generations',
        'search.seed', 'search.workers', 'search.representatives', 'search.stagnation_generations',
        'reporting.snapshot_every', 'reporting.gif_frame_ms',
        'services.max_through_routes', 'services.max_route_edges', 'services.plan_trials'}
    def check(values, defaults, prefix=''):
        for key, value in values.items():
            name = prefix + key
            expected = defaults[key]
            if isinstance(expected, dict):
                check(value, expected, name + '.')
            elif isinstance(expected, bool):
                if not isinstance(value, bool):
                    raise ValueError(f'{name} must be boolean')
            elif isinstance(expected, (float, int)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 and name not in ('demand.total_elasticity', 'demand.shift_elasticity')):
                    requirement = 'finite numeric' if name in ('demand.total_elasticity', 'demand.shift_elasticity') else 'finite and nonnegative'
                    raise ValueError(f'{name} must be {requirement}')
                if name in integer_keys and not isinstance(value, int):
                    raise ValueError(f'{name} must be an integer')
            elif not isinstance(value, type(expected)):
                raise ValueError(f'Invalid type for {name}')
    check(c, schema)
    d = c['demand']
    if d['model'] not in ('reference_elasticity', 'rail_share'):
        raise ValueError('demand.model must be reference_elasticity or rail_share')
    if not d['total_elasticity'] <= d['shift_elasticity'] < 0:
        raise ValueError('Demand elasticities require total <= shift < 0')
    if d['reference_speed_kmh'] <= 0 or d['reference_detour_ratio'] < 1:
        raise ValueError('Reference speed must be positive and detour ratio at least one')
    if c['infrastructure_layout'] not in ('single', 'double', 'four'):
        raise ValueError('infrastructure_layout must be single, double or four')
    if not 1 <= c['services']['plan_trials'] <= 3 or c['services']['max_route_edges'] < 1:
        raise ValueError('services requires 1..3 plan_trials and positive max_route_edges')
    for stock in c['rolling_stock'].values():
        if any(value <= 0 for value in stock.values()):
            raise ValueError('Rolling-stock parameters must be positive')
    if len(c['bbox']) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in c['bbox']):
        raise ValueError('bbox must contain four finite numbers')
    if any(not isinstance(country, str) or len(country) != 2 for country in c['countries']):
        raise ValueError('countries must contain ISO alpha-2 strings')
    w, s, e, n = c['bbox']
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError('bbox must be west, south, east, north in WGS84')
    if c['terrain_kind'] not in ('relief', 'dem', 'flat'):
        raise ValueError('terrain_kind must be relief, dem, or flat')
    if c['search']['algorithm'] not in ('incremental', 'nsga2', 'nsga3'):
        raise ValueError('search.algorithm must be incremental, nsga2 or nsga3')
    if c['search']['profit'] and c['search']['algorithm'] != 'incremental':
        raise ValueError('--profit requires the incremental algorithm')
    for section, keys in {
        'prep': ['max_cities', 'center_coefficient', 'access_speed_kmh', 'neighbors', 'max_edge_km'],
        'alignment': ['cell_km', 'window_factor'],
        'demand': ['a', 'beta', 'minimum_distance_km', 'maximum_distance_km', 'sigma_hours', 'value_of_time_eur', 'fare_reference_km'],
        'operations': ['maximum_train_units', 'hours_per_day', 'days_per_year', 'minimum_frequency', 'target_load_factor', 'maximum_load_factor', 'assignment_iterations', 'peak_factor'],
        'finance': ['track_life_years', 'station_life_years', 'fleet_life_years'],
        'search': ['capacity_repair_candidates', 'population', 'generations', 'workers', 'representatives', 'stagnation_generations'],
        'reporting': ['snapshot_every'],
    }.items():
        if any(c[section][key] <= 0 for key in keys):
            raise ValueError(f'{section}: required positive parameter is zero')
    for name, layout in c['track_layouts'].items():
        if layout['cost_multiplier'] <= 0:
            raise ValueError('Track layout cost multiplier must be positive')
        for frequency in layout['max_frequency'].values():
            if frequency < c['operations']['minimum_frequency']:
                raise ValueError('Track layout frequency is below minimum frequency')
    for t in c['tracks'].values():
        if any(value <= 0 for value in t.values()):
            raise ValueError('Track parameters must be positive')
        if t['max_frequency'] < c['operations']['minimum_frequency']:
            raise ValueError('Track frequency limit is below minimum frequency')
    for section, key in [('prep', 'uptake'), ('operations', 'target_load_factor'),
                         ('operations', 'additional_unit_cost_fraction'),
                         ('operations', 'maximum_load_factor'), ('constraints', 'maximum_spill_fraction'),
                         ('constraints', 'minimum_component_share'), ('search', 'greedy_fraction'),
                         ('search', 'mutation_rate'), ('search', 'structured_probability')]:
        if not 0 <= c[section][key] <= 1:
            raise ValueError(f'{section}.{key} must be between 0 and 1')
    if not 0 < c['search']['growth_fraction'] <= 1:
        raise ValueError('search.growth_fraction must be in (0, 1]')
    if not 0 < c['search']['retained_ridership'] <= 1:
        raise ValueError('search.retained_ridership must be in (0, 1]')
    duration = c['reporting']['gif_frame_ms']
    if not 10 <= duration <= 655350 or duration % 10:
        raise ValueError('reporting.gif_frame_ms must be a multiple of 10 between 10 and 655350')
    if c['operations']['target_load_factor'] > c['operations']['maximum_load_factor']:
        raise ValueError('Target load factor exceeds maximum')
    if c['operations']['initial_frequency'] <= 0:
        raise ValueError('Initial frequency must be positive')
    low = c['operations']['low_frequencies']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 < v < 1 for v in low) or low != sorted(set(low)):
        raise ValueError('operations.low_frequencies must be sorted unique finite frequencies between zero and one')
    if c['operations']['hours_per_day'] > 24 or c['operations']['days_per_year'] > 366:
        raise ValueError('Operating hours/days exceed a calendar year')
    if c['operations']['transfer_angle_degrees'] > 180:
        raise ValueError('Transfer angle must be at most 180 degrees')
    if c['prep']['via_detour_ratio'] < 1 or c['alignment']['window_factor'] < 1:
        raise ValueError('Via ratio and routing window factor must be at least one')
    if c['prep']['max_cities'] < 2 or c['search']['population'] < 4:
        raise ValueError('Need at least two cities and four population members')
    if c['terrain_kind'] != 'flat' and not c['terrain_file']:
        raise ValueError('A terrain file is required unless terrain_kind is flat')
    return c


def migrate_config(custom):
    """Move legacy rolling-stock keys out of tracks; explicit new keys win."""
    custom = deepcopy(custom)
    # Density-derived radii supersede fixed catchments in saved configs.
    custom.get('prep', {}).pop('catchment_km', None)
    for name, values in custom.get('tracks', {}).items():
        if name not in ('CR', 'HSR') or not isinstance(values, dict):
            continue
        stock = custom.setdefault('rolling_stock', {}).setdefault(name, {})
        for key in ('seats', 'train_meur', 'opex_eur_trainkm', 'yield_eur_pkm'):
            if key in values:
                stock.setdefault(key, values.pop(key))
        if 'speed_kmh' in values:
            stock.setdefault('speed_kmh', values['speed_kmh'])
        if not stock:
            del custom['rolling_stock'][name]
    if 'rolling_stock' in custom and not custom['rolling_stock']:
        del custom['rolling_stock']
    return custom


def load_config(path=None, profile='france', overrides=None):
    raw = json.loads(DEFAULT_CONFIG.read_text())
    c = raw['defaults']
    if path:
        custom = json.loads(Path(path).read_text())
        if 'defaults' in custom:
            if set(custom) - {'defaults', 'profiles'}:
                raise ValueError('Configuration document only accepts defaults and profiles')
            c = merge(c, migrate_config(custom['defaults']))
            raw['profiles'].update(custom.get('profiles', {}))
        else:
            c = merge(c, migrate_config(custom))
    if profile:
        if profile not in raw['profiles']:
            raise ValueError(f'Unknown profile: {profile}')
        c = merge(c, migrate_config(raw['profiles'][profile]))
    return validate(merge(c, migrate_config(overrides or {})))
