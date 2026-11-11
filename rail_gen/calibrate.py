"""Fit only user-supplied observed corridors; never manufacture backtest evidence."""
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from .demand import fare, rail_share
from .evaluator import capital_recovery_factor

REQUIRED = ('corridor', 'population_a', 'population_b', 'kappa_a', 'kappa_b',
            'distance_km', 'line_length_km', 'speed_kmh', 'frequency',
            'access_a_hours', 'access_b_hours', 'yield_eur_pkm',
            'observed_annual_passengers', 'split', 'source')


def read_corridors(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in REQUIRED):
            raise ValueError('Backtest CSV requires: ' + ', '.join(REQUIRED))
        rows = []
        for r in reader:
            if not r['source'].strip() or r['split'] not in ('train', 'test'):
                raise ValueError('Every corridor requires a source and train/test split')
            for key in REQUIRED:
                if key not in ('corridor', 'split', 'source'):
                    r[key] = float(r[key])
                    if not math.isfinite(r[key]) or r[key] < 0:
                        raise ValueError(f'Invalid corridor value: {key}')
            for key in ('population_a', 'population_b', 'distance_km', 'line_length_km',
                        'speed_kmh', 'frequency', 'observed_annual_passengers', 'kappa_a', 'kappa_b'):
                if r[key] <= 0:
                    raise ValueError(f'Corridor {r["corridor"]}: {key} must be positive')
            if r['kappa_a'] > 1 or r['kappa_b'] > 1 or r['line_length_km'] < .99 * r['distance_km']:
                raise ValueError('Invalid corridor capture fraction or line length')
            rows.append(r)
    if len({r['corridor'] for r in rows}) != len(rows):
        raise ValueError('Backtest corridor names must be unique')
    return rows


def predict_corridors(rows, parameters, dwell_hours=.05):
    result = []
    for r in rows:
        distance = max(parameters['minimum_distance_km'], r['distance_km'])
        base = parameters['k'] * (r['population_a'] * r['kappa_a']) ** parameters['a'] * (r['population_b'] * r['kappa_b']) ** parameters['a'] / distance ** parameters['beta']
        price = float(fare(r['distance_km'], r['yield_eur_pkm'], parameters))
        gt = r['access_a_hours'] + r['access_b_hours'] + min(1, .5 / r['frequency']) + r['line_length_km'] / r['speed_kmh'] + dwell_hours + price / parameters['value_of_time_eur']
        predicted = 2 * base * float(rail_share(gt, parameters))
        result.append(predicted if r['distance_km'] <= parameters['maximum_distance_km'] else 0.0)
    return np.array(result)


def fit(rows, config, fit_parameters=('k',)):
    parameters = config['demand'].copy()
    training = [r for r in rows if r['split'] == 'train']
    if len(training) < max(2, len(fit_parameters) + 1):
        raise ValueError('Not enough training corridors for demand fitting')
    if any(key not in ('k', 'a', 'beta', 't0_hours') for key in fit_parameters):
        raise ValueError('Supported fitted parameters: k, a, beta, t0_hours')
    if any(r['distance_km'] > parameters['maximum_distance_km'] for r in training):
        raise ValueError('Training corridor exceeds demand maximum_distance_km')
    bounds = {'k': (1e-12, 1e6), 'a': (.5, 1.2), 'beta': (.5, 3), 't0_hours': (.5, 8)}
    # Fit log(k) for numerical stability across population conventions.
    def decode(x):
        fitted = parameters.copy()
        for key, value in zip(fit_parameters, x):
            fitted[key] = float(np.exp(value) if key == 'k' else value)
        return fitted
    initial = [math.log(max(1e-12, parameters[k])) if k == 'k' else parameters[k] for k in fit_parameters]
    lower = [math.log(bounds[k][0]) if k == 'k' else bounds[k][0] for k in fit_parameters]
    upper = [math.log(bounds[k][1]) if k == 'k' else bounds[k][1] for k in fit_parameters]
    observed = np.array([r['observed_annual_passengers'] for r in training])
    def residual(x):
        predicted = predict_corridors(training, decode(x), config['operations']['dwell_hours'])
        return np.log(np.maximum(predicted, 1e-300)) - np.log(observed)
    solution = least_squares(residual, initial, bounds=(lower, upper))
    if not solution.success:
        raise ValueError(f'Demand calibration failed: {solution.message}')
    return decode(solution.x)


def backtest(rows, config, parameters):
    predictions = predict_corridors(rows, parameters, config['operations']['dwell_hours'])
    observed = np.array([r['observed_annual_passengers'] for r in rows])
    log_error = np.abs(np.log(np.maximum(predictions, 1e-300) / observed))
    held_out = np.array([r['split'] == 'test' for r in rows])
    anchors = np.array([str(r.get('anchor', '')).lower() in ('1', 'true', 'yes') for r in rows])
    strong = np.array([r.get('class') == 'strong' for r in rows])
    thin = np.array([r.get('class') == 'thin' for r in rows])
    checks = {'sample_size': dict(status='pass' if len(rows) >= 10 else 'not_assessed',
        corridors=len(rows), required=10)}
    if held_out.sum() >= 3:
        rho = float(spearmanr(predictions[held_out], observed[held_out]).statistic)
        checks['ranking'] = dict(status='pass' if np.isfinite(rho) and rho >= .8 else 'fail',
                                 spearman=rho if np.isfinite(rho) else None)
    else:
        checks['ranking'] = dict(status='not_assessed', reason='At least three held-out corridors required')
    if anchors.any():
        median = float(np.median(log_error[anchors]))
        worst = float(np.max(log_error[anchors]))
        checks['anchor_magnitude'] = dict(status='pass' if median <= math.log(2) and worst <= math.log(3) else 'fail',
            median_absolute_log_error=median, worst_factor=float(np.exp(min(700, worst))))
    else:
        checks['anchor_magnitude'] = dict(status='not_assessed', reason='Mark anchor corridors in CSV')
    if strong.any() and thin.any():
        ratio = float(predictions[strong].min() / max(1e-300, predictions[thin].max()))
        checks['strong_vs_thin'] = dict(status='pass' if ratio >= 10 else 'fail', ratio=ratio)
    else:
        checks['strong_vs_thin'] = dict(status='not_assessed', reason='Mark strong and thin corridor classes')
    o, f = config['operations'], config['finance']
    operations = []
    for r, predicted in zip(rows, predictions):
        track = config['tracks'][r.get('track_type') or 'HSR']
        avg_hourly = predicted / 2 / (o['hours_per_day'] * o['days_per_year'])
        frequency = max(o['minimum_frequency'], math.ceil(o['peak_factor'] * avg_hourly / (track['seats'] * o['target_load_factor'])))
        fleet = math.ceil(math.ceil(2 * frequency * (r['line_length_km'] / r['speed_kmh'] + o['dwell_hours']) * (1 + o['turnaround_overhead'])) * (1 + o['spare_ratio']))
        price = float(fare(r['distance_km'], r['yield_eur_pkm'], parameters))
        train_opex = 2 * frequency * r['line_length_km'] * o['hours_per_day'] * o['days_per_year'] * track['opex_eur_trainkm']
        operations.append(dict(corridor=r['corridor'], required_frequency=frequency, fleet=fleet,
            revenue_eur=float(predicted * price), train_opex_eur=train_opex,
            opex_recovery=float(predicted * price / train_opex)))
    strong_operations = [op for op, selected in zip(operations, strong) if selected]
    if strong_operations:
        checks['operations'] = dict(status='pass' if all(1 <= op['required_frequency'] <= 4 and 10 <= op['fleet'] < 100 for op in strong_operations) else 'fail')
        checks['operating_recovery'] = dict(status='pass' if all(op['opex_recovery'] >= .8 for op in strong_operations) else 'fail')
    else:
        checks['operations'] = dict(status='not_assessed', reason='No strong corridors')
        checks['operating_recovery'] = dict(status='not_assessed', reason='No strong corridors')
    crf = capital_recovery_factor(f['discount_rate'], f['track_life_years'])
    checks['capex_arithmetic'] = dict(status='pass' if .04 <= crf <= .06 else 'fail', track_crf=crf)
    # Network-level rejection and route loss checks are reported by report.py.
    statuses = [check['status'] for check in checks.values()]
    status = 'failed' if 'fail' in statuses else 'incomplete' if 'not_assessed' in statuses else 'passed'
    return dict(status=status, parameters=parameters, checks=checks, operations=operations,
        corridors=[dict(corridor=r['corridor'], split=r['split'], source=r['source'],
                       observed=float(obs), predicted=float(pred), absolute_log_error=float(error))
                   for r, obs, pred, error in zip(rows, observed, predictions, log_error)])


def calibrate(path, config, fit_parameters=('k',)):
    rows = read_corridors(path)
    parameters = fit(rows, config, fit_parameters)
    report = backtest(rows, config, parameters)
    report['source_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report['fitted_parameters'] = list(fit_parameters)
    return parameters, report
