"""Human-readable accounting for the currently published incumbent."""


def cost_breakdown_log(result):
    rows = ['Rail network cost breakdown (EUR)',
            f"Updated: {result['utc_time']} | phase: {result['phase']}",
            f"Solver status: {result.get('solver', {}).get('status', 'pending')}"]
    if not result.get('feasible'):
        return '\n'.join(rows + ['No feasible incumbent; costs are unavailable.']) + '\n'
    c, m = result['assumptions'], result['metrics']
    def amount(label, value):
        rows.append(f'{label}: EUR {value:,.2f}')

    rows += ['', 'INITIAL CAPITAL']
    for kind, label in [('track', 'Track construction'), ('station', 'Stations'), ('fleet', 'Train fleet')]:
        amount(label, m['lifecycle']['assets'][kind]['initial_capital'])
    amount('Total initial capital', m['capital_cost'])
    rows += ['', 'ANNUAL OPERATIONS']
    amount('Track maintenance', sum(t['annual_maintenance'] for t in result['tracks']))
    amount('Station operations', len(result['stations']) * c['station_annual_cost'])
    amount('Train distance costs', sum(l['annual_train_km'] for l in result['lines']) * c['train_cost_per_km'])
    amount('Train departure costs', sum(l['annual_departures'] for l in result['lines']) * c['train_cost_per_departure'])
    amount('Total annual operating cost', m['annual_operating_cost'])
    amount('Annual fare revenue', m['annual_revenue'])
    amount('Annual operating surplus', m['annual_revenue'] - m['annual_operating_cost'])
    rows += ['', f"UNDISCOUNTED {c['accounting_years']}-YEAR ACCOUNTING (initial capital + operations)"]
    amount('Revenue', m['horizon_revenue'])
    amount('Cost', m['horizon_cost'])
    amount('Profit', m['horizon_profit'])
    lc = m['lifecycle']
    rows += ['', f"LIFECYCLE PRESENT VALUE ({lc['years']} years, {lc['discount_rate']:.2%} discount rate)"]
    for label, key in [('Revenue', 'pv_revenue'), ('Operating cost', 'pv_operating'),
                       ('Asset renewals', 'pv_renewals'), ('Terminal residual credit', 'pv_residual'),
                       ('Total cost (initial + operating + renewals - residual)', 'pv_cost'),
                       ('Net present value', 'npv'), ('Equivalent annual net value', 'equivalent_annual_net_value')]:
        amount(label, lc[key])
    rows += ['', 'SOFT PROFITABILITY TARGET']
    amount('Required margin on lifecycle PV cost', c['financial_margin'] * lc['pv_cost'])
    amount('Minimum additional PV profit', c['minimum_horizon_profit'])
    amount('Shortfall against target', m['profit_shortfall'])
    rows.append(f"Profit target met: {m['profit_target_met']}")
    rows.append(f"Shortfall objective penalty (dimensionless, not a cash cost): {m['profit_shortfall_penalty']:.9g}")
    rows += ['', 'SELECTED ASSETS']
    for track in result['tracks']:
        amount(f"Track {track['id']} ({track['km']:.2f} km), capital", track['capital_cost'])
    for station in result['stations']:
        amount(f"Station {station['id']} ({station['name']}), capital", c['station_capital_cost'])
    for line in result['lines']:
        label = f"Line {line['id']} ({' -> '.join(line['stop_names'])}, {line['frequency_per_hour_per_direction']}/h/direction, {line['fleet']} trainsets)"
        amount(label + ', fleet capital', line['capital_cost'])
        amount(label + ', annual train operations', line['annual_operating_cost'])
    return '\n'.join(rows) + '\n'
