"""Constant-real-price, end-of-year lifecycle cash flows."""


def annuity_factor(years, rate):
    return sum((1 + rate) ** -year for year in range(1, years + 1))


def asset_cashflows(capital, life, years):
    """Full like-for-like replacement; straight-line remaining-service residual.

    No replacement is purchased at the terminal year. Assets are fully used
    before replacement; residual is a scenario value, not guaranteed resale.
    """
    renewals = {year: capital for year in range(life, years, life)}
    last_purchase = max(renewals, default=0)
    residual = capital * max(0, life - (years - last_purchase)) / life
    return renewals, residual


def asset_pv_factor(life, years, rate):
    renewals, residual = asset_cashflows(1, life, years)
    return 1 + sum(value / (1 + rate)**year for year, value in renewals.items()) - residual / (1 + rate)**years


def lifecycle_report(capitals, revenue, operating, config, rate):
    years = config.appraisal_years
    annual = annuity_factor(years, rate)
    renewals = {}
    residual = 0
    assets = {}
    for kind, capital in capitals.items():
        schedule, remaining = asset_cashflows(capital, config.asset_lives_years[kind], years)
        assets[kind] = dict(initial_capital=capital, life_years=config.asset_lives_years[kind],
                            renewals_by_year=schedule, terminal_residual=remaining)
        residual += remaining
        for year, amount in schedule.items():
            renewals[year] = renewals.get(year, 0) + amount
    capital = sum(capitals.values())
    pv_renewals = sum(value / (1 + rate)**year for year, value in renewals.items())
    pv_residual = residual / (1 + rate)**years
    pv_cost = capital + operating * annual + pv_renewals - pv_residual
    pv_revenue = revenue * annual
    return dict(years=years, discount_rate=rate, annuity_factor=annual, assets=assets,
                initial_capital=capital, pv_revenue=pv_revenue, pv_operating=operating * annual,
                pv_renewals=pv_renewals, terminal_residual=residual, pv_residual=pv_residual,
                pv_cost=pv_cost, npv=pv_revenue-pv_cost,
                equivalent_annual_net_value=(pv_revenue-pv_cost)/annual,
                cashflows=[dict(year=0, revenue=0, operating=0, renewals=0, residual=0, net=-capital)] +
                    [dict(year=y, revenue=revenue, operating=operating, renewals=renewals.get(y, 0),
                          residual=residual if y == years else 0,
                          net=revenue-operating-renewals.get(y, 0)+(residual if y == years else 0))
                     for y in range(1, years+1)])
