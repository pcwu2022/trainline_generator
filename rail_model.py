"""Small, dependency-free building blocks for the rail planning model.

These evaluate assumptions and a fixed all-stop line; they do not optimize a network.
Distances are km, density people/km², demand directional trips/hour, money one
consistent user-chosen currency. Finance uses representative operating hours/year.
"""

from dataclasses import dataclass
import math


def nonnegative(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def center_radius(density, coefficient=0.1):
    """User's city-search radius; coefficient has units km²/sqrt(person)."""
    nonnegative(density=density, coefficient=coefficient)
    return coefficient * math.sqrt(density)


def uptake(radius, p_max=0.6, a=0.1, alpha=1.0):
    """Aggregate uptake for a whole catchment, not individual distance decay."""
    nonnegative(radius=radius, p_max=p_max, a=a, alpha=alpha)
    if p_max > 1 or alpha == 0:
        raise ValueError("p_max must be <= 1 and alpha must be > 0")
    return p_max * -math.expm1(-a * radius ** alpha)


def gravity(pop_a, pop_b, distance, scale, distance_floor=1.0, beta=2.0):
    """Unconstrained directional OD trips/hour; scale MUST be calibrated."""
    nonnegative(pop_a=pop_a, pop_b=pop_b, distance=distance, scale=scale,
                distance_floor=distance_floor, beta=beta)
    if distance_floor == 0 or beta == 0:
        raise ValueError("distance_floor and beta must be > 0")
    return scale * pop_a * pop_b / max(distance, distance_floor) ** beta


def urban_cost_per_km(density, base_cost, urban_coefficient):
    nonnegative(density=density, base_cost=base_cost,
                urban_coefficient=urban_coefficient)
    return base_cost + urban_coefficient * density


def capital_recovery_factor(rate, years):
    nonnegative(rate=rate, years=years)
    if years == 0:
        raise ValueError("years must be > 0")
    if rate == 0:
        return 1 / years
    return rate / -math.expm1(-years * math.log1p(rate))


def assign_catchments(cells, stations, distance_fn):
    """Assign each cell once to the largest covering station.

    Cells: dictionaries with id, density, population and any location fields.
    Stations: id, radius_km, size (fixed exogenous rank), and location fields.
    distance_fn(cell, station) must return km. Ties break by station id.
    Whole-cell allocation is deliberately coarse: use polygon fractions later.
    """
    if len({s['id'] for s in stations}) != len(stations):
        raise ValueError("station IDs must be unique")
    result = {s['id']: 0.0 for s in stations}
    assignment = {}
    for cell in cells:
        if cell['id'] in assignment:
            raise ValueError("cell IDs must be unique")
        nonnegative(population=cell['population'])
        covering = []
        for station in stations:
            nonnegative(radius=station['radius_km'], size=station['size'])
            distance = distance_fn(cell, station)
            nonnegative(distance=distance)
            if distance <= station['radius_km']:
                covering.append(station)
        winner = min(covering, key=lambda s: (-s['size'], s['id'])) if covering else None
        assignment[cell['id']] = winner['id'] if winner else None
        if winner:
            result[winner['id']] += cell['population']
    return assignment, result


@dataclass(frozen=True)
class Line:
    segment_km: tuple
    frequency_per_hour: int = 1  # Each direction; evenly spaced services.
    seats: int = 500
    speed_kmh: float = 100.0
    dwell_minutes: float = 2.0


def evaluate_line(line, demand, *, ticket_per_km, capital_cost,
                  operating_hours_per_year=5840, train_cost_per_km=15.0,
                  train_cost_per_departure=0.0, maintenance_fraction=0.02,
                  station_operating_cost=0.0, discount_rate=0.0,
                  accounting_years=10, max_headway_hours=1.0, budget_margin=0.05):
    """Evaluate ONE fixed, bidirectional all-stop line serving ALL supplied OD.

    demand maps (origin index, destination index) to directional trips/hour.
    This reports infeasibility rather than silently dropping excess passengers.
    Hourly demand is assumed identical in every operating hour, only for a demo.
    Passenger time includes mean initial wait + in-train time; no access/transfer.
    station_operating_cost is the annual cost for ALL stations on the line.
    Default finance is ten-year undiscounted payback, including upfront capital.
    An optional discount rate converts this to a ten-year NPV check. The annual
    equivalent check is algebraically identical for constant yearly cash flows.
    """
    if not line.segment_km:
        raise ValueError("at least one segment is required")
    for length in line.segment_km:
        nonnegative(segment_km=length)
        if length == 0:
            raise ValueError("segment lengths must be > 0")
    if (isinstance(line.frequency_per_hour, bool)
            or not isinstance(line.frequency_per_hour, int)
            or line.frequency_per_hour < 1):
        raise ValueError("frequency must be a positive integer")
    if not isinstance(line.seats, int) or line.seats < 1:
        raise ValueError("seats must be a positive integer")
    nonnegative(speed=line.speed_kmh, dwell=line.dwell_minutes,
                ticket=ticket_per_km, capital=capital_cost,
                operating_hours=operating_hours_per_year,
                train_cost=train_cost_per_km, departure_cost=train_cost_per_departure,
                maintenance=maintenance_fraction, station_cost=station_operating_cost,
                max_headway=max_headway_hours, budget_margin=budget_margin)
    if line.speed_kmh == 0 or max_headway_hours == 0 or operating_hours_per_year == 0:
        raise ValueError("speed, max_headway and operating_hours must be > 0")
    n = len(line.segment_km) + 1
    forward = [0.0] * (n - 1)
    reverse = [0.0] * (n - 1)
    passenger_km = passenger_hours = trips = 0.0
    for (i, j), volume in demand.items():
        if not (isinstance(i, int) and isinstance(j, int) and 0 <= i < n and 0 <= j < n and i != j):
            raise ValueError("OD endpoints must be distinct valid station indices")
        nonnegative(volume=volume)
        lo, hi = sorted((i, j))
        distance = sum(line.segment_km[lo:hi])
        loads = forward if i < j else reverse
        for segment in range(lo, hi):
            loads[segment] += volume
        passenger_km += volume * distance
        passenger_hours += volume * (distance / line.speed_kmh
                                     + (hi - lo - 1) * line.dwell_minutes / 60
                                     + 0.5 / line.frequency_per_hour)
        trips += volume
    capacity = line.seats * line.frequency_per_hour
    annual_revenue = passenger_km * ticket_per_km * operating_hours_per_year
    recovery = capital_recovery_factor(discount_rate, accounting_years)
    annual_capital = capital_cost * recovery
    annual_operating = (2 * line.frequency_per_hour * operating_hours_per_year
                        * (sum(line.segment_km) * train_cost_per_km + train_cost_per_departure)
                        + maintenance_fraction * capital_cost + station_operating_cost)
    annual_cost = annual_capital + annual_operating
    capacity_ok = max(forward + reverse) <= capacity
    headway_ok = 1 / line.frequency_per_hour <= max_headway_hours
    financial_ok = (1 + budget_margin) * annual_cost <= annual_revenue
    return dict(trips_per_hour=trips, passenger_km_per_hour=passenger_km,
                passenger_hours_per_hour=passenger_hours,
                forward_loads=forward, reverse_loads=reverse,
                capacity_per_direction_per_hour=capacity,
                capacity_ok=capacity_ok, headway_ok=headway_ok,
                financial_ok=financial_ok, feasible=capacity_ok and headway_ok and financial_ok,
                annual_revenue=annual_revenue, annual_cost=annual_cost,
                annual_operating_cost=annual_operating,
                annual_equivalent_profit=annual_revenue - annual_cost,
                accounting_years=accounting_years,
                horizon_revenue=annual_revenue * accounting_years,
                horizon_cost=capital_cost + annual_operating * accounting_years,
                horizon_profit=(annual_revenue - annual_operating) * accounting_years - capital_cost,
                net_present_value=(annual_revenue - annual_operating) / recovery - capital_cost,
                simple_payback_years=(capital_cost / (annual_revenue - annual_operating)
                                      if annual_revenue > annual_operating else None),
                revenue_is_potential_if_capacity_fails=not capacity_ok)
