"""Validated, reproducible model assumptions (km, hours, people, currency)."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import math


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.json'


def read_configuration(path=DEFAULT_CONFIG_PATH):
    """Read the canonical defaults and named profiles, or a flat override file."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a JSON object')
    if 'defaults' not in data:
        return data, {}
    if set(data) - {'defaults', 'profiles'}:
        raise ValueError('Configuration document accepts only defaults and profiles')
    defaults, profiles = data['defaults'], data.get('profiles', {})
    if not isinstance(defaults, dict) or not isinstance(profiles, dict):
        raise ValueError('defaults and profiles must be JSON objects')
    if any(not isinstance(v, dict) for v in profiles.values()):
        raise ValueError('Each profile must be a JSON object of overrides')
    return defaults, profiles


def configuration_dataclass(cls):
    """Attach independent defaults from the single editable configuration file."""
    defaults, _ = read_configuration()
    if set(defaults) != set(cls.__annotations__):
        missing = set(cls.__annotations__) - set(defaults)
        extra = set(defaults) - set(cls.__annotations__)
        raise ValueError(f'config.json schema mismatch: missing={sorted(missing)}, unknown={sorted(extra)}')
    for name, value in defaults.items():
        setattr(cls, name, field(default_factory=lambda value=value: deepcopy(value)))
    return dataclass(cls)


@configuration_dataclass
class Config:
    gravity_heuristic: bool
    gravity_heuristic_demand_threshold: float
    speed_up: bool
    # Input files and study region
    # Choose source datasets and the bounding box [west, south, east, north].
    # Population cells are filtered by bbox and country polygons; cities supply labels only.
    population_file: str
    cities_file: str
    country_boundaries_file: str
    terrain_dem_file: str # Optional elevation DEM in metres; overrides land difficulty, preserves relief water
    terrain_file: str
    land_file: str
    bbox: list
    countries: list

    # Population and city selection
    # Greedy marginal coverage from eligible population cells.
    # Legacy city setting names now apply to population-derived markets.
    minimum_seed_spacing_km: float # Zero uses market_radius_km, independent of adaptive radius
    max_cities: int
    market_radius_km: float # Seed scoring neighborhood radius; does not limit catchments
    minimum_city_population: int # Minimum eligible population for seed selection
    density_threshold: float
    population_subdivision: int

    # Station locations and catchments
    # Generate station positions and catchment radii (km) around each city.
    # Access normalization and tolerance control the station-location calculation.
    center_method: str
    center_coefficient: float # radius_km = coefficient * sqrt(seed density in people/km²)
    locations_per_city: int
    station_offset_km: float
    access_normalization_floor_km: float
    access_optimizer_tolerance_km: float

    # Passenger demand and uptake
    # Estimate directional trips from population and distance (km), then apply
    # catchment uptake. Production and gravity modes use different demand scales.
    demand_model: str
    trips_per_person_hour: float
    gravity_scale: float
    gravity_beta: float
    distance_floor_km: float
    maximum_trip_distance_km: float
    p_max: float
    uptake_a: float
    uptake_alpha: float
    uptake_reference_radius_km: float # Zero uses catchment radius; positive decouples uptake from market sizing

    # Track routing and terrain
    # Control candidate corridor connections, routing resolution and connectivity.
    # Terrain ranges are millions of euros/km; position 0/0.5/1 selects low/mid/high costs.
    neighbors_per_city: int
    maximum_edge_km: float
    require_connected_network: bool
    routing_sample_km: float
    terrain_search_limit: int
    water_track_cost_per_km: float # Short-crossing baseline; EUR/km
    water_cost_growth_km: float # Exponential e-folding length of a continuous crossing
    water_routing_sample_km: float # Narrow rivers below this can still be missed
    terrain_cost_position: float
    terrain_cost_ranges_million: dict

    # Train services and passenger journeys
    # Define candidate lines, departures/hour/direction, train capacity and speed
    # (km/h). Stop/transfer times are minutes; the OD-average journey limit is hours.
    maximum_line_stops: int
    maximum_extra_lines: int
    maximum_shared_track_lines: int # Additional three-stop hub candidates, independent of the generic line cap.
    maximum_shared_track_detour: float # Maximum length / direct endpoint distance.
    maximum_alignment_variants_per_line: int # Fastest, cheapest and mixed physical alignments.
    frequencies_per_hour: list
    track_trains_per_hour_per_direction: int
    train_seats: int # CR seats per train; HSR carries twice this number.
    train_speed_kmh: float
    train_types: list
    hsr_capex_ratio: float
    hsr_opex_ratio: float
    hsr_speed_ratio: float
    hsr_fare_ratio: float
    dwell_minutes: float
    turnaround_minutes: float
    access_speed_kmh: float
    transfer_minutes: float
    maximum_journey_detour_ratio: float # Zero disables OD-average rail-time detour cap
    maximum_journey_hours: float

    # Operating calendar
    # Set daily departure hours and operating days/year for annual service,
    # revenue and operating-cost calculations.
    operating_start_hour: int
    operating_end_hour: int
    operating_days_per_year: int

    # Construction and operating costs
    # Set capital and recurring costs in euros. Track costs are per km; the urban
    # coefficient multiplies density above the threshold (people/km²).
    # Maintenance is a yearly fraction of track capital.
    base_track_cost_per_km: float
    urban_cost_coefficient: float
    urban_density_threshold: float
    station_capital_cost: float
    station_annual_cost: float
    train_capital_cost: float
    train_cost_per_km: float
    train_cost_per_departure: float
    maintenance_fraction: float

    # Revenue and financial targets
    # Set fares (euros/passenger-km), accounting years and the soft profit target.
    # The financial margin and minimum profit shape the target; the upfront budget is hard.
    ticket_per_km: float
    accounting_years: int # Undiscounted cash-flow report only.
    appraisal_years: int
    discount_rate: float # Real rate; costs and revenue are constant real prices.
    discount_rates: list
    asset_lives_years: dict
    financial_margin: float
    minimum_horizon_profit: float
    upfront_budget: float # Zero means no separate capital ceiling.
    maximum_annual_subsidy: float # Zero disables the hard lifecycle annual-equivalent subsidy ceiling.

    # Minimum service requirements
    # Require minimum satisfied shares of fixed rail-eligible needs, globally and per
    # origin. The global target must be positive; zero disables the per-origin target.
    minimum_served_fraction: float # Minimum completed trips / fixed rail-eligible needs.
    minimum_origin_served_fraction: float # Fraction of fixed rail-eligible origin needs.
    minimum_city_served_fraction: float # Fraction of candidate markets with operating service.
    minimum_selected_origin_served_fraction: float # Fixed rail-eligible origin need share, conditional on a station.
    minimum_service_frequency: int # Minimum frequency of each selected line, per direction.

    # Objective weights
    # Balance ridership benefits against journey time, station access, costs and
    # profit shortfall. Larger weights give a term more influence; zero disables its term.
    population_weight: float # Reward the share of fixed market population within selected catchments.
    served_weight: float # Reward completed passenger trips; transfers do not count as extra trips.
    time_weight: float # Passenger-hours, including access, waits and transfers.
    center_weight: float
    access_weight: float
    cost_weight: float
    profit_shortfall_weight: float # Penalty per 100M euros of annual-equivalent shortfall.

    # Solver and output settings
    # Control solver seconds, relative optimality gap, reproducibility and model size.
    # Snapshots count solver progress callbacks; GIF frame duration is milliseconds.
    solver_method: str # annealing, heuristic or milp
    annealing_proposal_cost_exponent: float # Demand / estimated corridor capital ** exponent
    annealing_action_fraction: float # Fraction of move families eligible per proposal
    annealing_candidate_pair_limit: int # Sampled regional pairs; zero restores exhaustive scanning
    annealing_growth_probability: float # Extra probability of choosing a corridor addition
    annealing_coverage_metric: str # needs, markets, population, or legacy demand
    annealing_consolidation_coverage: float
    annealing_integration_coverage: float
    annealing_connectivity_target: float
    annealing_connectivity_penalty: float
    annealing_network_effect_weight: float
    annealing_exploration_radius_km: float
    annealing_retained_ridership: float
    annealing_journey_time_allowance: float
    annealing_consolidation_candidates: int
    annealing_budget_mode: str # goals: finish stages; bounded: enforce total budgets
    annealing_iterations: int
    annealing_start_temperature: float
    annealing_end_temperature: float
    annealing_initial_financial_weight: float
    annealing_coverage_penalty: float
    heuristic_iterations: int
    heuristic_through_lines: int
    heuristic_demand_connections: int
    solver_time_limit: float
    mip_relative_gap: float
    random_seed: int
    maximum_model_variables: int
    snapshot_every: int
    demand_map_top_percent: float # Strongest undirected city pairs; always omit at least one pair.
    demand_map_min_trips_per_hour: float
    gif_frame_ms: int

    def seats_per_train(self, train_type='CR'):
        return self.train_seats * (2 if train_type == 'HSR' else 1)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        config = cls(**data)
        config.validate()
        return config

    def validate(self):
        if type(self.gravity_heuristic) is not bool or type(self.speed_up) is not bool:
            raise ValueError('gravity_heuristic and speed_up must be booleans')
        if not 0 < self.gravity_heuristic_demand_threshold <= 1:
            raise ValueError('gravity_heuristic_demand_threshold must be in (0, 1]')
        if self.gravity_heuristic and 'CR' not in self.train_types:
            raise ValueError('gravity_heuristic requires CR in train_types')
        if (not isinstance(self.train_types, list) or not self.train_types
                or any(t not in ('CR', 'HSR') for t in self.train_types)
                or len(set(self.train_types)) != len(self.train_types)):
            raise ValueError('train_types must contain unique CR and/or HSR entries')
        for key in ('hsr_capex_ratio', 'hsr_opex_ratio', 'hsr_speed_ratio', 'hsr_fare_ratio'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
                raise ValueError(f'{key} must be finite and at least 1')
        if not 0 < self.demand_map_top_percent <= 100:
            raise ValueError("demand_map_top_percent must be in (0, 100]")
        for definition in fields(self):
            value = getattr(self, definition.name)
            if definition.type in (int, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f'{definition.name} must be numeric')
                if definition.type is int and type(value) is not int:
                    raise ValueError(f'{definition.name} must be an integer')
            elif definition.type is bool and type(value) is not bool:
                raise ValueError(f'{definition.name} must be true or false')
            elif definition.type is str and not isinstance(value, str):
                raise ValueError(f'{definition.name} must be a string')
            elif definition.type is list and not isinstance(value, list):
                raise ValueError(f'{definition.name} must be a list')
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{key} must be finite and nonnegative')
        for key in ('max_cities', 'locations_per_city', 'neighbors_per_city', 'train_seats',
                    'accounting_years', 'appraisal_years', 'operating_days_per_year', 'terrain_search_limit',
                    'maximum_line_stops', 'track_trains_per_hour_per_direction', 'maximum_model_variables'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('market_radius_km', 'train_speed_kmh', 'access_speed_kmh', 'distance_floor_km',
                    'access_normalization_floor_km', 'access_optimizer_tolerance_km', 'solver_time_limit',
                    'gravity_beta', 'uptake_alpha', 'maximum_edge_km', 'routing_sample_km', 'water_track_cost_per_km',
                    'water_cost_growth_km', 'water_routing_sample_km',
                    'maximum_journey_hours', 'maximum_trip_distance_km', 'served_weight', 'minimum_horizon_profit'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('minimum_served_fraction', 'minimum_origin_served_fraction', 'p_max',
                    'terrain_cost_position', 'mip_relative_gap', 'financial_margin', 'maintenance_fraction',
                    'minimum_city_served_fraction', 'minimum_selected_origin_served_fraction'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must be between 0 and 1')
        if self.minimum_served_fraction == 0:
            raise ValueError('minimum_served_fraction must be > 0 to exclude the empty network')
        if not self.discount_rates or any(isinstance(r, bool) or not isinstance(r, (int, float))
                or not math.isfinite(r) or r < 0 for r in self.discount_rates):
            raise ValueError('discount_rates must contain finite nonnegative rates')
        if (not isinstance(self.asset_lives_years, dict)
                or set(self.asset_lives_years) != {'track', 'station', 'fleet'}
                or any(type(v) is not int or v < 1 for v in self.asset_lives_years.values())):
            raise ValueError('asset_lives_years must specify positive integer track, station and fleet lives')
        if not all(isinstance(country, str) for country in self.countries):
            raise ValueError('countries must be a list of country codes')
        if (len(self.bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.bbox)
                or not -180 <= self.bbox[0] < self.bbox[2] <= 180
                or not -90 < self.bbox[1] < self.bbox[3] < 90):
            raise ValueError('bbox must be [west, south, east, north], without antimeridian crossing')
        if not 0 <= self.operating_start_hour < self.operating_end_hour <= 24:
            raise ValueError('invalid operating hours')
        if type(self.population_subdivision) is not int or not 0 <= self.population_subdivision <= 2:
            raise ValueError('population_subdivision must be 0, 1 or 2')
        if not 2 <= self.maximum_line_stops <= 20:
            raise ValueError('maximum_line_stops must be between 2 and 20')
        if type(self.maximum_extra_lines) is not int or self.maximum_extra_lines < 0:
            raise ValueError('maximum_extra_lines must be a nonnegative integer')
        if type(self.maximum_shared_track_lines) is not int or self.maximum_shared_track_lines < 0:
            raise ValueError('maximum_shared_track_lines must be a nonnegative integer')
        if self.maximum_shared_track_detour < 1:
            raise ValueError('maximum_shared_track_detour must be at least 1')
        if type(self.maximum_alignment_variants_per_line) is not int or self.maximum_alignment_variants_per_line < 2:
            raise ValueError('maximum_alignment_variants_per_line must be an integer >= 2')
        if self.solver_method not in ('annealing', 'heuristic', 'milp'):
            raise ValueError('solver_method must be annealing, heuristic or milp')
        if 0 < self.maximum_journey_detour_ratio < 1:
            raise ValueError('maximum_journey_detour_ratio must be zero or at least 1')
        if not 0 <= self.annealing_proposal_cost_exponent <= 4:
            raise ValueError('annealing_proposal_cost_exponent must be in [0, 4]')
        if not 0 < self.annealing_action_fraction <= 1:
            raise ValueError('annealing_action_fraction must be in (0, 1]')
        if (isinstance(self.annealing_candidate_pair_limit, bool)
                or not isinstance(self.annealing_candidate_pair_limit, int)
                or self.annealing_candidate_pair_limit < 0):
            raise ValueError('annealing_candidate_pair_limit must be a nonnegative integer')
        if not 0 <= self.annealing_growth_probability <= 1:
            raise ValueError('annealing_growth_probability must be in [0, 1]')
        if self.annealing_coverage_metric not in ('needs', 'markets', 'population', 'demand'):
            raise ValueError('annealing_coverage_metric must be needs, markets, population, or demand')
        if self.annealing_budget_mode not in ('goals', 'bounded'):
            raise ValueError('annealing_budget_mode must be goals or bounded')
        if not 0 < self.annealing_integration_coverage <= 1:
            raise ValueError('annealing_integration_coverage must be in (0, 1]')
        if not 0 <= self.annealing_connectivity_target <= 1:
            raise ValueError('annealing_connectivity_target must be in [0, 1]')
        if self.annealing_exploration_radius_km <= 0:
            raise ValueError('annealing_exploration_radius_km must be positive')
        if not 0 < self.annealing_consolidation_coverage <= 1:
            raise ValueError('annealing_consolidation_coverage must be in (0, 1]')
        if not 0 < self.annealing_retained_ridership <= 1:
            raise ValueError('annealing_retained_ridership must be in (0, 1]')
        if self.annealing_consolidation_candidates < 1:
            raise ValueError('annealing_consolidation_candidates must be positive')
        if self.annealing_iterations < 2:
            raise ValueError('annealing_iterations must be at least 2')
        if not 0 < self.annealing_end_temperature <= self.annealing_start_temperature:
            raise ValueError('annealing temperatures must satisfy 0 < end <= start')
        if not 0 < self.annealing_initial_financial_weight <= 1:
            raise ValueError('annealing_initial_financial_weight must be in (0, 1]')
        if self.annealing_coverage_penalty <= 0:
            raise ValueError('annealing_coverage_penalty must be positive')
        if self.heuristic_iterations < 1:
            raise ValueError('heuristic_iterations must be positive')
        if self.center_method not in ('peak', 'circle', 'access') or self.demand_model not in ('gravity', 'production'):
            raise ValueError('invalid center_method or demand_model')
        if (not self.frequencies_per_hour or any(type(f) is not int or f < 1 for f in self.frequencies_per_hour)
                or len(set(self.frequencies_per_hour)) != len(self.frequencies_per_hour)):
            raise ValueError('frequencies_per_hour must contain unique positive integers')
        if type(self.minimum_service_frequency) is not int or self.minimum_service_frequency < 1:
            raise ValueError('minimum_service_frequency must be a positive integer')
        if not any(f >= self.minimum_service_frequency for f in self.frequencies_per_hour):
            raise ValueError('frequencies_per_hour has no option meeting minimum_service_frequency')

        if not isinstance(self.terrain_cost_ranges_million, dict):
            raise ValueError('terrain_cost_ranges_million must be an object')
        if 'bridge' in self.terrain_cost_ranges_million:
            raise ValueError('bridge terrain class was removed; use water for crossings')
        for name, bounds in self.terrain_cost_ranges_million.items():
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in bounds)
                    or bounds[0] > bounds[1]):
                raise ValueError(f'Invalid terrain cost range: {name}')
        if self.snapshot_every < 1 or not 10 <= self.gif_frame_ms <= 655350 or self.gif_frame_ms % 10:
            raise ValueError('Invalid snapshot_every or gif_frame_ms')

    @property
    def annual_hours(self):
        return (self.operating_end_hour - self.operating_start_hour) * self.operating_days_per_year


def load_config(path=DEFAULT_CONFIG_PATH, profile=None):
    """Load canonical defaults, optional external overrides, and a named profile."""
    defaults, profiles = read_configuration()
    if Path(path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        overrides, extra_profiles = read_configuration(path)
        defaults.update(overrides)
        profiles.update(extra_profiles)
    if profile is not None:
        if profile not in profiles:
            raise ValueError(f'Unknown profile {profile!r}; choose from {", ".join(sorted(profiles))}')
        defaults.update(profiles[profile])
    return Config.from_dict(defaults)
