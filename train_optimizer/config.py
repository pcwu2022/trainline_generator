"""Validated, reproducible model assumptions (km, hours, people, currency)."""
from dataclasses import asdict, dataclass, field, fields
import math


@dataclass
class Config:
    # Input files and study region
    # Choose source datasets and the bounding box [west, south, east, north].
    # Population cells are filtered by bbox and country polygons; cities supply labels only.
    population_file: str = 'data/europe_3km.geojson'
    cities_file: str = 'data/cities50000.json'
    country_boundaries_file: str = 'data/country_boundaries.geojson'
    terrain_file: str = ''
    land_file: str = ''
    bbox: list = field(default_factory=lambda: [2.5, 49.4, 7.5, 53.7])
    countries: list = field(default_factory=lambda: ['NL', 'BE'])

    # Population and city selection
    # Greedy marginal coverage from eligible population cells.
    # Legacy city setting names now apply to population-derived markets.
    minimum_seed_spacing_km: float = 0  # Zero uses market_radius_km, independent of adaptive radius
    max_cities: int = 100
    market_radius_km: float = 15  # Maximum scoring and density-adaptive market radius
    minimum_city_population: int = 50000  # Target eligible population for adaptive radius; minimum seed score
    density_threshold: float = 100
    population_subdivision: int = 0

    # Station locations and catchments
    # Generate station positions and catchment radii (km) around each city.
    # Access normalization and tolerance control the station-location calculation.
    center_method: str = 'access'
    center_coefficient: float = 0.1  # Legacy metadata; unified market radius replaces it
    locations_per_city: int = 5
    station_offset_km: float = 12
    access_normalization_floor_km: float = 0.1
    access_optimizer_tolerance_km: float = 0.1

    # Passenger demand and uptake
    # Estimate directional trips from population and distance (km), then apply
    # catchment uptake. Production and gravity modes use different demand scales.
    demand_model: str = 'production'
    trips_per_person_hour: float = 0.005
    gravity_scale: float = 0.00002
    gravity_beta: float = 2
    distance_floor_km: float = 10
    maximum_trip_distance_km: float = 400
    p_max: float = 0.6
    uptake_a: float = 0.1
    uptake_alpha: float = 1
    uptake_reference_radius_km: float = 0  # Zero uses catchment radius; positive decouples uptake from market sizing

    # Track routing and terrain
    # Control candidate corridor connections, routing resolution and connectivity.
    # Terrain ranges are millions of euros/km; position 0/0.5/1 selects low/mid/high costs.
    neighbors_per_city: int = 3
    maximum_edge_km: float = 180
    require_connected_network: bool = True
    routing_sample_km: float = 5
    terrain_search_limit: int = 15000
    terrain_cost_position: float = 0.5
    terrain_cost_ranges_million: dict = field(default_factory=lambda: {
        "bridge": [20, 50], "hilly": [15, 25],
        "moderate_mountains": [25, 40], "difficult_mountains": [40, 70],
        "very_difficult": [70, 120],
    })

    # Train services and passenger journeys
    # Define candidate lines, departures/hour/direction, train capacity and speed
    # (km/h). Stop/transfer times are minutes; the OD-average journey limit is hours.
    maximum_line_stops: int = 20
    maximum_extra_lines: int = 40
    maximum_shared_track_lines: int = 40  # Additional three-stop hub candidates, independent of the generic line cap.
    maximum_shared_track_detour: float = 1.35  # Maximum length / direct endpoint distance.
    maximum_alignment_variants_per_line: int = 16  # Fastest, cheapest and mixed physical alignments.
    frequencies_per_hour: list = field(default_factory=lambda: [1, 2, 3, 4, 6, 10, 12])
    track_trains_per_hour_per_direction: int = 12
    train_seats: int = 500
    train_speed_kmh: float = 120
    dwell_minutes: float = 2
    turnaround_minutes: float = 5
    access_speed_kmh: float = 25
    transfer_minutes: float = 5
    maximum_journey_hours: float = 5

    # Operating calendar
    # Set daily departure hours and operating days/year for annual service,
    # revenue and operating-cost calculations.
    operating_start_hour: int = 6
    operating_end_hour: int = 22
    operating_days_per_year: int = 365

    # Construction and operating costs
    # Set capital and recurring costs in euros. Track costs are per km; the urban
    # coefficient multiplies density above the threshold (people/km²).
    # Maintenance is a yearly fraction of track capital.
    base_track_cost_per_km: float = 10_000_000
    urban_cost_coefficient: float = 5_000
    urban_density_threshold: float = 5_000
    station_capital_cost: float = 10_000_000
    station_annual_cost: float = 400_000
    train_capital_cost: float = 5_000_000
    train_cost_per_km: float = 15
    train_cost_per_departure: float = 0
    maintenance_fraction: float = 0.02

    # Revenue and financial targets
    # Set fares (euros/passenger-km), accounting years and the soft profit target.
    # The financial margin and minimum profit shape the target; the upfront budget is hard.
    ticket_per_km: float = 0.2
    accounting_years: int = 10  # Undiscounted cash-flow report only.
    appraisal_years: int = 30
    discount_rate: float = 0.04  # Real rate; costs and revenue are constant real prices.
    discount_rates: list = field(default_factory=lambda: [0.02, 0.04, 0.06])
    asset_lives_years: dict = field(default_factory=lambda: {'track': 40, 'station': 50, 'fleet': 25})
    financial_margin: float = 0.05
    minimum_horizon_profit: float = 1
    upfront_budget: float = 0  # Zero means no separate capital ceiling.
    maximum_annual_subsidy: float = 0  # Zero disables the hard lifecycle annual-equivalent subsidy ceiling.

    # Minimum service requirements
    # Require minimum served shares of fixed potential demand, globally and per
    # origin. The global target must be positive; zero disables the per-origin target.
    minimum_served_fraction: float = 0.001
    minimum_origin_served_fraction: float = 0
    minimum_city_served_fraction: float = 0  # Fraction of candidate markets with operating service.
    minimum_selected_origin_served_fraction: float = 0  # Fixed origin demand share, conditional on a station.
    minimum_service_frequency: int = 1  # Minimum frequency of each selected line, per direction.

    # Objective weights
    # Balance ridership benefits against journey time, station access, costs and
    # profit shortfall. Larger weights give a term more influence; zero disables its term.
    population_weight: float = 1  # Reward the share of fixed market population within selected catchments.
    served_weight: float = 10  # Reward completed passenger trips; transfers do not count as extra trips.
    time_weight: float = 0.5  # Passenger-hours, including access, waits and transfers.
    center_weight: float = 0
    access_weight: float = 0.05
    cost_weight: float = 0.001
    profit_shortfall_weight: float = 0.25  # Penalty per 100M euros of annual-equivalent shortfall.

    # Solver and output settings
    # Control solver seconds, relative optimality gap, reproducibility and model size.
    # Snapshots count solver progress callbacks; GIF frame duration is milliseconds.
    solver_time_limit: float = 600
    mip_relative_gap: float = 0.01
    random_seed: int = 42
    maximum_model_variables: int = 2_000_000
    snapshot_every: int = 5
    gif_frame_ms: int = 500

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        config = cls(**data)
        config.validate()
        return config

    def validate(self):
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
                    'gravity_beta', 'uptake_alpha', 'maximum_edge_km', 'routing_sample_km',
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
