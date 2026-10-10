"""Configuration parameters, data contracts, and spatial constants for Chennai ingestion."""

from pathlib import Path
from typing import Dict, Any, List

# Workspace Root Paths
INGESTION_DIR = Path(__file__).resolve().parent
SRC_DIR = INGESTION_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"

# Input Data File Paths
CHENNAI_BOUNDARY_PATH = DATA_RAW_DIR / "chennai_tamil_nadu_india_boundary.geojson"

# Sibling workspace sample building path (fallback reference)
SIBLING_BUILDING_SAMPLE_PATH = (
    PROJECT_ROOT.parent / "urban-heat-python" / "data" / "raw" / "chennai_buildings_sample.geojson"
)

# Coordinate Reference Systems
GEOGRAPHIC_CRS = "EPSG:4326"   # WGS84 (Lon/Lat) for web services & GeoJSON
PROJECTED_CRS = "EPSG:32644"    # WGS 84 / UTM Zone 44N (metric coordinates for Chennai)

# Chennai Municipal Extents (Greater Chennai Corporation - GCC)
CHENNAI_BBOX = {
    "west": 80.1339233,
    "south": 12.8519771,
    "east": 80.3328982,
    "north": 13.2351580,
}

CHENNAI_CENTROID = {
    "latitude": 13.0801,
    "longitude": 80.2808,
}

EXPECTED_AREA_KM2_MIN = 400.0
EXPECTED_AREA_KM2_MAX = 460.0

# Open-Meteo Weather Forecast Settings
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FORECAST_HOURS = 168  # 7-day hourly forecast
WEATHER_CACHE_TTL_SECONDS = 3600  # 1 hour cache validity

WEATHER_REQUIRED_HOURLY_FIELDS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
]

WEATHER_OPTIONAL_HOURLY_FIELDS = [
    "precipitation",
    "wind_speed_10m",
    "direct_normal_irradiance",
]

# Plausible Meteorological Value Ranges (for quality validation)
WEATHER_VALUE_BOUNDS = {
    "temperature_2m": {"min": 10.0, "max": 52.0},         # Chennai tropical range (°C)
    "relative_humidity_2m": {"min": 5.0, "max": 100.0},    # Percentage (%)
    "apparent_temperature": {"min": 10.0, "max": 65.0},    # Heat index (°C)
}

# Overpass API Settings & Endpoints
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
USER_AGENT = "ClimateGuardAI-ChennaiIngestion/1.0"

# Critical Facility Amenity Types
FACILITY_CATEGORIES: Dict[str, str] = {
    "hospitals": "amenity=hospital",
    "schools": "amenity=school",
    "community_centres": "amenity=community_centre",
    "libraries": "amenity=library",
    "parks": "leisure=park",
    "drinking_water": "amenity=drinking_water",
}
