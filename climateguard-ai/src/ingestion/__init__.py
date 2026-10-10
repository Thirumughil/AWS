"""ClimateGuard AI - Data Ingestion and Validation Package for Chennai.

Provides robust, modular, and cached ingestion for administrative boundaries,
meteorological forecasts, critical facilities, and building footprints.
"""

from src.ingestion.config import (
    CHENNAI_BBOX,
    CHENNAI_CENTROID,
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
)
from src.ingestion.boundary_loader import ChennaiBoundaryLoader
from src.ingestion.weather_ingestor import WeatherIngestor
from src.ingestion.facilities_ingestor import FacilitiesIngestor
from src.ingestion.building_ingestor import BuildingFootprintIngestor

__all__ = [
    "CHENNAI_BBOX",
    "CHENNAI_CENTROID",
    "GEOGRAPHIC_CRS",
    "PROJECTED_CRS",
    "ChennaiBoundaryLoader",
    "WeatherIngestor",
    "FacilitiesIngestor",
    "BuildingFootprintIngestor",
]
