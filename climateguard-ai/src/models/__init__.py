"""ClimateGuard AI - Analytical and Predictive Models Package.

Contains biometeorological hazard models, heat index calculations,
heat alert tier classifications, and forecast integration services.
"""

from src.models.heat_hazard_engine import (
    BiometeorologicalHazardEngine,
    calculate_steadman_apparent_temperature,
    calculate_noaa_heat_index,
    classify_heat_alert_tier,
    DEFAULT_ALERT_THRESHOLDS,
    ALERT_TIERS,
)
from src.models.forecast_risk_service import (
    ForecastHeatHazardService,
    DEFAULT_HEAT_HAZARD_OUTPUT_PATH,
)

__all__ = [
    "BiometeorologicalHazardEngine",
    "calculate_steadman_apparent_temperature",
    "calculate_noaa_heat_index",
    "classify_heat_alert_tier",
    "DEFAULT_ALERT_THRESHOLDS",
    "ALERT_TIERS",
    "ForecastHeatHazardService",
    "DEFAULT_HEAT_HAZARD_OUTPUT_PATH",
]
