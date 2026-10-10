"""ClimateGuard AI - Local Heat Hazard Dashboard Package.

Provides a read-only local dashboard and report loader for Chennai
biometeorological heat hazard forecast reports.
"""

from src.dashboard.report_loader import (
    HeatHazardReportLoader,
    filter_forecast_records,
)

__all__ = [
    "HeatHazardReportLoader",
    "filter_forecast_records",
]
