"""Forecast Heat Hazard Integration Service for ClimateGuard AI.

Ingests the validated 168-hour weather forecast, validates individual hourly
meteorological parameters, evaluates biometeorological hazard metrics using the
hazard engine, and produces a structured, reproducible heat hazard forecast report.

Scientific & Operational Notice:
These metrics are derived from numerical weather model forecasts at the Chennai centroid.
They are NOT ground-truth physical station observations and must NOT be interpreted
as clinical health predictions.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from src.ingestion.config import (
    CACHE_DIR,
    DATA_PROCESSED_DIR,
    CHENNAI_CENTROID,
)
from src.models.heat_hazard_engine import (
    BiometeorologicalHazardEngine,
    DEFAULT_ALERT_THRESHOLDS,
    ALERT_TIERS,
)

DEFAULT_FORECAST_CACHE_PATH = CACHE_DIR / "weather_chennai_forecast_168h.json"
DEFAULT_HEAT_HAZARD_OUTPUT_PATH = DATA_PROCESSED_DIR / "chennai" / "heat_hazard_forecast.json"


class ForecastHeatHazardService:
    """Service processing 168-hour weather forecast records into biometeorological
    heat hazard levels and public health alert tiers without modifying source data.
    """

    def __init__(self,
                 cache_file_path: Optional[Path] = None,
                 output_file_path: Optional[Path] = None,
                 thresholds: Optional[Dict[str, float]] = None):
        self.cache_file = Path(cache_file_path or DEFAULT_FORECAST_CACHE_PATH)
        self.output_file = Path(output_file_path or DEFAULT_HEAT_HAZARD_OUTPUT_PATH)
        self.thresholds = thresholds or dict(DEFAULT_ALERT_THRESHOLDS)
        self.engine = BiometeorologicalHazardEngine(thresholds=self.thresholds)

    def load_raw_forecast(self) -> Dict[str, Any]:
        """Read existing cached forecast JSON file without modifying it."""
        if not self.cache_file.is_file():
            raise FileNotFoundError(
                f"Weather forecast cache file not found at: {self.cache_file}. "
                "Run weather ingestion first."
            )
        with open(self.cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def validate_hourly_inputs(temp_val: Any,
                               rh_val: Any,
                               wind_val: Any) -> Tuple[bool, Optional[str], Optional[float], Optional[float], Optional[float]]:
        """Validate individual hourly observation fields.
        
        Returns:
            (is_valid, error_reason, clean_temp, clean_rh, clean_wind)
        """
        if temp_val is None:
            return False, "Missing air temperature (temperature_2m is None)", None, None, None
        if rh_val is None:
            return False, "Missing relative humidity (relative_humidity_2m is None)", None, None, None

        try:
            t = float(temp_val)
            rh = float(rh_val)
            w = float(wind_val) if wind_val is not None else 0.0
        except (ValueError, TypeError) as exc:
            return False, f"Non-numeric weather value: {exc}", None, None, None

        if np.isnan(t) or np.isnan(rh):
            return False, "NaN value in temperature or humidity", None, None, None

        # Plausibility physical boundaries
        if not (-30.0 <= t <= 65.0):
            return False, f"Temperature {t}°C out of physical range [-30, 65]", None, None, None
        if not (0.0 <= rh <= 100.0):
            return False, f"Relative humidity {rh}% out of range [0, 100]", None, None, None
        if w < 0.0:
            return False, f"Wind speed {w} km/h cannot be negative", None, None, None

        return True, None, t, rh, w

    def process_forecast(self, raw_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process hourly forecast records, preserving timestamps and logging invalid points."""
        data = raw_data if raw_data is not None else self.load_raw_forecast()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        rhs = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("wind_speed_10m", [0.0] * len(times))

        if not times or not temps or not rhs:
            raise ValueError("Forecast payload does not contain required hourly time/temperature/humidity arrays.")

        records: List[Dict[str, Any]] = []
        valid_count = 0
        invalid_count = 0

        for idx, t_str in enumerate(times):
            temp_raw = temps[idx] if idx < len(temps) else None
            rh_raw = rhs[idx] if idx < len(rhs) else None
            wind_raw = winds[idx] if idx < len(winds) else 0.0

            is_valid, reason, t_clean, rh_clean, w_clean = self.validate_hourly_inputs(temp_raw, rh_raw, wind_raw)

            if is_valid and t_clean is not None and rh_clean is not None:
                valid_count += 1
                condition_eval = self.engine.evaluate_condition(
                    temp_c=t_clean,
                    relative_humidity=rh_clean,
                    wind_speed_kmh=w_clean
                )
                records.append({
                    "time": str(t_str),
                    "valid": True,
                    "validation_notes": None,
                    "air_temperature_c": t_clean,
                    "relative_humidity_pct": rh_clean,
                    "wind_speed_kmh": w_clean,
                    "steadman_apparent_temp_c": condition_eval["steadman_apparent_temp_c"],
                    "noaa_heat_index_c": condition_eval["noaa_heat_index_c"],
                    "selected_hazard_metric": condition_eval["primary_hazard_metric"],
                    "hazard_temperature_c": condition_eval["primary_hazard_value_c"],
                    "alert_tier": condition_eval["alert_tier"],
                })
            else:
                invalid_count += 1
                records.append({
                    "time": str(t_str) if t_str else None,
                    "valid": False,
                    "validation_notes": reason,
                    "air_temperature_c": None,
                    "relative_humidity_pct": None,
                    "wind_speed_kmh": None,
                    "steadman_apparent_temp_c": None,
                    "noaa_heat_index_c": None,
                    "selected_hazard_metric": None,
                    "hazard_temperature_c": None,
                    "alert_tier": ALERT_TIERS["UNKNOWN"],
                })

        # Summary statistics calculation
        valid_hazards = [r["hazard_temperature_c"] for r in records if r["valid"] and r["hazard_temperature_c"] is not None]
        valid_air_temps = [r["air_temperature_c"] for r in records if r["valid"] and r["air_temperature_c"] is not None]

        tier_counts: Dict[str, int] = {}
        for r in records:
            tier = r["alert_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        peak_record = None
        if valid_hazards:
            max_hazard = max(valid_hazards)
            min_hazard = min(valid_hazards)
            mean_hazard = round(sum(valid_hazards) / len(valid_hazards), 2)
            peak_item = next(r for r in records if r.get("hazard_temperature_c") == max_hazard)
            peak_record = {
                "time": peak_item["time"],
                "hazard_temperature_c": peak_item["hazard_temperature_c"],
                "air_temperature_c": peak_item["air_temperature_c"],
                "relative_humidity_pct": peak_item["relative_humidity_pct"],
                "alert_tier": peak_item["alert_tier"],
            }
        else:
            max_hazard, min_hazard, mean_hazard = None, None, None

        payload = {
            "metadata": {
                "project_name": "ClimateGuard AI - Chennai Urban Heat Risk Planner",
                "model_component": "Local Biometeorological Heat Hazard Forecast Integration",
                "target_location": {
                    "city": "Chennai, Tamil Nadu, India",
                    "latitude": data.get("latitude", CHENNAI_CENTROID["latitude"]),
                    "longitude": data.get("longitude", CHENNAI_CENTROID["longitude"]),
                    "timezone": data.get("timezone", "Asia/Kolkata"),
                },
                "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "forecast_source_file": str(self.cache_file),
                "total_forecast_hours": len(records),
                "valid_records_count": valid_count,
                "invalid_records_count": invalid_count,
                "alert_tier_distribution": tier_counts,
                "hazard_temperature_summary_c": {
                    "min": min_hazard,
                    "max": max_hazard,
                    "mean": mean_hazard,
                    "air_temp_max": max(valid_air_temps) if valid_air_temps else None,
                },
                "peak_hazard_observation": peak_record,
                "thresholds_used": self.thresholds,
            },
            "scientific_limitations_and_disclaimer": {
                "forecast_nature": (
                    "Hazard scores are computed from Open-Meteo numerical weather prediction forecasts at the Chennai centroid. "
                    "They are not verified ground station observations from IMD instruments."
                ),
                "biometeorological_scope": (
                    "Indices estimate environmental thermal stress on standard human physiology in shade with light wind. "
                    "They are NOT clinical health predictions or medical advice."
                ),
                "solar_radiation_warning": (
                    "Calculations assume shaded conditions. Direct sunlight exposure can elevate perceived heat load by up to 4°C to 8°C."
                ),
            },
            "hourly_forecast_hazard_records": records,
        }

        return payload

    def save_report(self, payload: Dict[str, Any]) -> Path:
        """Serialize processed forecast report to JSON output directory."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return self.output_file

    def run(self) -> Dict[str, Any]:
        """Execute full forecast hazard processing and save output JSON."""
        payload = self.process_forecast()
        self.save_report(payload)
        return payload
