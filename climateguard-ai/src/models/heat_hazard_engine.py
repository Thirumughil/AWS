"""Biometeorological Heat Hazard Engine for ClimateGuard AI.

Calculates Steadman's Apparent Temperature and the NOAA/NWS Rothfusz Heat Index
with relative humidity adjustments, and assigns heat alert tiers.

Scientific and Operational Notice:
These calculations are biometeorological thermal stress indices. They are
intended for urban heat hazard assessment and public-health advisory planning,
and must NOT be interpreted as a clinically validated individual diagnostic tool.
Alert thresholds represent pilot project configurations adapted from international
guidance and do NOT claim official endorsement from TNSDMA or IMD.
"""

import math
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np

# Default Alert Thresholds in Celsius (°C)
# Sourced from NOAA National Weather Service heat index alert categories
# adapted for tropical coastal urban environments.
DEFAULT_ALERT_THRESHOLDS = {
    "green_max": 32.0,    # Normal (< 32.0°C)
    "yellow_max": 41.0,   # Caution (32.0°C - 40.9°C)
    "orange_max": 54.0,   # Extreme Caution (41.0°C - 53.9°C)
    # Danger / Extreme Danger is >= 54.0°C
}

ALERT_TIERS = {
    "GREEN": "Green (Normal)",
    "YELLOW": "Yellow (Caution)",
    "ORANGE": "Orange (Extreme Caution)",
    "RED": "Red (Danger)",
    "UNKNOWN": "Unknown (Invalid/Missing Data)",
}


def calculate_water_vapor_pressure(temp_c: float, relative_humidity: float) -> float:
    """Calculate ambient water vapor pressure (e) in hectopascals (hPa).
    
    Uses the Tetens/Magnus formula for saturation vapor pressure over water:
      e_sat(T) = 6.105 * exp((17.27 * T) / (237.7 + T))
      e = (RH / 100) * e_sat(T)
      
    Args:
        temp_c: Dry-bulb air temperature in degrees Celsius (°C).
        relative_humidity: Relative humidity in percentage (0 to 100%).
        
    Returns:
        Vapor pressure in hPa (millibars).
    """
    if temp_c is None or relative_humidity is None:
        raise ValueError("Temperature and humidity cannot be None.")
    
    # Magnus formula saturation pressure in hPa
    e_sat = 6.105 * math.exp((17.27 * temp_c) / (237.7 + temp_c))
    vapor_pressure = (relative_humidity / 100.0) * e_sat
    return vapor_pressure


def calculate_steadman_apparent_temperature(temp_c: float,
                                            relative_humidity: float,
                                            wind_speed_mps: float = 0.0) -> float:
    """Calculate Steadman's Apparent Temperature (AT) in degrees Celsius (°C).
    
    Formula (Steadman 1979, Australian Bureau of Meteorology formulation):
      AT = T + (0.33 * e) - (0.70 * v) - 4.00
      
    where:
      T = Dry-bulb temperature (°C)
      e = Ambient water vapor pressure (hPa)
      v = Wind speed at 10m in meters per second (m/s)
      
    Args:
        temp_c: Dry-bulb air temperature in °C (valid range: -30°C to 65°C).
        relative_humidity: Relative humidity in % (valid range: 0% to 100%).
        wind_speed_mps: Wind speed in m/s (valid range: >= 0 m/s). Default is 0.0 m/s.
        
    Returns:
        Apparent temperature in °C, rounded to 2 decimal places.
    """
    if temp_c is None or relative_humidity is None:
        return float("nan")
        
    try:
        t = float(temp_c)
        rh = float(relative_humidity)
        v = float(wind_speed_mps) if wind_speed_mps is not None else 0.0
    except (TypeError, ValueError):
        return float("nan")

    if math.isnan(t) or math.isnan(rh) or math.isnan(v):
        return float("nan")

    # Input boundary validation
    if not (-30.0 <= t <= 65.0) or not (0.0 <= rh <= 100.0) or v < 0.0:
        return float("nan")

    e = calculate_water_vapor_pressure(t, rh)
    apparent_temp = t + (0.33 * e) - (0.70 * v) - 4.00
    return round(apparent_temp, 2)


def calculate_noaa_heat_index(temp_c: float, relative_humidity: float) -> float:
    """Calculate NOAA/NWS Heat Index using the full Rothfusz polynomial regression.
    
    The National Weather Service algorithm operates in degrees Fahrenheit (°F):
    1. If T_F < 80°F, use Steadman's simple linear equation:
         HI_simple = 0.5 * (T_F + 61.0 + ((T_F - 68.0) * 1.2) + (RH * 0.094))
       If the mean of HI_simple and T_F is < 80°F, HI = HI_simple.
    2. If T_F >= 80°F, use the full 9-term Rothfusz polynomial:
         HI = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH
              - 0.00683783*T^2 - 0.05481717*RH^2 + 0.00122874*T^2*RH
              + 0.00085282*T*RH^2 - 0.00000199*T^2*RH^2
    3. Low humidity adjustment (if RH < 13% and 80°F <= T_F <= 112°F):
         adj = -((13 - RH) / 4) * sqrt((17 - |T_F - 95|) / 17)
    4. High humidity adjustment (if RH > 85% and 80°F <= T_F <= 87°F):
         adj = ((RH - 85) / 10) * ((87 - T_F) / 5)
         
    Args:
        temp_c: Dry-bulb air temperature in °C (valid range: 15°C to 60°C).
        relative_humidity: Relative humidity in % (valid range: 0% to 100%).
        
    Returns:
        Heat Index in degrees Celsius (°C), rounded to 2 decimal places.
    """
    if temp_c is None or relative_humidity is None:
        return float("nan")

    try:
        t_c = float(temp_c)
        rh = float(relative_humidity)
    except (TypeError, ValueError):
        return float("nan")

    if math.isnan(t_c) or math.isnan(rh):
        return float("nan")

    if not (-20.0 <= t_c <= 65.0) or not (0.0 <= rh <= 100.0):
        return float("nan")

    # Below 20°C (68°F), the heat index does not apply; return dry-bulb temperature
    if t_c < 20.0:
        return round(t_c, 2)

    # Convert Celsius to Fahrenheit
    t_f = (t_c * 9.0 / 5.0) + 32.0

    # Steadman simple approximation
    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))

    # If simple heat index average is under 80°F, use simple index
    if ((hi_simple + t_f) / 2.0) < 80.0:
        hi_f = hi_simple
    else:
        # Full Rothfusz regression equation
        hi_f = (
            -42.379
            + (2.04901523 * t_f)
            + (10.14333127 * rh)
            - (0.22475541 * t_f * rh)
            - (0.00683783 * (t_f ** 2))
            - (0.05481717 * (rh ** 2))
            + (0.00122874 * (t_f ** 2) * rh)
            + (0.00085282 * t_f * (rh ** 2))
            - (0.00000199 * (t_f ** 2) * (rh ** 2))
        )

        # Low humidity adjustment
        if rh < 13.0 and (80.0 <= t_f <= 112.0):
            adj = -((13.0 - rh) / 4.0) * math.sqrt((17.0 - abs(t_f - 95.0)) / 17.0)
            hi_f += adj

        # High humidity adjustment
        elif rh > 85.0 and (80.0 <= t_f <= 87.0):
            adj = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
            hi_f += adj

    # Convert result back to Celsius
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return round(hi_c, 2)


def classify_heat_alert_tier(hazard_temp_c: float,
                             thresholds: Optional[Dict[str, float]] = None) -> str:
    """Classify a thermal hazard temperature into an alert tier.
    
    Args:
        hazard_temp_c: Heat index or apparent temperature in °C.
        thresholds: Dict with 'green_max', 'yellow_max', 'orange_max'.
        
    Returns:
        Alert tier string ('Green (Normal)', 'Yellow (Caution)', etc.).
    """
    if hazard_temp_c is None:
        return ALERT_TIERS["UNKNOWN"]
        
    try:
        val = float(hazard_temp_c)
    except (TypeError, ValueError):
        return ALERT_TIERS["UNKNOWN"]

    if math.isnan(val) or math.isinf(val):
        return ALERT_TIERS["UNKNOWN"]

    th = thresholds or DEFAULT_ALERT_THRESHOLDS

    if val < th["green_max"]:
        return ALERT_TIERS["GREEN"]
    elif val < th["yellow_max"]:
        return ALERT_TIERS["YELLOW"]
    elif val < th["orange_max"]:
        return ALERT_TIERS["ORANGE"]
    else:
        return ALERT_TIERS["RED"]


class BiometeorologicalHazardEngine:
    """Engine orchestrating thermal stress calculations and alert tier assignments."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or dict(DEFAULT_ALERT_THRESHOLDS)

    def evaluate_condition(self,
                           temp_c: float,
                           relative_humidity: float,
                           wind_speed_kmh: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate a single meteorological observation.
        
        Selection rationale:
        - When wind speed is available and conditions are tropical humid, Steadman's
          Apparent Temperature provides convective cooling realism.
        - NOAA Heat Index provides standard NWS evaporation-inhibition reference.
        - The primary hazard score is the NOAA Heat Index for T >= 20°C, supplemented
          by Steadman's Apparent Temperature.
        """
        wind_mps = (float(wind_speed_kmh) / 3.6) if (wind_speed_kmh is not None and not math.isnan(wind_speed_kmh)) else 0.0

        apparent_temp = calculate_steadman_apparent_temperature(temp_c, relative_humidity, wind_mps)
        noaa_hi = calculate_noaa_heat_index(temp_c, relative_humidity)

        # Primary index selection
        primary_metric = "NOAA_Heat_Index" if not math.isnan(noaa_hi) else "Steadman_Apparent_Temp"
        hazard_value = noaa_hi if not math.isnan(noaa_hi) else apparent_temp
        alert_tier = classify_heat_alert_tier(hazard_value, self.thresholds)

        return {
            "air_temperature_c": temp_c,
            "relative_humidity_pct": relative_humidity,
            "wind_speed_kmh": wind_speed_kmh,
            "steadman_apparent_temp_c": apparent_temp,
            "noaa_heat_index_c": noaa_hi,
            "primary_hazard_metric": primary_metric,
            "primary_hazard_value_c": hazard_value,
            "alert_tier": alert_tier,
        }

    def process_cached_forecast_file(self, cache_file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Process 168-hour forecast from local cache without modifying the source file.
        
        Preserves original timestamps and reports missing/anomalous observations.
        """
        from src.ingestion.config import CACHE_DIR
        c_path = Path(cache_file_path or (CACHE_DIR / "weather_chennai_forecast_168h.json"))
        
        if not c_path.is_file():
            raise FileNotFoundError(f"Forecast cache file not found at {c_path}")

        import json
        with open(c_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        rhs = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("wind_speed_10m", [0.0] * len(times))

        records: List[Dict[str, Any]] = []
        valid_count = 0
        missing_count = 0

        for t, temp, rh, wind in zip(times, temps, rhs, winds):
            if temp is None or rh is None:
                missing_count += 1
                records.append({
                    "time": t,
                    "air_temperature_c": None,
                    "relative_humidity_pct": None,
                    "steadman_apparent_temp_c": None,
                    "noaa_heat_index_c": None,
                    "primary_hazard_value_c": None,
                    "alert_tier": ALERT_TIERS["UNKNOWN"],
                })
            else:
                valid_count += 1
                eval_res = self.evaluate_condition(temp, rh, wind)
                eval_res["time"] = t
                records.append(eval_res)

        df = pd.DataFrame(records)

        # Summary of alert distributions
        alert_counts = df["alert_tier"].value_counts().to_dict()
        max_hazard = df["primary_hazard_value_c"].max()
        mean_hazard = df["primary_hazard_value_c"].mean()

        return {
            "source_file": str(c_path),
            "total_timesteps": len(records),
            "valid_timesteps": valid_count,
            "missing_timesteps": missing_count,
            "max_hazard_c": round(float(max_hazard), 2) if not math.isnan(max_hazard) else None,
            "mean_hazard_c": round(float(mean_hazard), 2) if not math.isnan(mean_hazard) else None,
            "alert_tier_distribution": alert_counts,
            "records": records,
            "df": df,
        }
