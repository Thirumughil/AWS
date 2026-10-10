"""Cached Open-Meteo weather forecast ingestor and validation gate for Chennai."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import requests
import pandas as pd
import numpy as np

from src.ingestion.config import (
    OPEN_METEO_FORECAST_URL,
    CHENNAI_CENTROID,
    WEATHER_FORECAST_HOURS,
    WEATHER_CACHE_TTL_SECONDS,
    WEATHER_REQUIRED_HOURLY_FIELDS,
    WEATHER_OPTIONAL_HOURLY_FIELDS,
    WEATHER_VALUE_BOUNDS,
    CACHE_DIR,
)


class WeatherIngestor:
    """Ingests, caches, and validates 168-hour weather forecast from Open-Meteo.
    
    Operates idempotently with a local file cache to prevent redundant API queries.
    """

    def __init__(self,
                 cache_dir: Optional[Path] = None,
                 cache_ttl_seconds: int = WEATHER_CACHE_TTL_SECONDS):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "weather_chennai_forecast_168h.json"
        self.cache_ttl = cache_ttl_seconds

    def is_cache_valid(self) -> bool:
        """Check if local forecast cache exists and is within TTL."""
        if not self.cache_file.is_file():
            return False
        mtime = self.cache_file.stat().st_mtime
        age = time.time() - mtime
        return age < self.cache_ttl

    def fetch_forecast_raw(self, force_refresh: bool = False) -> Tuple[Dict[str, Any], bool]:
        """Fetch raw JSON payload either from disk cache or Open-Meteo API.
        
        Returns:
            (payload_dict, from_cache_boolean)
        """
        if not force_refresh and self.is_cache_valid():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data, True
            except Exception:
                pass  # Fall through to network fetch if cache is corrupted

        # Query Open-Meteo API
        params = {
            "latitude": CHENNAI_CENTROID["latitude"],
            "longitude": CHENNAI_CENTROID["longitude"],
            "hourly": ",".join(WEATHER_REQUIRED_HOURLY_FIELDS[1:] + WEATHER_OPTIONAL_HOURLY_FIELDS),
            "forecast_days": 7,
            "timezone": "auto",
        }

        try:
            resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            # Save to disk cache
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return data, False
        except Exception as exc:
            # If network fails but expired cache exists, return stale cache with warning
            if self.cache_file.is_file():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    stale_data = json.load(f)
                stale_data["_stale_cache_warning"] = str(exc)
                return stale_data, True
            raise ConnectionError(
                f"Failed to fetch Open-Meteo forecast and no cache exists: {exc}"
            )

    def ingest_and_validate(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Execute full forecast ingestion and field-level validation gate.
        
        Returns:
            Dictionary containing 'status', 'df_forecast', 'metrics', and 'report'.
        """
        raw_data, from_cache = self.fetch_forecast_raw(force_refresh=force_refresh)
        hourly = raw_data.get("hourly", {})

        # Check for required fields
        missing_fields = [f for f in WEATHER_REQUIRED_HOURLY_FIELDS if f not in hourly]
        if missing_fields:
            raise KeyError(f"Weather response missing required hourly fields: {missing_fields}")

        # Assemble DataFrame
        df = pd.DataFrame({f: hourly[f] for f in WEATHER_REQUIRED_HOURLY_FIELDS if f in hourly})
        for opt in WEATHER_OPTIONAL_HOURLY_FIELDS:
            if opt in hourly:
                df[opt] = hourly[opt]

        df["time"] = pd.to_datetime(df["time"])
        total_hours = len(df)

        # Check timestamp continuity (hourly steps)
        time_diffs = df["time"].diff().dropna()
        has_gaps = any(d != pd.Timedelta(hours=1) for d in time_diffs)

        # Quality validation on value bounds & missing values
        anomalies: Dict[str, int] = {}
        null_counts = df[WEATHER_REQUIRED_HOURLY_FIELDS].isnull().sum().to_dict()

        for col, bounds in WEATHER_VALUE_BOUNDS.items():
            if col in df.columns:
                out_of_bounds = ((df[col] < bounds["min"]) | (df[col] > bounds["max"])).sum()
                anomalies[col] = int(out_of_bounds)

        has_nulls = sum(null_counts.values()) > 0
        total_anomalies = sum(anomalies.values())
        hours_ok = (total_hours >= WEATHER_FORECAST_HOURS)

        status = "PASS" if (hours_ok and not has_nulls and total_anomalies == 0) else "WARN"

        summary_metrics = {
            "hours_returned": total_hours,
            "start_time": str(df["time"].min()),
            "end_time": str(df["time"].max()),
            "temp_min_c": round(float(df["temperature_2m"].min()), 1),
            "temp_max_c": round(float(df["temperature_2m"].max()), 1),
            "temp_mean_c": round(float(df["temperature_2m"].mean()), 1),
            "apparent_temp_max_c": round(float(df["apparent_temperature"].max()), 1),
            "humidity_mean_pct": round(float(df["relative_humidity_2m"].mean()), 1),
        }

        report = {
            "status": status,
            "source": OPEN_METEO_FORECAST_URL,
            "from_cache": from_cache,
            "cache_file": str(self.cache_file),
            "total_records": total_hours,
            "expected_hours": WEATHER_FORECAST_HOURS,
            "time_continuity_valid": not has_gaps,
            "null_values": null_counts,
            "out_of_bounds_anomalies": anomalies,
            "summary_metrics": summary_metrics,
        }

        return {
            "status": status,
            "df": df,
            "summary": summary_metrics,
            "report": report,
        }
