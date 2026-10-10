"""Report loader and record filtering utility for ClimateGuard AI Dashboard."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.ingestion.config import DATA_PROCESSED_DIR

DEFAULT_REPORT_PATH = DATA_PROCESSED_DIR / "chennai" / "heat_hazard_forecast.json"

REQUIRED_TOP_LEVEL_KEYS = [
    "metadata",
    "scientific_limitations_and_disclaimer",
    "hourly_forecast_hazard_records",
]

REQUIRED_METADATA_KEYS = [
    "generation_timestamp_utc",
    "forecast_source_file",
    "total_forecast_hours",
    "valid_records_count",
    "alert_tier_distribution",
]


class HeatHazardReportLoader:
    """Safe, read-only loader for the heat hazard forecast JSON report."""

    def __init__(self, report_path: Optional[Path] = None):
        self.report_path = Path(report_path or DEFAULT_REPORT_PATH)

    def file_exists(self) -> bool:
        """Check if target report file exists on disk."""
        return self.report_path.is_file()

    def load_report(self) -> Dict[str, Any]:
        """Load and validate the report JSON payload.
        
        Returns:
            Dict containing:
              - 'status': 'SUCCESS' or 'ERROR'
              - 'message': Descriptive status explanation
              - 'data': Full payload if SUCCESS, None if ERROR
              - 'available_dates': List of unique YYYY-MM-DD date strings
              - 'available_tiers': List of alert tier strings present
        """
        if not self.file_exists():
            return {
                "status": "ERROR",
                "message": f"Report file not found at: {self.report_path}. Run the heat hazard service first.",
                "data": None,
                "available_dates": [],
                "available_tiers": [],
            }

        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            return {
                "status": "ERROR",
                "message": f"Report file is malformed JSON: {exc}",
                "data": None,
                "available_dates": [],
                "available_tiers": [],
            }
        except Exception as exc:
            return {
                "status": "ERROR",
                "message": f"Unable to read report file: {exc}",
                "data": None,
                "available_dates": [],
                "available_tiers": [],
            }

        # Check required schema keys
        for key in REQUIRED_TOP_LEVEL_KEYS:
            if key not in payload:
                return {
                    "status": "ERROR",
                    "message": f"Report schema missing required top-level key '{key}'",
                    "data": None,
                    "available_dates": [],
                    "available_tiers": [],
                }

        meta = payload.get("metadata", {})
        for m_key in REQUIRED_METADATA_KEYS:
            if m_key not in meta:
                return {
                    "status": "ERROR",
                    "message": f"Report metadata missing required field '{m_key}'",
                    "data": None,
                    "available_dates": [],
                    "available_tiers": [],
                }

        records = payload.get("hourly_forecast_hazard_records", [])
        
        # Extract unique dates and alert tiers for filter UI
        dates_set = set()
        tiers_set = set()
        for r in records:
            t = r.get("time")
            if t and isinstance(t, str) and len(t) >= 10:
                dates_set.add(t[:10])
            tier = r.get("alert_tier")
            if tier:
                tiers_set.add(tier)

        sorted_dates = sorted(list(dates_set))
        sorted_tiers = sorted(list(tiers_set))

        return {
            "status": "SUCCESS",
            "message": "Report loaded and verified successfully.",
            "data": payload,
            "available_dates": sorted_dates,
            "available_tiers": sorted_tiers,
        }


def filter_forecast_records(records: List[Dict[str, Any]],
                            alert_tier: Optional[str] = None,
                            date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """Filter hourly records by alert tier and/or forecast date (YYYY-MM-DD)."""
    filtered = []
    tier_filter = alert_tier.strip().upper() if alert_tier and alert_tier.strip().upper() != "ALL" else None
    date_filter = date_str.strip() if date_str and date_str.strip().upper() != "ALL" else None

    for r in records:
        # Check tier
        if tier_filter:
            r_tier = str(r.get("alert_tier") or "").upper()
            if tier_filter not in r_tier and not r_tier.startswith(tier_filter):
                continue
        # Check date prefix
        t = str(r.get("time") or "")
        if date_filter and not t.startswith(date_filter):
            continue
        filtered.append(r)

    return filtered
