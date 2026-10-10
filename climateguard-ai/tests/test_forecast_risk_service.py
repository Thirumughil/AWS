"""Automated unit and integration tests for ForecastHeatHazardService."""

import json
import tempfile
import unittest
from pathlib import Path

from src.models.forecast_risk_service import (
    ForecastHeatHazardService,
    DEFAULT_FORECAST_CACHE_PATH,
)
from src.models.heat_hazard_engine import ALERT_TIERS


class TestForecastHeatHazardService(unittest.TestCase):
    """Test suite verifying forecast validation, timestamp preservation,
    alert tier counts, error reporting on invalid data, and non-destructive caching.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.output_file = self.temp_path / "heat_hazard_forecast_test.json"
        self.service = ForecastHeatHazardService(output_file_path=self.output_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_real_cached_forecast_all_168_hours(self):
        """Verify processing of real cached forecast: all 168 hours processed with preserved timestamps."""
        self.assertTrue(DEFAULT_FORECAST_CACHE_PATH.is_file())
        payload = self.service.process_forecast()

        meta = payload["metadata"]
        self.assertEqual(meta["total_forecast_hours"], 168)
        self.assertEqual(meta["valid_records_count"], 168)
        self.assertEqual(meta["invalid_records_count"], 0)

        # Verify hourly records
        records = payload["hourly_forecast_hazard_records"]
        self.assertEqual(len(records), 168)
        self.assertEqual(records[0]["time"], "2026-10-10T00:00")
        self.assertEqual(records[-1]["time"], "2026-10-16T23:00")

        # Verify alert tier counts exist
        tier_counts = meta["alert_tier_distribution"]
        self.assertGreater(sum(tier_counts.values()), 0)
        self.assertEqual(sum(tier_counts.values()), 168)

        # Check peak hazard metrics
        peak = meta["peak_hazard_observation"]
        self.assertIsNotNone(peak)
        self.assertIn("hazard_temperature_c", peak)
        self.assertGreater(peak["hazard_temperature_c"], 30.0)

    def test_save_report_json_output(self):
        """Verify report serialization to JSON with proper metadata and scientific notices."""
        payload = self.service.run()
        self.assertTrue(self.output_file.is_file())

        with open(self.output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("scientific_limitations_and_disclaimer", data)
        self.assertIn("hourly_forecast_hazard_records", data)
        self.assertIn("forecast_nature", data["scientific_limitations_and_disclaimer"])
        self.assertIn("biometeorological_scope", data["scientific_limitations_and_disclaimer"])

    def test_invalid_and_missing_fields_fixture(self):
        """Verify that missing or physically invalid records are marked as Unknown with reasons."""
        mock_raw = {
            "latitude": 13.08,
            "longitude": 80.28,
            "timezone": "Asia/Kolkata",
            "hourly": {
                "time": [
                    "2026-10-10T00:00",
                    "2026-10-10T01:00",
                    "2026-10-10T02:00",
                    "2026-10-10T03:00",
                ],
                "temperature_2m": [
                    32.0,            # Valid
                    None,            # Missing temp
                    "bad_temp",      # Non-numeric
                    120.0,           # Out of physical bounds
                ],
                "relative_humidity_2m": [
                    65.0,            # Valid
                    70.0,
                    60.0,
                    50.0,
                ],
                "wind_speed_10m": [10.0, 10.0, 10.0, 10.0],
            }
        }

        payload = self.service.process_forecast(raw_data=mock_raw)
        meta = payload["metadata"]
        records = payload["hourly_forecast_hazard_records"]

        self.assertEqual(meta["total_forecast_hours"], 4)
        self.assertEqual(meta["valid_records_count"], 1)
        self.assertEqual(meta["invalid_records_count"], 3)

        # Record 0 (Valid)
        self.assertTrue(records[0]["valid"])
        self.assertIsNotNone(records[0]["hazard_temperature_c"])
        self.assertNotEqual(records[0]["alert_tier"], ALERT_TIERS["UNKNOWN"])

        # Record 1 (Missing temperature)
        self.assertFalse(records[1]["valid"])
        self.assertIn("Missing air temperature", records[1]["validation_notes"])
        self.assertEqual(records[1]["alert_tier"], ALERT_TIERS["UNKNOWN"])

        # Record 2 (Non-numeric temperature)
        self.assertFalse(records[2]["valid"])
        self.assertIn("Non-numeric", records[2]["validation_notes"])
        self.assertEqual(records[2]["alert_tier"], ALERT_TIERS["UNKNOWN"])

        # Record 3 (Out of physical bounds)
        self.assertFalse(records[3]["valid"])
        self.assertIn("out of physical range", records[3]["validation_notes"])
        self.assertEqual(records[3]["alert_tier"], ALERT_TIERS["UNKNOWN"])

    def test_repeat_runs_without_modifying_source_data(self):
        """Verify idempotence and that source forecast file is never touched or modified."""
        source_mtime_before = DEFAULT_FORECAST_CACHE_PATH.stat().st_mtime
        source_size_before = DEFAULT_FORECAST_CACHE_PATH.stat().st_size

        # Run 1
        res1 = self.service.run()
        # Run 2
        res2 = self.service.run()

        source_mtime_after = DEFAULT_FORECAST_CACHE_PATH.stat().st_mtime
        source_size_after = DEFAULT_FORECAST_CACHE_PATH.stat().st_size

        # Source cache must be completely identical
        self.assertEqual(source_mtime_before, source_mtime_after)
        self.assertEqual(source_size_before, source_size_after)

        # Results must be consistent
        self.assertEqual(res1["metadata"]["total_forecast_hours"], res2["metadata"]["total_forecast_hours"])
        self.assertEqual(res1["metadata"]["valid_records_count"], res2["metadata"]["valid_records_count"])


if __name__ == "__main__":
    unittest.main()
