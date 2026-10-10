"""Automated tests for the Biometeorological Heat Hazard Engine."""

import math
import json
import unittest
from pathlib import Path

from src.models.heat_hazard_engine import (
    calculate_water_vapor_pressure,
    calculate_steadman_apparent_temperature,
    calculate_noaa_heat_index,
    classify_heat_alert_tier,
    BiometeorologicalHazardEngine,
    ALERT_TIERS,
)
from src.ingestion.config import CACHE_DIR


class TestHeatHazardEngine(unittest.TestCase):
    """Test suite validating biometeorological calculations against independent
    published reference tables, humidity adjustments, edge cases, and forecast integration.
    """

    def setUp(self):
        self.engine = BiometeorologicalHazardEngine()

    def test_water_vapor_pressure_magnus(self):
        """Verify Magnus vapor pressure against psychrometric reference at 30°C and 60% RH.
        Reference: Saturation vapor pressure at 30°C is ~42.4 hPa -> e = 0.60 * 42.4 ~ 25.4 hPa.
        """
        vp = calculate_water_vapor_pressure(30.0, 60.0)
        self.assertAlmostEqual(vp, 25.37, delta=0.2)

    def test_noaa_heat_index_warm_humid_reference(self):
        """Reference Case A: 90°F (32.22°C) at 70% RH.
        Published NOAA NWS Heat Index table value: 106°F (41.11°C).
        """
        temp_c = (90.0 - 32.0) * 5.0 / 9.0  # 32.222°C
        hi_c = calculate_noaa_heat_index(temp_c, 70.0)
        expected_c = (106.0 - 32.0) * 5.0 / 9.0  # 41.111°C
        self.assertAlmostEqual(hi_c, expected_c, delta=0.6)

    def test_noaa_heat_index_extreme_heat_reference(self):
        """Reference Case B: 96°F (35.56°C) at 65% RH.
        Published NOAA NWS Heat Index table value: 121°F (49.44°C).
        """
        temp_c = (96.0 - 32.0) * 5.0 / 9.0  # 35.556°C
        hi_c = calculate_noaa_heat_index(temp_c, 65.0)
        expected_c = (121.0 - 32.0) * 5.0 / 9.0  # 49.444°C
        self.assertAlmostEqual(hi_c, expected_c, delta=0.6)

    def test_noaa_heat_index_moderate_reference(self):
        """Reference Case C: 80°F (26.67°C) at 40% RH.
        Published NOAA NWS Heat Index table value: 80°F (26.67°C).
        """
        temp_c = (80.0 - 32.0) * 5.0 / 9.0
        hi_c = calculate_noaa_heat_index(temp_c, 40.0)
        self.assertAlmostEqual(hi_c, 26.67, delta=0.5)

    def test_noaa_heat_index_low_humidity_adjustment(self):
        """Reference Case D: Low humidity adjustment where RH < 13% and 80°F <= T <= 112°F.
        Condition: 100°F (37.78°C) at 10% RH.
        Published NOAA NWS table value: 95°F (35.00°C).
        """
        temp_c = (100.0 - 32.0) * 5.0 / 9.0
        hi_c = calculate_noaa_heat_index(temp_c, 10.0)
        expected_c = (95.0 - 32.0) * 5.0 / 9.0
        self.assertAlmostEqual(hi_c, expected_c, delta=0.8)

    def test_noaa_heat_index_high_humidity_adjustment(self):
        """Reference Case E: High humidity adjustment where RH > 85% and 80°F <= T <= 87°F.
        Condition: 84°F (28.89°C) at 90% RH.
        Published NOAA NWS table value: 98°F (36.67°C).
        """
        temp_c = (84.0 - 32.0) * 5.0 / 9.0
        hi_c = calculate_noaa_heat_index(temp_c, 90.0)
        expected_c = (98.0 - 32.0) * 5.0 / 9.0  # 36.67°C
        self.assertAlmostEqual(hi_c, expected_c, delta=0.5)

    def test_steadman_apparent_temperature_with_wind(self):
        """Verify Steadman's Apparent Temperature and convective cooling by wind.
        T = 30°C, RH = 60%:
          e = 25.37 hPa.
          At v = 0 m/s: AT = 30 + (0.33 * 25.37) - 4.0 = 34.37°C.
          At v = 5 m/s: AT = 34.37 - (0.70 * 5) = 30.87°C (cooling effect).
        """
        at_calm = calculate_steadman_apparent_temperature(30.0, 60.0, wind_speed_mps=0.0)
        self.assertAlmostEqual(at_calm, 34.37, delta=0.2)

        at_wind = calculate_steadman_apparent_temperature(30.0, 60.0, wind_speed_mps=5.0)
        self.assertAlmostEqual(at_wind, 30.87, delta=0.2)
        self.assertLess(at_wind, at_calm)

    def test_invalid_and_missing_inputs(self):
        """Verify invalid or physically impossible values yield NaN / Unknown without crashing."""
        self.assertTrue(math.isnan(calculate_noaa_heat_index(None, 60.0)))
        self.assertTrue(math.isnan(calculate_noaa_heat_index(35.0, None)))
        self.assertTrue(math.isnan(calculate_noaa_heat_index(-50.0, 60.0)))
        self.assertTrue(math.isnan(calculate_noaa_heat_index(35.0, 150.0)))  # RH > 100%
        self.assertTrue(math.isnan(calculate_steadman_apparent_temperature(35.0, -10.0)))

        self.assertEqual(classify_heat_alert_tier(None), ALERT_TIERS["UNKNOWN"])
        self.assertEqual(classify_heat_alert_tier(float("nan")), ALERT_TIERS["UNKNOWN"])

    def test_alert_tier_boundary_thresholds(self):
        """Verify alert tier boundaries (Green < 32°C, Yellow 32-41°C, Orange 41-54°C, Red >= 54°C)."""
        self.assertEqual(classify_heat_alert_tier(31.9), ALERT_TIERS["GREEN"])
        self.assertEqual(classify_heat_alert_tier(32.0), ALERT_TIERS["YELLOW"])
        self.assertEqual(classify_heat_alert_tier(40.9), ALERT_TIERS["YELLOW"])
        self.assertEqual(classify_heat_alert_tier(41.0), ALERT_TIERS["ORANGE"])
        self.assertEqual(classify_heat_alert_tier(53.9), ALERT_TIERS["ORANGE"])
        self.assertEqual(classify_heat_alert_tier(54.0), ALERT_TIERS["RED"])
        self.assertEqual(classify_heat_alert_tier(60.0), ALERT_TIERS["RED"])

    def test_forecast_cache_processing_integration(self):
        """Verify end-to-end processing of real cached 168-hour forecast file."""
        cache_path = CACHE_DIR / "weather_chennai_forecast_168h.json"
        self.assertTrue(cache_path.is_file(), f"Forecast cache not found at {cache_path}")

        mtime_before = cache_path.stat().st_mtime
        res = self.engine.process_cached_forecast_file(cache_file_path=cache_path)
        mtime_after = cache_path.stat().st_mtime

        # Verify source file was NOT modified
        self.assertEqual(mtime_before, mtime_after)

        self.assertEqual(res["total_timesteps"], 168)
        self.assertEqual(res["valid_timesteps"], 168)
        self.assertEqual(res["missing_timesteps"], 0)
        self.assertGreater(res["max_hazard_c"], 30.0)

        # Check records schema and timestamp preservation
        first_rec = res["records"][0]
        self.assertIn("time", first_rec)
        self.assertIn("air_temperature_c", first_rec)
        self.assertIn("steadman_apparent_temp_c", first_rec)
        self.assertIn("noaa_heat_index_c", first_rec)
        self.assertIn("alert_tier", first_rec)


if __name__ == "__main__":
    unittest.main()
