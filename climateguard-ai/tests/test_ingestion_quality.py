"""Automated tests for Chennai Data Quality and Validation Gate."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.quality_validator import ChennaiQualityValidator
from src.ingestion.config import (
    CHENNAI_BOUNDARY_PATH,
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
)


class TestChennaiQualityValidator(unittest.TestCase):
    """Test suite for quality validator, schemas, and temporary fixtures."""

    def setUp(self):
        self.validator = ChennaiQualityValidator()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_boundary_validation(self):
        """Verify real Chennai boundary validates with expected area ~433.7 km²."""
        res = self.validator.validate_boundary(CHENNAI_BOUNDARY_PATH)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["available"])
        self.assertEqual(res["coverage_scope"], "verified_city_wide")
        self.assertAlmostEqual(res["area_km2"], 433.71, delta=5.0)
        self.assertTrue(res["checks"]["geometry_valid"])
        self.assertEqual(res["crs_identified"], "EPSG:4326")

    def test_missing_boundary_file(self):
        """Verify graceful error reporting when boundary file is missing."""
        missing_path = self.temp_path / "non_existent_boundary.geojson"
        res = self.validator.validate_boundary(missing_path)
        self.assertEqual(res["status"], "FAIL")
        self.assertFalse(res["available"])
        self.assertGreater(len(res["errors"]), 0)

    def test_invalid_boundary_geometry_fixture(self):
        """Verify invalid self-intersecting bowtie polygon fails geometry check."""
        invalid_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "Invalid Bowtie"},
                "geometry": {
                    "type": "Polygon",
                    # Bowtie self-intersection at (1.0, 1.0)
                    "coordinates": [[[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]]
                }
            }]
        }
        invalid_path = self.temp_path / "invalid_geom.geojson"
        with open(invalid_path, "w", encoding="utf-8") as f:
            json.dump(invalid_geojson, f)

        res = self.validator.validate_boundary(invalid_path)
        self.assertEqual(res["status"], "FAIL")
        self.assertFalse(res["checks"]["geometry_valid"])

    def test_weather_forecast_validation(self):
        """Verify weather validator checks 168 hours and plausible ranges."""
        res = self.validator.validate_weather_forecast()
        self.assertIn(res["status"], ["PASS", "WARN"])
        self.assertTrue(res["available"])
        self.assertGreaterEqual(res["record_count"], 168)
        self.assertTrue(res["checks"]["expected_168_hours"])
        self.assertTrue(res["checks"]["time_continuity_valid"])

    def test_weather_missing_fields_fixture(self):
        """Verify incomplete weather response missing apparent_temperature is flagged."""
        corrupt_weather = {
            "hourly": {
                "time": ["2026-10-10T00:00", "2026-10-10T01:00"],
                "temperature_2m": [30.0, 31.0],
                # Missing 'apparent_temperature' and 'relative_humidity_2m'
            }
        }
        corrupt_file = self.temp_path / "corrupt_weather.json"
        with open(corrupt_file, "w", encoding="utf-8") as f:
            json.dump(corrupt_weather, f)

        res = self.validator.validate_weather_forecast(weather_cache_file=corrupt_file)
        self.assertEqual(res["status"], "FAIL")
        self.assertGreater(len(res["errors"]), 0)

    def test_facilities_validation(self):
        """Verify critical facilities validation checks essential hospitals and schools."""
        res = self.validator.validate_facilities()
        self.assertIn(res["status"], ["PASS", "WARN"])
        self.assertTrue(res["checks"]["essential_hospitals_present"])
        self.assertTrue(res["checks"]["essential_schools_present"])

    def test_building_sample_scope_and_validation(self):
        """Verify building validator marks sample dataset strictly as 'sample_only'."""
        res = self.validator.validate_buildings()
        self.assertIn(res["status"], ["PASS_WITH_WARNINGS", "PASS"])
        self.assertEqual(res["coverage_scope"], "sample_only")
        self.assertEqual(res["duplicate_id_count"], 0)
        self.assertGreater(res["record_count"], 25000)
        # Verify clear coverage limitation warning exists
        has_warning = any("sample area" in w.lower() for w in res["warnings"])
        self.assertTrue(has_warning)

    def test_building_duplicate_ids_fixture(self):
        """Verify duplicate feature IDs are detected in temporary fixture."""
        dup_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"@id": "way/9999", "building": "yes"},
                    "geometry": {"type": "Polygon", "coordinates": [[[80.2, 13.0], [80.21, 13.0], [80.21, 13.01], [80.2, 13.01], [80.2, 13.0]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"@id": "way/9999", "building": "residential"},  # Duplicate ID
                    "geometry": {"type": "Polygon", "coordinates": [[[80.22, 13.0], [80.23, 13.0], [80.23, 13.01], [80.22, 13.01], [80.22, 13.0]]]}
                }
            ]
        }
        dup_path = self.temp_path / "dup_buildings.geojson"
        with open(dup_path, "w", encoding="utf-8") as f:
            json.dump(dup_geojson, f)

        res = self.validator.validate_buildings(geojson_path=dup_path)
        self.assertEqual(res["duplicate_id_count"], 1)
        self.assertFalse(res["checks"]["no_duplicate_ids"])

    def test_missing_rasters_properly_flagged(self):
        """Verify un-downloaded population and Landsat rasters are identified cleanly without crashing."""
        empty_raster_dir = self.temp_path / "empty_rasters"
        empty_raster_dir.mkdir()

        pop_res = self.validator.validate_population_rasters(population_dir=empty_raster_dir)
        self.assertEqual(pop_res["status"], "MISSING_LOCAL_FILE")
        self.assertEqual(pop_res["coverage_scope"], "not_downloaded")

        landsat_res = self.validator.validate_landsat_rasters(landsat_dir=empty_raster_dir)
        self.assertEqual(landsat_res["status"], "MISSING_LOCAL_FILE")
        self.assertEqual(landsat_res["coverage_scope"], "not_downloaded")

    def test_full_quality_report_generation(self):
        """Verify end-to-end quality report generation and output JSON writing."""
        out_report = self.temp_path / "test_report.json"
        custom_validator = ChennaiQualityValidator(output_report_path=out_report)
        report = custom_validator.generate_full_report()

        self.assertTrue(out_report.is_file())
        self.assertEqual(report["report_metadata"]["target_pilot_city"], "Chennai, Tamil Nadu, India")
        self.assertIn("datasets", report)
        self.assertIn("coverage_scope_summary", report)
        self.assertEqual(len(report["datasets"]), 6)


if __name__ == "__main__":
    unittest.main()
