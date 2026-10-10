"""Integration tests for ClimateGuard AI Chennai Ingestion Pipeline."""

import os
import json
import unittest
from pathlib import Path

from src.ingestion.config import (
    CHENNAI_BOUNDARY_PATH,
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    CHENNAI_BBOX,
    DATA_PROCESSED_DIR,
)
from src.ingestion.boundary_loader import ChennaiBoundaryLoader
from src.ingestion.weather_ingestor import WeatherIngestor
from src.ingestion.facilities_ingestor import FacilitiesIngestor
from src.ingestion.building_ingestor import BuildingFootprintIngestor
from src.ingestion.quality_validator import ChennaiQualityValidator, DEFAULT_REPORT_PATH


class TestIngestionIntegration(unittest.TestCase):
    """End-to-end integration tests verifying cross-module workflow, schema
    compatibility, idempotency, and accurate quality reporting.
    """

    @classmethod
    def setUpClass(cls):
        cls.validator = ChennaiQualityValidator()
        cls.report_path = DEFAULT_REPORT_PATH

    def test_complete_local_workflow(self):
        """Step 1: Execute complete local workflow across all ingestion modules."""
        # 1. Load boundary
        b_loader = ChennaiBoundaryLoader()
        b_res = b_loader.load_and_validate()
        self.assertEqual(b_res["status"], "PASS")
        self.assertAlmostEqual(b_res["area_km2"], 433.71, delta=5.0)

        # 2. Ingest weather forecast
        w_ingestor = WeatherIngestor()
        w_res = w_ingestor.ingest_and_validate()
        self.assertEqual(w_res["status"], "PASS")
        self.assertEqual(len(w_res["df"]), 168)

        # 3. Ingest facilities using boundary polygon
        f_ingestor = FacilitiesIngestor()
        f_res = f_ingestor.ingest_and_validate(boundary_polygon=b_res["geometry"])
        self.assertEqual(f_res["status"], "PASS")
        self.assertGreaterEqual(f_res["counts"]["hospitals"], 5)

        # 4. Ingest sample building footprints
        bld_ingestor = BuildingFootprintIngestor()
        bld_res = bld_ingestor.load_and_validate_local_sample()
        self.assertEqual(bld_res["status"], "PASS")
        self.assertGreater(bld_res["valid_count"], 25000)

        # 5. Run quality validator & write report
        report = self.validator.generate_full_report()
        self.assertTrue(self.report_path.is_file())
        self.assertEqual(report["report_metadata"]["target_pilot_city"], "Chennai, Tamil Nadu, India")

    def test_data_contracts_and_schema_compatibility(self):
        """Step 2: Verify cross-layer spatial compatibility and coordinate systems."""
        b_loader = ChennaiBoundaryLoader()
        b_res = b_loader.load_and_validate()
        boundary_geom = b_res["geometry"]

        bld_ingestor = BuildingFootprintIngestor()
        bld_res = bld_ingestor.load_and_validate_local_sample(max_features=500)
        bld_bounds = bld_res["report"]["bounding_box"]

        # Verify building sample coordinates fall within administrative boundary bounding box
        self.assertGreaterEqual(bld_bounds["min_lon"], CHENNAI_BBOX["west"])
        self.assertLessEqual(bld_bounds["max_lon"], CHENNAI_BBOX["east"])
        self.assertGreaterEqual(bld_bounds["min_lat"], CHENNAI_BBOX["south"])
        self.assertLessEqual(bld_bounds["max_lat"], CHENNAI_BBOX["north"])

        # Check CRS conventions
        self.assertEqual(b_res["gdf"].crs.to_string(), GEOGRAPHIC_CRS)
        self.assertEqual(b_res["gdf_projected"].crs.to_string(), PROJECTED_CRS)

    def test_missing_datasets_and_coverage_scopes(self):
        """Step 3: Verify strict reporting of missing datasets and sample boundaries."""
        report = self.validator.generate_full_report()
        datasets = report["datasets"]

        # Building footprints MUST be strictly classified as sample_only
        self.assertEqual(datasets["building_footprints"]["coverage_scope"], "sample_only")
        has_sample_warning = any(
            "sample area" in w.lower() for w in datasets["building_footprints"]["warnings"]
        )
        self.assertTrue(has_sample_warning)

        # Population and Landsat rasters MUST be marked as MISSING_LOCAL_FILE
        self.assertEqual(datasets["population_vulnerability_rasters"]["status"], "MISSING_LOCAL_FILE")
        self.assertEqual(datasets["population_vulnerability_rasters"]["coverage_scope"], "not_downloaded")

        self.assertEqual(datasets["thermal_satellite_rasters"]["status"], "MISSING_LOCAL_FILE")
        self.assertEqual(datasets["thermal_satellite_rasters"]["coverage_scope"], "not_downloaded")

        # Zero fake measurements fabricated
        self.assertFalse(datasets["thermal_satellite_rasters"]["checks"]["st_b10_band_present"])

    def test_workflow_idempotency_run_twice(self):
        """Step 4: Verify that running the workflow consecutively produces identical results without corruption."""
        report_run_1 = self.validator.generate_full_report()
        mtime_1 = self.report_path.stat().st_mtime

        report_run_2 = self.validator.generate_full_report()

        # Check key record counts match exactly
        self.assertEqual(
            report_run_1["report_metadata"]["layers_passing"],
            report_run_2["report_metadata"]["layers_passing"]
        )
        self.assertEqual(
            report_run_1["datasets"]["weather_forecast"]["record_count"],
            report_run_2["datasets"]["weather_forecast"]["record_count"]
        )
        self.assertEqual(
            report_run_1["datasets"]["critical_facilities"]["record_count"],
            report_run_2["datasets"]["critical_facilities"]["record_count"]
        )
        self.assertEqual(
            report_run_1["datasets"]["building_footprints"]["record_count"],
            report_run_2["datasets"]["building_footprints"]["record_count"]
        )

        # Verify cached execution occurred
        w_ingestor = WeatherIngestor()
        self.assertTrue(w_ingestor.is_cache_valid())
        f_ingestor = FacilitiesIngestor()
        self.assertTrue(f_ingestor.has_cached_data())

    def test_quality_report_json_validity_and_contents(self):
        """Step 5: Verify that generated report is valid JSON with complete audit keys."""
        self.assertTrue(self.report_path.is_file())
        with open(self.report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "report_metadata",
            "spatial_reference_systems",
            "datasets",
            "coverage_scope_summary",
            "key_warnings",
            "unresolved_data_quality_issues",
        ]
        for key in required_keys:
            self.assertIn(key, data)

        self.assertIn("overall_validation_status", data["report_metadata"])
        self.assertIn("administrative_boundary", data["datasets"])
        self.assertIn("weather_forecast", data["datasets"])
        self.assertIn("critical_facilities", data["datasets"])
        self.assertIn("building_footprints", data["datasets"])


if __name__ == "__main__":
    unittest.main()
