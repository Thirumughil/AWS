"""Unit tests for the ClimateGuard AI Chennai Heat Hazard Dashboard."""

import hashlib
import json
import shutil
import tempfile
import threading
import time
import urllib.request
import unittest
from pathlib import Path

from src.dashboard.report_loader import (
    HeatHazardReportLoader,
    filter_forecast_records,
    DEFAULT_REPORT_PATH,
)
from src.dashboard.server import create_server, STATIC_DIR


class TestHeatHazardReportLoader(unittest.TestCase):
    """Test suite for reading and validating heat hazard forecast reports."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def test_load_real_forecast_report(self):
        """Verify the existing generated forecast report loads accurately."""
        self.assertTrue(DEFAULT_REPORT_PATH.exists(), f"Default report {DEFAULT_REPORT_PATH} must exist.")
        loader = HeatHazardReportLoader(DEFAULT_REPORT_PATH)
        result = loader.load_report()

        self.assertEqual(result["status"], "SUCCESS")
        self.assertIsNotNone(result["data"])
        self.assertIn("metadata", result["data"])
        self.assertIn("scientific_limitations_and_disclaimer", result["data"])
        self.assertIn("hourly_forecast_hazard_records", result["data"])

        # Check metadata fields
        meta = result["data"]["metadata"]
        self.assertIn("generation_timestamp_utc", meta)
        self.assertIn("alert_tier_distribution", meta)
        self.assertGreater(meta["total_forecast_hours"], 0)
        self.assertEqual(meta["total_forecast_hours"], len(result["data"]["hourly_forecast_hazard_records"]))

        # Check available dates & tiers
        self.assertGreater(len(result["available_dates"]), 0)
        self.assertIn("Green (Normal)", result["available_tiers"])
        self.assertIn("Yellow (Caution)", result["available_tiers"])

    def test_load_missing_report_file(self):
        """Ensure missing report gracefully returns an ERROR status without fabricating data."""
        non_existent = Path(self.temp_dir) / "does_not_exist.json"
        loader = HeatHazardReportLoader(non_existent)
        result = loader.load_report()

        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["data"])
        self.assertIn("Report file not found", result["message"])
        self.assertEqual(result["available_dates"], [])
        self.assertEqual(result["available_tiers"], [])

    def test_load_malformed_json_file(self):
        """Ensure malformed JSON gracefully returns ERROR without crashing."""
        broken_file = Path(self.temp_dir) / "corrupt.json"
        with open(broken_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json: missing_quotes, ]")

        loader = HeatHazardReportLoader(broken_file)
        result = loader.load_report()

        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["data"])
        self.assertIn("malformed JSON", result["message"])

    def test_load_missing_required_keys(self):
        """Ensure report missing top-level or metadata schema keys is rejected."""
        incomplete_file = Path(self.temp_dir) / "incomplete.json"
        with open(incomplete_file, "w", encoding="utf-8") as f:
            json.dump({"metadata": {}}, f)

        loader = HeatHazardReportLoader(incomplete_file)
        result = loader.load_report()

        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["data"])
        self.assertIn("missing required top-level key", result["message"])

    def test_source_file_immutability(self):
        """Verify that loading the report is strictly read-only and leaves the file untouched."""
        self.assertTrue(DEFAULT_REPORT_PATH.exists())

        def get_file_hash(p: Path) -> str:
            with open(p, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        hash_before = get_file_hash(DEFAULT_REPORT_PATH)
        loader = HeatHazardReportLoader(DEFAULT_REPORT_PATH)
        _ = loader.load_report()
        hash_after = get_file_hash(DEFAULT_REPORT_PATH)

        self.assertEqual(hash_before, hash_after, "Source report file must not be modified when read.")

    def test_filter_forecast_records_by_tier(self):
        """Test record filtering by alert tier."""
        mock_records = [
            {"time": "2026-10-11T00:00", "alert_tier": "GREEN"},
            {"time": "2026-10-11T06:00", "alert_tier": "YELLOW"},
            {"time": "2026-10-11T12:00", "alert_tier": "ORANGE"},
            {"time": "2026-10-11T14:00", "alert_tier": "RED"},
            {"time": "2026-10-12T00:00", "alert_tier": "GREEN"},
        ]

        green_only = filter_forecast_records(mock_records, alert_tier="GREEN")
        self.assertEqual(len(green_only), 2)
        self.assertTrue(all(r["alert_tier"] == "GREEN" for r in green_only))

        yellow_only = filter_forecast_records(mock_records, alert_tier="YELLOW")
        self.assertEqual(len(yellow_only), 1)

        all_records = filter_forecast_records(mock_records, alert_tier="ALL")
        self.assertEqual(len(all_records), 5)

    def test_filter_forecast_records_by_date(self):
        """Test record filtering by forecast date."""
        mock_records = [
            {"time": "2026-10-11T00:00", "alert_tier": "GREEN"},
            {"time": "2026-10-11T12:00", "alert_tier": "ORANGE"},
            {"time": "2026-10-12T00:00", "alert_tier": "GREEN"},
            {"time": "2026-10-12T12:00", "alert_tier": "RED"},
        ]

        day11 = filter_forecast_records(mock_records, date_str="2026-10-11")
        self.assertEqual(len(day11), 2)
        self.assertTrue(all(r["time"].startswith("2026-10-11") for r in day11))

        day12 = filter_forecast_records(mock_records, date_str="2026-10-12")
        self.assertEqual(len(day12), 2)

        day13 = filter_forecast_records(mock_records, date_str="2026-10-13")
        self.assertEqual(len(day13), 0)

    def test_filter_forecast_records_combined(self):
        """Test record filtering with combined tier and date parameters."""
        mock_records = [
            {"time": "2026-10-11T00:00", "alert_tier": "GREEN"},
            {"time": "2026-10-11T12:00", "alert_tier": "ORANGE"},
            {"time": "2026-10-12T00:00", "alert_tier": "GREEN"},
            {"time": "2026-10-12T12:00", "alert_tier": "ORANGE"},
        ]

        filtered = filter_forecast_records(mock_records, alert_tier="ORANGE", date_str="2026-10-11")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["time"], "2026-10-11T12:00")
        self.assertEqual(filtered[0]["alert_tier"], "ORANGE")


class TestDashboardHTTPServer(unittest.TestCase):
    """Test suite for the Dashboard HTTP server and API endpoints."""

    @classmethod
    def setUpClass(cls):
        # Bind to port 0 (OS picks an ephemeral available port)
        cls.server = create_server(host="127.0.0.1", port=0, report_path=DEFAULT_REPORT_PATH)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)  # small startup delay

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_root_serves_html(self):
        """Verify GET / returns 200 and the index.html content."""
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get_content_type(), "text/html")
            content = resp.read().decode("utf-8")
            self.assertIn("ClimateGuard AI — Chennai Heat Hazard Dashboard", content)
            self.assertIn("Scientific Advisory &amp; Anti-Fabrication Notice", content)

    def test_get_api_health(self):
        """Verify GET /api/health returns 200 with service status."""
        url = f"http://127.0.0.1:{self.port}/api/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "HEALTHY")

    def test_get_api_report(self):
        """Verify GET /api/report returns 200 with full forecast report."""
        url = f"http://127.0.0.1:{self.port}/api/report"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "SUCCESS")
            self.assertIn("data", data)
            self.assertIn("metadata", data["data"])
            self.assertEqual(data["data"]["metadata"]["total_forecast_hours"], 168)


if __name__ == "__main__":
    unittest.main()
