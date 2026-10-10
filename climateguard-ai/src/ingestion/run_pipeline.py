"""CLI Pipeline runner for Chennai Data Ingestion and Quality Validation Gate."""

import sys
import json
from pathlib import Path
from src.ingestion.quality_validator import ChennaiQualityValidator, DEFAULT_REPORT_PATH


def main():
    print("=" * 72)
    print("  CLIMATEGUARD AI — CHENNAI DATA INGESTION & QUALITY PIPELINE")
    print("=" * 72)

    validator = ChennaiQualityValidator()
    print("\n[+] Running centralized quality validator across all data layers...")
    report = validator.generate_full_report()

    meta = report["report_metadata"]
    print(f"\n[Summary] Overall Status: {meta['overall_validation_status']}")
    print(f"          Layers Passing: {meta['layers_passing']} / {meta['layers_evaluated']}")
    print(f"          Timestamp:      {meta['generation_timestamp_utc']}")

    print("\n[Layer Statuses]")
    for key, item in report["datasets"].items():
        name = item.get("dataset_name", key)
        status = item.get("status", "UNKNOWN")
        scope = item.get("coverage_scope", "N/A")
        records = item.get("record_count", 0)
        print(f"  * {name:<42} : [{status}] (Scope: {scope}, Records: {records})")

    print(f"\n[+] Full quality report written to:")
    print(f"    {DEFAULT_REPORT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
