from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORECAST_FILE = ROOT / "data/raw/chennai_forecast.csv"
PATTERN_FILE = ROOT / "data/processed/chennai_hex_lst_pattern.csv"
RISK_FILE = ROOT / "data/processed/chennai_hex_risk.csv"
OUTPUT_FILE = ROOT / "data/processed/chennai_hex_hazard_forecast.csv"
TIMEZONE = "Asia/Kolkata"
DEFAULT_FACTOR = 0.3


def nws_heat_index_c(temp_c, rh):
    """Estimate heat index using the NWS Rothfusz regression in Celsius."""
    t = np.asarray(temp_c, dtype=float) * 9.0 / 5.0 + 32.0
    rh = np.asarray(rh, dtype=float)

    simple = 0.5 * (
        t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094
    )
    simple = (simple + t) / 2.0

    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 6.83783e-3 * t**2
        - 5.481717e-2 * rh**2
        + 1.22874e-3 * t**2 * rh
        + 8.5282e-4 * t * rh**2
        - 1.99e-6 * t**2 * rh**2
    )

    low = (rh < 13) & (t >= 80) & (t <= 112)
    hi[low] -= (
        ((13 - rh[low]) / 4)
        * np.sqrt((17 - np.abs(t[low] - 95)) / 17)
    )

    high = (rh > 85) & (t >= 80) & (t <= 87)
    hi[high] += (
        ((rh[high] - 85) / 10)
        * ((87 - t[high]) / 5)
    )

    # The regression is used only where the simple HI reaches 80 F.
    result_f = np.where(simple >= 80, hi, simple)
    return (result_f - 32) * 5.0 / 9.0


def stull_wet_bulb_c(temp_c, rh):
    """Stull approximate wet-bulb temperature, with RH constrained to 5-99%."""
    t = np.asarray(temp_c, dtype=float)
    humidity = np.clip(np.asarray(rh, dtype=float), 5, 99)

    return (
        t * np.arctan(0.151977 * np.sqrt(humidity + 8.313659))
        + np.arctan(t + humidity)
        - np.arctan(humidity - 1.676331)
        + 0.00391838 * humidity**1.5
        * np.arctan(0.023101 * humidity)
        - 4.686035
    )


def heat_index_category(hi_c):
    """Standard NWS heat-index bands, converted from Fahrenheit."""
    hi_f = np.asarray(hi_c) * 9 / 5 + 32
    return np.select(
        [hi_f < 80, hi_f < 90, hi_f < 103, hi_f < 125],
        ["Below Caution", "Caution", "Extreme Caution", "Danger"],
        default="Extreme Danger",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the existing local output intentionally."
    )
    parser.add_argument(
        "--include-past", action="store_true",
        help="Keep past forecast timestamps for retrospective inspection."
    )
    args = parser.parse_args()

    if OUTPUT_FILE.exists() and not args.overwrite:
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Inspect it first, or use --overwrite to replace it.")
        return 1

    for path in (FORECAST_FILE, PATTERN_FILE, RISK_FILE):
        if not path.exists():
            print(f"Required input missing: {path}")
            return 1

    forecast = pd.read_csv(FORECAST_FILE)
    pattern = pd.read_csv(PATTERN_FILE)
    risk = pd.read_csv(RISK_FILE)

    required = {
        "time", "temperature_2m", "relative_humidity_2m",
        "apparent_temperature"
    }
    missing = required - set(forecast.columns)
    if missing:
        print(f"Forecast columns missing: {sorted(missing)}")
        return 1

    if not {"h3_id", "anomaly_cv"}.issubset(pattern.columns):
        print("LST pattern needs h3_id and anomaly_cv.")
        return 1
    if "h3_id" not in risk.columns:
        print("Risk dataset needs h3_id.")
        return 1

    # Open-Meteo was requested with timezone=Asia/Kolkata.
    # Keep its timestamps as local wall-clock times.
    forecast["time_local"] = pd.to_datetime(
        forecast["time"], errors="coerce"
    )

    if forecast["time_local"].isna().any():
        print("Invalid forecast timestamps found; stopping.")
        return 1

    for col in [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature"
    ]:
        forecast[col] = pd.to_numeric(forecast[col], errors="coerce")

    forecast = forecast.dropna(
        subset=[
            "temperature_2m", "relative_humidity_2m",
            "apparent_temperature"
        ]
    ).copy()

    if forecast.empty:
        print("No valid forecast rows found.")
        return 1
    if not forecast["relative_humidity_2m"].between(0, 100).all():
        print("Relative humidity outside 0-100%; stopping.")
        return 1
    if forecast["time_local"].duplicated().any():
        print("Duplicate forecast timestamps found; stopping.")
        return 1

    original_hours = len(forecast)
    if not args.include_past:
        now_local = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
        current_hour = now_local.replace(minute=0, second=0, microsecond=0)
        forecast = forecast[forecast["time_local"] >= current_hour].copy()

        print(f"Current Chennai-local hour: {current_hour}")
        print(f"Past forecast rows excluded: {original_hours - len(forecast)}")

    if forecast.empty:
        print(
            "No future forecast hours remain. Refresh the forecast file "
            "before running T17; no output was written."
        )
        return 1

    pattern = pattern[["h3_id", "anomaly_cv"]].copy()
    pattern["h3_id"] = pattern["h3_id"].astype(str)
    pattern["anomaly_cv"] = pd.to_numeric(
        pattern["anomaly_cv"], errors="coerce"
    )

    if pattern["h3_id"].duplicated().any():
        print("Duplicate H3 IDs in LST pattern; stopping.")
        return 1
    if pattern["anomaly_cv"].isna().any():
        print("Missing LST anomaly values; stopping.")
        return 1

    risk_ids = set(risk["h3_id"].astype(str))
    pattern_ids = set(pattern["h3_id"])
    if risk_ids != pattern_ids:
        print(
            f"H3 ID mismatch: {len(risk_ids - pattern_ids)} risk hexes "
            f"missing patterns; {len(pattern_ids - risk_ids)} extra patterns."
        )
        return 1

    forecast = forecast.sort_values("time_local").reset_index(drop=True)
    forecast["_join"] = 1
    pattern["_join"] = 1
    out = forecast.merge(pattern, on="_join", how="inner").drop(
        columns="_join"
    )

    # The factor is a modelling assumption, not a measured air-temperature offset.
    out["local_temp_adjustment_c"] = DEFAULT_FACTOR * out["anomaly_cv"]
    out["local_temperature_c"] = (
        out["temperature_2m"] + out["local_temp_adjustment_c"]
    )

    out["heat_index_c"] = nws_heat_index_c(
        out["temperature_2m"].to_numpy(),
        out["relative_humidity_2m"].to_numpy()
    )
    out["local_heat_index_c"] = nws_heat_index_c(
        out["local_temperature_c"].to_numpy(),
        out["relative_humidity_2m"].to_numpy()
    )
    out["wet_bulb_c"] = stull_wet_bulb_c(
        out["temperature_2m"].to_numpy(),
        out["relative_humidity_2m"].to_numpy()
    )
    out["local_wet_bulb_c"] = stull_wet_bulb_c(
        out["local_temperature_c"].to_numpy(),
        out["relative_humidity_2m"].to_numpy()
    )

    out["hazard_score"] = out.groupby("time_local")[
        "local_heat_index_c"
    ].rank(method="average", pct=True)

    out["heat_index_category"] = heat_index_category(
        out["local_heat_index_c"].to_numpy()
    )
    out["timezone"] = TIMEZONE

    columns = [
        "time_local", "timezone", "h3_id",
        "temperature_2m", "relative_humidity_2m",
        "apparent_temperature", "anomaly_cv",
        "local_temp_adjustment_c", "local_temperature_c",
        "heat_index_c", "local_heat_index_c",
        "wet_bulb_c", "local_wet_bulb_c",
        "hazard_score", "heat_index_category"
    ]

    out["time_local"] = out["time_local"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out[columns].to_csv(OUTPUT_FILE, index=False, float_format="%.4f")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Forecast hours used: {len(forecast)}")
    print(f"H3 hexagons: {len(pattern)}")
    print(f"Output rows: {len(out):,}")
    print(f"Missing local heat index: {out['local_heat_index_c'].isna().sum()}")
    print(f"Missing local wet bulb: {out['local_wet_bulb_c'].isna().sum()}")

    print("\nHeat-index categories across all hex-hour records:")
    print(out["heat_index_category"].value_counts().to_string())

    print("\nSensitivity to local adjustment factor:")
    for factor in (0.0, 0.3, 0.5):
        adjusted = (
            np.repeat(forecast["temperature_2m"].to_numpy(), len(pattern))
            + factor * np.tile(pattern["anomaly_cv"].to_numpy(), len(forecast))
        )
        humidity = np.repeat(
            forecast["relative_humidity_2m"].to_numpy(), len(pattern)
        )
        hi = nws_heat_index_c(adjusted, humidity)
        print(
            f"factor={factor:.1f}: mean={np.mean(hi):.2f} C, "
            f"P95={np.percentile(hi, 95):.2f} C, max={np.max(hi):.2f} C"
        )

    print(
        "\nNote: hazard_score is a relative percentile within each hour, "
        "not an absolute probability of heat illness."
    )
    print("No S3 upload was performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
