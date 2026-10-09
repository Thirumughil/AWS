#!/usr/bin/env python
"""
Chennai pre-Phase-1 checks. Run from the project root (T:\\AWS Project):

    python src/chennai_checks.py plot      # boundary area, H3 hex estimate, map file
    python src/chennai_checks.py sat       # wider Landsat / Sentinel-2 search
    python src/chennai_checks.py weather   # forecast + ERA5 history + backtest days
    python src/chennai_checks.py all

Needs: geopandas shapely pandas requests pystac-client
Optional for the map: folium mapclassify matplotlib
"""
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]          # script lives in <project>/src
CITY = "chennai_tamil_nadu_india"
BOUNDARY = ROOT / "data" / "raw" / f"{CITY}_boundary.geojson"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

UTM = 32644                      # UTM zone 44N, used only for measuring area
TZ = "Asia/Kolkata"

# satellite search
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTIONS = {"landsat": "landsat-c2-l2", "sentinel2": "sentinel-2-c1-l2a"}
MONTHS = range(3, 9)             # March to August
YEARS_BACK = 4
MAX_CLOUD = 40                   # scene-level cloud %, mask cloudy pixels later
MIN_COVER = 0.95                 # scene must cover this share of the city

# weather
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,relative_humidity_2m,apparent_temperature"
ERA5_FIRST_YEAR = 2015


# ----------------------------------------------------------------- helpers
def load_city():
    if not BOUNDARY.exists():
        sys.exit(f"Boundary file not found: {BOUNDARY}")
    return gpd.read_file(BOUNDARY).to_crs(4326)


def union(g):
    geom = g.geometry
    return geom.union_all() if hasattr(geom, "union_all") else geom.unary_union


def get_json(url, params, tries=4):
    for i in range(tries):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 200:
            return r.json()
        print(f"  HTTP {r.status_code} (try {i + 1}/{tries}): {r.text[:150]}")
        time.sleep(8 * (i + 1))
    raise RuntimeError(f"Request failed: {url}")


def nws_heat_index_c(t_c, rh):
    """US National Weather Service heat index (Rothfusz), in degrees C."""
    t = t_c * 9 / 5 + 32
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (simple + t) / 2
    if hi >= 80:
        hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
              - 6.83783e-3 * t * t - 5.481717e-2 * rh * rh + 1.22874e-3 * t * t * rh
              + 8.5282e-4 * t * rh * rh - 1.99e-6 * t * t * rh * rh)
        if rh < 13 and 80 <= t <= 112:
            hi -= ((13 - rh) / 4) * ((17 - abs(t - 95)) / 17) ** 0.5
        elif rh > 85 and 80 <= t <= 87:
            hi += ((rh - 85) / 10) * ((87 - t) / 5)
    return (hi - 32) * 5 / 9


# ----------------------------------------------------------------- step 1
def plot():
    g = load_city()
    area = g.to_crs(UTM).area.sum() / 1e6
    minx, miny, maxx, maxy = g.total_bounds
    print(f"Area: {area:.1f} km2 | features: {len(g)} | types: {g.geom_type.unique().tolist()}")
    print(f"Bounds: lon {minx:.3f} to {maxx:.3f}, lat {miny:.3f} to {maxy:.3f}")
    for res, hex_km2 in ((8, 0.7373), (9, 0.1053)):
        print(f"H3 resolution {res}: about {area / hex_km2:,.0f} hexes")

    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        m = g.explore(style_kwds={"fillOpacity": 0.2})
        out = REPORTS / "chennai_boundary.html"
        m.save(str(out))
        print(f"Saved {out}  (open it in a browser and compare with the city)")
    except Exception as e:                       # folium/mapclassify missing
        print("Interactive map failed:", e)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = g.plot(edgecolor="black", alpha=0.4, figsize=(7, 7))
        ax.set_title(f"{CITY}  {area:.0f} km2")
        out = REPORTS / "chennai_boundary.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")


# ----------------------------------------------------------------- step 2
def sat():
    from pystac_client import Client
    from shapely.geometry import mapping, shape

    g = load_city()
    city = union(g)
    simple = city.simplify(0.002)
    today = date.today()
    start = f"{today.year - YEARS_BACK}-03-01"
    end = today.isoformat()
    cat = Client.open(STAC_URL)

    rows = []
    for sensor, coll in COLLECTIONS.items():
        print(f"Searching {coll} {start} to {end}, cloud < {MAX_CLOUD}% ...")
        search = cat.search(
            collections=[coll],
            intersects=mapping(simple),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
            max_items=3000,
        )
        for it in search.items():
            if it.datetime.month not in MONTHS:
                continue
            p = it.properties
            if sensor == "landsat" and p.get("platform") not in (None, "landsat-8", "landsat-9"):
                continue                         # skip Landsat 7 (striping)
            cover = shape(it.geometry).intersection(city).area / city.area
            rows.append({
                "sensor": sensor,
                "id": it.id,
                "date": it.datetime.date().isoformat(),
                "cloud_pct": p.get("eo:cloud_cover"),
                "platform": p.get("platform"),
                "tile": p.get("grid:code") or f"{p.get('landsat:wrs_path')}/{p.get('landsat:wrs_row')}",
                "covers_city": round(cover, 3),
            })

    if not rows:
        sys.exit("No scenes found. Try raising MAX_CLOUD or YEARS_BACK.")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "chennai_scenes.csv"
    df.to_csv(out, index=False)

    print(f"\nSaved {len(df)} scenes to {out}\n")
    for sensor, sub in df.groupby("sensor"):
        ok = sub[sub["covers_city"] >= MIN_COVER]
        print(f"== {sensor}: {len(sub)} found, {len(ok)} cover >= {MIN_COVER:.0%} of the city, "
              f"{ok['date'].nunique()} distinct dates")
        print("   cloud < 20%:", int((ok["cloud_pct"] < 20).sum()),
              "| 20-40%:", int(((ok["cloud_pct"] >= 20) & (ok["cloud_pct"] < 40)).sum()))
        print("   per year:", ok.groupby(ok["date"].dt.year).size().to_dict())
        print("   per month:", ok.groupby(ok["date"].dt.month).size().to_dict())


# ----------------------------------------------------------------- step 3
def weather():
    g = load_city()
    c = gpd.GeoSeries([union(g.to_crs(UTM)).centroid], crs=UTM).to_crs(4326).iloc[0]
    lat, lon = round(c.y, 4), round(c.x, 4)
    print(f"City centroid: {lat}, {lon}")
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # forecast, including apparent temperature
    f = get_json(FORECAST_URL, {"latitude": lat, "longitude": lon, "hourly": HOURLY,
                                "forecast_days": 7, "timezone": TZ})
    fdf = pd.DataFrame(f["hourly"])
    fdf["time"] = pd.to_datetime(fdf["time"])
    fdf.to_csv(RAW / "chennai_forecast.csv", index=False)
    print(f"Forecast: {len(fdf)} hours | apparent temp max {fdf['apparent_temperature'].max():.1f} C"
          f" | air temp max {fdf['temperature_2m'].max():.1f} C")

    # ERA5 history, one request per year
    last = date.today() - timedelta(days=7)      # archive lags a few days
    frames = []
    for y in range(ERA5_FIRST_YEAR, last.year + 1):
        end = min(date(y, 12, 31), last)
        print(f"  ERA5 {y} ...")
        j = get_json(ARCHIVE_URL, {"latitude": lat, "longitude": lon, "hourly": HOURLY,
                                   "start_date": f"{y}-01-01", "end_date": end.isoformat(),
                                   "timezone": TZ})
        frames.append(pd.DataFrame(j["hourly"]))
        time.sleep(1)
    h = pd.concat(frames, ignore_index=True)
    h["time"] = pd.to_datetime(h["time"])
    h = h.dropna(subset=["temperature_2m", "relative_humidity_2m", "apparent_temperature"])
    h["heat_index"] = [nws_heat_index_c(t, rh)
                       for t, rh in zip(h["temperature_2m"], h["relative_humidity_2m"])]
    h.to_csv(RAW / "chennai_era5_hourly.csv", index=False)

    h["day"] = h["time"].dt.normalize()
    daily = h.groupby("day").agg(tmax=("temperature_2m", "max"),
                                 apparent_max=("apparent_temperature", "max"),
                                 heat_index_max=("heat_index", "max"),
                                 rh_mean=("relative_humidity_2m", "mean")).round(1)
    daily.to_csv(PROCESSED / "chennai_era5_daily.csv")

    p95 = daily["tmax"].quantile(0.95)
    print(f"\nERA5 daily: {len(daily)} days | P95 of daily max air temp = {p95:.1f} C")
    top = daily.sort_values("apparent_max", ascending=False).head(15).copy()

    scenes_csv = RAW / "chennai_scenes.csv"
    if scenes_csv.exists():                      # nearest clear Landsat date
        s = pd.read_csv(scenes_csv, parse_dates=["date"])
        s = s[(s["sensor"] == "landsat") & (s["covers_city"] >= MIN_COVER)]
        if len(s):
            top["nearest_landsat_days"] = [int((s["date"] - d).abs().dt.days.min())
                                           for d in top.index]
    top.to_csv(REPORTS / "chennai_backtest_candidates.csv")
    print("\nTop 15 days by apparent temperature (backtest candidates):")
    print(top.to_string())
    print("\nSaved reports/chennai_backtest_candidates.csv")


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    steps = {"plot": plot, "sat": sat, "weather": weather}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "all":
        for fn in steps.values():
            fn()
    elif cmd in steps:
        steps[cmd]()
    else:
        sys.exit(f"Usage: python src/chennai_checks.py [{'|'.join(steps)}|all]")
