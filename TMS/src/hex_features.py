#!/usr/bin/env python
"""
T09: per-hex surface temperature, NDVI and land cover for the Chennai grid.
Run from the project root (needs data/processed/chennai_h3_res9.geojson and data/raw/chennai_scenes.csv).

    python src/hex_features.py                          # the two backtest dates only
    python src/hex_features.py --dates 2025-04-10        # specific Landsat dates
    python src/hex_features.py --all                     # every clear Landsat scene in chennai_scenes.csv
    python src/hex_features.py --upload                  # also copy the CSVs to S3

Outputs (data/processed):
    chennai_hex_lst.csv        h3_id, date, lst_c, lst_valid_frac, ndvi_l8     (one row per hex per date)
    chennai_hex_ndvi_s2.csv    h3_id, date, ndvi_s2, s2_valid_frac, s2_scene_date
    chennai_hex_landcover.csv  h3_id, built_frac, tree_frac, grass_shrub_frac, crop_frac, bare_frac, water_frac, other_frac
Maps: reports/chennai_lst_<date>.html

Landsat files sit in a requester-pays bucket, so your AWS profile is used for those reads.
Needs: geopandas pandas numpy rasterio boto3 pystac-client (folium for maps)
"""
import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.session import AWSSession
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as win_bounds, from_bounds

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "processed" / "chennai_h3_res9.geojson"
SCENES = ROOT / "data" / "raw" / "chennai_scenes.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = {"landsat": "landsat-c2-l2", "sentinel2": "sentinel-2-c1-l2a"}
DEFAULT_DATES = ["2026-05-21", "2024-05-31"]          # backtest days
S2_MAX_GAP_DAYS = 7                                   # nearest Sentinel-2 scene allowed
MIN_COVER = 0.95
MAX_CLOUD_ALL = 20                                    # for --all

PROFILE = os.environ.get("AWS_PROFILE", "climateguard")
WORLDCOVER_URL = "s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_N12E078_Map.tif"
S3_PREFIX = "s3://urban-heat-risk-sai-001/processed/"
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"

LST_DEFAULT = (0.00341802, 149.0)                     # Landsat C2 L2 surface temperature, to kelvin
LANDSAT_REFL_DEFAULT = (0.0000275, -0.2)
S2_REFL_DEFAULT = (0.0001, -0.1)
S2_BAD_SCL = [0, 1, 3, 8, 9, 10, 11]                  # no data, saturated, shadow, cloud, cirrus, snow

WC_CLASSES = {"built_frac": [50], "tree_frac": [10], "grass_shrub_frac": [20, 30],
              "crop_frac": [40], "bare_frac": [60], "water_frac": [80, 90, 95]}


# ----------------------------------------------------------------- raster helpers
def paid_env():
    """Reads from requester-pays buckets (Landsat) with your AWS profile."""
    sess = boto3.Session(profile_name=PROFILE, region_name="us-west-2")
    return rasterio.Env(AWSSession(sess, requester_pays=True),
                        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                        GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2")


def open_env():
    """Reads public buckets (WorldCover) without credentials."""
    return rasterio.Env(AWSSession(aws_unsigned=True), AWS_REGION="eu-central-1",
                        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                        GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2")


def asset_url(asset):
    alt = (asset.extra_fields or {}).get("alternate", {})
    return alt.get("s3", {}).get("href") or asset.href


def scale_offset(asset, default):
    rb = (asset.extra_fields or {}).get("raster:bands")
    if rb and "scale" in rb[0]:
        return rb[0]["scale"], rb[0].get("offset", 0.0)
    return default


def read_window(url, bounds_ll):
    """Read the part of a raster covering the city. Returns (array, transform, crs)."""
    with rasterio.open(url) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bounds_ll)
        w = from_bounds(*b, transform=src.transform)
        win = Window(int(np.floor(w.col_off)), int(np.floor(w.row_off)),
                     int(np.ceil(w.width)) + 1, int(np.ceil(w.height)) + 1)
        win = win.intersection(Window(0, 0, src.width, src.height))
        return src.read(1, window=win), src.window_transform(win), src.crs


def read_like(url, transform, shape, crs):
    """Read a coarser raster (Sentinel-2 SCL, 20 m) onto the grid of a finer one."""
    with rasterio.open(url) as src:
        b = win_bounds(Window(0, 0, shape[1], shape[0]), transform)
        win = from_bounds(*b, transform=src.transform)
        return src.read(1, window=win, out_shape=shape, resampling=Resampling.nearest,
                        boundless=True, fill_value=0)


_label_cache = {}


def hex_labels(grid, transform, shape, crs):
    """Raster of hex numbers (1..n, 0 = outside) on the given pixel grid."""
    key = (tuple(transform)[:6], shape, crs.to_wkt())
    if key not in _label_cache:
        g = grid.to_crs(crs.to_wkt())
        shapes = ((geom, i + 1) for i, geom in enumerate(g.geometry))
        _label_cache[key] = rasterize(shapes, out_shape=shape, transform=transform,
                                      fill=0, dtype="int32")
    return _label_cache[key]


def zonal_mean(values, valid, labels, n):
    """Mean of `values` per hex over valid pixels, plus the valid share of each hex's pixels."""
    inside = labels > 0
    total = np.bincount(labels[inside], minlength=n + 1).astype(float)
    m = valid & inside
    cnt = np.bincount(labels[m], minlength=n + 1).astype(float)
    sm = np.bincount(labels[m], weights=values[m].astype(float), minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, sm / cnt, np.nan)
        frac = np.where(total > 0, cnt / total, 0.0)
    return mean[1:], frac[1:]


# ----------------------------------------------------------------- land cover
def landcover(grid, bounds_ll):
    with open_env():
        data, tr, crs = read_window(WORLDCOVER_URL, bounds_ll)
    labels = hex_labels(grid, tr, data.shape, crs)
    n = len(grid)
    inside = labels > 0
    total = np.bincount(labels[inside], minlength=n + 1).astype(float)[1:]
    out = pd.DataFrame({"h3_id": grid["h3_id"].values})
    used = np.zeros(n)
    for name, codes in WC_CLASSES.items():
        m = inside & np.isin(data, codes)
        cnt = np.bincount(labels[m], minlength=n + 1).astype(float)[1:]
        out[name] = np.where(total > 0, cnt / total, np.nan).round(4)
        used += np.nan_to_num(out[name].values)
    out["other_frac"] = np.clip(1 - used, 0, 1).round(4)
    return out


# ----------------------------------------------------------------- Landsat
def landsat_features(item, grid, bounds_ll):
    a = item.assets
    with paid_env():
        arr = {}
        for key in ("lwir11", "red", "nir08", "qa_pixel"):
            arr[key], tr, crs = read_window(asset_url(a[key]), bounds_ll)
    shapes = {v.shape for v in arr.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"Landsat bands have different shapes: {shapes}")
    labels = hex_labels(grid, tr, arr["lwir11"].shape, crs)
    n = len(grid)

    bad = (arr["qa_pixel"] & 0b11111) != 0                   # fill, dilated cloud, cirrus, cloud, shadow
    s, o = scale_offset(a["lwir11"], LST_DEFAULT)
    lst_c = arr["lwir11"].astype(float) * s + o - 273.15
    lst_ok = (arr["lwir11"] > 0) & ~bad & (lst_c > 10) & (lst_c < 80)

    sr, orr = scale_offset(a["red"], LANDSAT_REFL_DEFAULT)
    sn, on = scale_offset(a["nir08"], LANDSAT_REFL_DEFAULT)
    red = np.clip(arr["red"].astype(float) * sr + orr, 0, 1)
    nir = np.clip(arr["nir08"].astype(float) * sn + on, 0, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    ndvi_ok = (arr["red"] > 0) & (arr["nir08"] > 0) & ~bad & ((nir + red) > 0.02) & np.isfinite(ndvi)
    ndvi = np.clip(ndvi, -1, 1)

    lst_mean, lst_frac = zonal_mean(lst_c, lst_ok, labels, n)
    ndvi_mean, _ = zonal_mean(np.nan_to_num(ndvi), ndvi_ok, labels, n)
    return pd.DataFrame({"h3_id": grid["h3_id"].values, "lst_c": lst_mean.round(2),
                         "lst_valid_frac": lst_frac.round(3), "ndvi_l8": ndvi_mean.round(3)})


# ----------------------------------------------------------------- Sentinel-2
def sentinel_features(item, grid, bounds_ll):
    a = item.assets
    with paid_env():
        red, tr, crs = read_window(asset_url(a["red"]), bounds_ll)
        nir, _, _ = read_window(asset_url(a["nir"]), bounds_ll)
        scl = read_like(asset_url(a["scl"]), tr, red.shape, crs)
    labels = hex_labels(grid, tr, red.shape, crs)
    n = len(grid)
    sr, orr = scale_offset(a["red"], S2_REFL_DEFAULT)
    sn, on = scale_offset(a["nir"], S2_REFL_DEFAULT)
    r = np.clip(red.astype(float) * sr + orr, 0, 1)
    nr = np.clip(nir.astype(float) * sn + on, 0, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nr - r) / (nr + r)
    ok = (red > 0) & (nir > 0) & ~np.isin(scl, S2_BAD_SCL) & ((nr + r) > 0.02) & np.isfinite(ndvi)
    ndvi = np.clip(ndvi, -1, 1)
    mean, frac = zonal_mean(np.nan_to_num(ndvi), ok, labels, n)
    return pd.DataFrame({"h3_id": grid["h3_id"].values, "ndvi_s2": mean.round(3),
                         "s2_valid_frac": frac.round(3)})


# ----------------------------------------------------------------- misc
def get_item(cat, sensor, item_id):
    return next(cat.search(collections=[COLLECTION[sensor]], ids=[item_id]).items())


def append_csv(path, new, keys):
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last")
    new.to_csv(path, index=False)
    return new


def make_map(grid, df, date):
    try:
        g = grid[["h3_id", "geometry"]].merge(df, on="h3_id")
        m = g.explore(column="lst_c", cmap="inferno", legend=True, tiles=ESRI, attr="Tiles (c) Esri",
                      style_kwds={"weight": 0, "fillOpacity": 0.75})
        out = REPORTS / f"chennai_lst_{date}.html"
        m.save(str(out))
        print(f"  map: {out}")
    except Exception as e:
        print("  map skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", help="Landsat dates, YYYY-MM-DD")
    ap.add_argument("--all", action="store_true", help="every clear Landsat scene in the scene list")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--redo", action="store_true", help="re-process dates already in the output files")
    args = ap.parse_args()

    for p in (GRID, SCENES):
        if not p.exists():
            sys.exit(f"Missing input: {p}")
    grid = gpd.read_file(GRID).reset_index(drop=True)
    bounds_ll = tuple(grid.total_bounds)
    scenes = pd.read_csv(SCENES, parse_dates=["date"])
    ls = scenes[(scenes["sensor"] == "landsat") & (scenes["covers_city"] >= MIN_COVER)]
    s2 = scenes[(scenes["sensor"] == "sentinel2") & (scenes["covers_city"] >= MIN_COVER)]
    if args.all:
        dates = sorted(ls[ls["cloud_pct"] < MAX_CLOUD_ALL]["date"].dt.strftime("%Y-%m-%d"))
    else:
        dates = args.dates or DEFAULT_DATES
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    lst_path = PROCESSED / "chennai_hex_lst.csv"
    s2_path = PROCESSED / "chennai_hex_ndvi_s2.csv"

    done = set()
    if lst_path.exists() and not args.redo:
        done = set(pd.read_csv(lst_path, usecols=["date"])["date"].astype(str))
    todo = [d for d in dates if d not in done]
    print(f"{len(grid)} hexes | {len(dates)} date(s): {len(dates) - len(todo)} already saved, "
          f"{len(todo)} to do | AWS profile {PROFILE}")
    cat = Client.open(STAC_URL)

    print("\nLand cover (ESA WorldCover) ...")
    lc = landcover(grid, bounds_ll)
    lc.to_csv(PROCESSED / "chennai_hex_landcover.csv", index=False)
    print(f"  built-up share of the city: {(lc['built_frac'] * grid['area_km2']).sum() / grid['area_km2'].sum():.1%} | "
          f"tree: {(lc['tree_frac'] * grid['area_km2']).sum() / grid['area_km2'].sum():.1%}")

    try:
        for i, d in enumerate(todo, 1):
            row = ls[ls["date"] == d]
            if row.empty:
                print(f"\n[{i}/{len(todo)}] {d}: no clear Landsat scene in the scene list, skipping")
                continue
            print(f"\n[{i}/{len(todo)}] {d}: Landsat {row.iloc[0]['id']} (cloud {row.iloc[0]['cloud_pct']}%)")
            try:
                item = get_item(cat, "landsat", row.iloc[0]["id"])
                df = landsat_features(item, grid, bounds_ll)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                print("  If this is a 403 / AccessDenied, your IAM user cannot read the requester-pays Landsat bucket.")
                continue
            df.insert(1, "date", d)
            append_csv(lst_path, df, ["h3_id", "date"])              # saved straight away
            print(f"  LST: mean {df['lst_c'].mean():.1f} C | min {df['lst_c'].min():.1f} | max {df['lst_c'].max():.1f} | "
                  f"hexes without data: {int(df['lst_c'].isna().sum())} | mean valid share {df['lst_valid_frac'].mean():.0%}")
            cand = s2.assign(gap=(s2["date"] - pd.Timestamp(d)).abs().dt.days)
            cand = cand[cand["gap"] <= S2_MAX_GAP_DAYS].sort_values(["gap", "cloud_pct"])
            if cand.empty:
                print(f"  no Sentinel-2 scene within {S2_MAX_GAP_DAYS} days, NDVI from Landsat only")
            else:
                try:
                    srow = cand.iloc[0]
                    sd = srow["date"].strftime("%Y-%m-%d")
                    sdf = sentinel_features(get_item(cat, "sentinel2", srow["id"]), grid, bounds_ll)
                    sdf.insert(1, "date", d)
                    sdf["s2_scene_date"] = sd
                    append_csv(s2_path, sdf, ["h3_id", "date"])
                    print(f"  Sentinel-2 {sd} ({int(srow['gap'])} days away): NDVI mean {sdf['ndvi_s2'].mean():.2f}")
                except Exception as e:
                    print(f"  Sentinel-2 failed: {type(e).__name__}: {e}")
            if d in DEFAULT_DATES or args.dates:                      # maps only for the dates you care about
                make_map(grid, df, d)
    except KeyboardInterrupt:
        print("\nStopped. Everything finished so far is saved. Run the same command again to continue.")

    if not lst_path.exists():
        sys.exit("\nNo Landsat scene processed. See the errors above.")

    # per-date summary: how usable is each scene, and does it behave as expected
    x = pd.read_csv(lst_path).merge(lc, on="h3_id")
    rows = []
    for d, g in x.groupby("date"):
        g = g.dropna(subset=["lst_c"])
        rows.append({"date": d, "mean_lst_c": round(g["lst_c"].mean(), 1),
                     "valid_share": round(x[x["date"] == d]["lst_valid_frac"].mean(), 2),
                     "corr_built": round(g["lst_c"].corr(g["built_frac"]), 2),
                     "corr_ndvi": round(g["lst_c"].corr(g["ndvi_l8"]), 2)})
    summ = pd.DataFrame(rows)
    summ.to_csv(REPORTS / "chennai_scene_summary.csv", index=False)
    print(f"\nScene summary ({len(summ)} dates saved; {int((summ['valid_share'] >= 0.8).sum())} have at least 80% valid pixels)")
    print(summ.to_string(index=False))

    if args.upload:
        for name in ("chennai_hex_lst.csv", "chennai_hex_ndvi_s2.csv", "chennai_hex_landcover.csv"):
            f = PROCESSED / name
            if f.exists():
                cmd = ["aws", "s3", "cp", str(f), S3_PREFIX + name, "--profile", PROFILE]
                try:
                    subprocess.run(cmd, check=True)
                except (FileNotFoundError, subprocess.CalledProcessError) as e:
                    print("Upload problem:", e, "\n  run: " + " ".join(cmd))
    print("\nDone.")


if __name__ == "__main__":
    main()
