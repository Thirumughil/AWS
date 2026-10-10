#!/usr/bin/env python
"""
T10 (part A): buildings and facilities per hex for the Chennai grid.
Run from the project root:

    python src/hex_exposure.py             # download from OpenStreetMap (cached), build the tables
    python src/hex_exposure.py --upload    # also copy the outputs to S3
    python src/hex_exposure.py --refetch   # ignore the cache and download again

Outputs (data/processed):
    chennai_hex_exposure.csv   one row per hex: building cover, facility counts, distance to the nearest facility
    chennai_facilities.csv     one row per OSM facility (type, name, lat, lon, in_city)
Raw downloads are cached in cache/osm/, so a re-run takes seconds.

Distances are straight-line from the hex centre, not walking distances.
No API key is needed. Overpass is shared and rate limited, so the script waits between requests.
Needs: numpy pandas geopandas pyproj scipy requests h3
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "processed" / "chennai_h3_res9.geojson"
PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "cache" / "osm"

H3_RES = 9
UTM = 32644                                   # UTM zone 44N, used for areas and distances
TILES_X, TILES_Y = 4, 6                       # the city is downloaded in tiles of about 5 x 7 km
PAD = 0.001                                   # degrees of padding around the grid
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "ClimateGuard-AI-hackathon/0.1 (student project)"}
PAUSE_S = 6                                   # wait between tiles
TRIES = 3

SMALL_BLDG_M2 = 40                            # footprint below this counts as a "small building" (proxy only)
COOLING_TYPES = ["community_centre", "library", "school"]   # assumption: confirm with the team

# what the city screen counted, for a rough comparison
SCREEN = {"building": 276231, "hospital": 650, "school": 577, "community_centre": 55,
          "library": 92, "park": 417, "water": 33}

PROFILE = os.environ.get("AWS_PROFILE", "climateguard")
S3_BUCKET = os.environ.get("S3_BUCKET", "urban-heat-risk-sai-001")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = os.environ.get("S3_PREFIX", f"s3://{S3_BUCKET}/processed/")


V4 = hasattr(h3, "latlng_to_cell")
TO_XY = Transformer.from_crs(4326, UTM, always_xy=True)


def cell(lat, lon):
    return h3.latlng_to_cell(lat, lon, H3_RES) if V4 else h3.geo_to_h3(lat, lon, H3_RES)


# ----------------------------------------------------------------- Overpass
def overpass(query):
    last = None
    for attempt in range(TRIES):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=300)
            if r.status_code == 200:
                j = r.json()
                remark = j.get("remark", "")
                if "runtime error" in remark or "timed out" in remark:
                    raise RuntimeError(remark[:120])       # partial answer: do not trust it
                return j
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        wait = 15 * (attempt + 1)
        print(f"    retry {attempt + 1}/{TRIES} in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"Overpass failed after {TRIES} tries: {last}")


def tile_boxes(grid):
    minx, miny = grid["lon"].min() - PAD, grid["lat"].min() - PAD
    maxx, maxy = grid["lon"].max() + PAD, grid["lat"].max() + PAD
    xs = np.linspace(minx, maxx, TILES_X + 1)
    ys = np.linspace(miny, maxy, TILES_Y + 1)
    out = []
    for j in range(TILES_Y):
        for i in range(TILES_X):
            s, w, n, e = ys[j], xs[i], ys[j + 1], xs[i + 1]
            has_hex = ((grid["lon"] >= w) & (grid["lon"] <= e) & (grid["lat"] >= s) & (grid["lat"] <= n)).any()
            if has_hex:
                out.append((j * TILES_X + i, (s, w, n, e)))
    return out


def fetch_buildings_tile(k, bbox, refetch, name=None, depth=0):
    name = name or f"{k:02d}"
    f = CACHE / f"buildings_tile_{name}.csv"
    if f.exists() and not refetch:
        return pd.read_csv(f)
    s, w, n, e = bbox
    try:
        j = overpass(f'[out:json][timeout:240];way["building"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});out geom;')
    except RuntimeError as err:
        if depth >= 2:
            raise
        print(f"    tile {name} is too heavy for Overpass right now, splitting it into 4 smaller pieces")
        time.sleep(30)
        ym, xm = (s + n) / 2, (w + e) / 2
        subs = [(s, w, ym, xm), (s, xm, ym, e), (ym, w, n, xm), (ym, xm, n, e)]
        pieces = []
        for q, sb in enumerate(subs):
            pieces.append(fetch_buildings_tile(k, sb, refetch, f"{name}_{q}", depth + 1))
            time.sleep(PAUSE_S)
        df = pd.concat(pieces, ignore_index=True).drop_duplicates("osm_id")
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, index=False)
        print(f"    tile {name}: {len(df):,} buildings from the 4 pieces")
        return df
    rows, skipped = [], 0
    for el in j.get("elements", []):
        g = el.get("geometry")
        if el.get("type") != "way" or not g or len(g) < 4 or g[0] != g[-1]:
            skipped += 1
            continue
        lon = np.array([p["lon"] for p in g])
        lat = np.array([p["lat"] for p in g])
        x, y = TO_XY.transform(lon, lat)
        area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        rows.append((el["id"], lat[:-1].mean(), lon[:-1].mean(), area))
    df = pd.DataFrame(rows, columns=["osm_id", "lat", "lon", "area_m2"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    print(f"    {len(df):,} buildings ({skipped} unclosed or empty ways skipped)")
    return df


def fetch_facilities(bbox, refetch):
    f = CACHE / "facilities.csv"
    if f.exists() and not refetch:
        return pd.read_csv(f)
    s, w, n, e = bbox
    box = f"({s:.5f},{w:.5f},{n:.5f},{e:.5f})"
    j = overpass('[out:json][timeout:240];('
                 f'nwr["amenity"~"^(hospital|clinic|school|community_centre|library|drinking_water)$"]{box};'
                 f'nwr["leisure"~"^(park|garden)$"]{box};'
                 ');out center tags;')
    rows = []
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat") if "lat" in el else el.get("center", {}).get("lat")
        lon = el.get("lon") if "lon" in el else el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        if tags.get("amenity") == "drinking_water":
            ftype = "water"
        elif "amenity" in tags:
            ftype = tags["amenity"]
        else:
            ftype = "park"
        rows.append((f"{el['type']}/{el['id']}", ftype, tags.get("name", ""), lat, lon))
    df = pd.DataFrame(rows, columns=["facility_id", "type", "name", "lat", "lon"])
    df = df.drop_duplicates(subset=["type", "lat", "lon"]).reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    if not GRID.exists():
        sys.exit(f"Missing grid: {GRID}")
    g = gpd.read_file(GRID)
    grid = pd.DataFrame(g.drop(columns="geometry"))[["h3_id", "lat", "lon", "area_km2"]].reset_index(drop=True)
    ids = set(grid["h3_id"])
    tiles = tile_boxes(grid)
    print(f"{len(grid)} hexes | {len(tiles)} building tiles | cache: {CACHE}")

    # ---------------------------------------------------------- buildings
    parts = []
    for n_done, (k, bbox) in enumerate(tiles, 1):
        cached = (CACHE / f"buildings_tile_{k:02d}.csv").exists() and not args.refetch
        print(f"[{n_done}/{len(tiles)}] buildings, tile {k}{' (cached)' if cached else ''}")
        parts.append(fetch_buildings_tile(k, bbox, args.refetch))
        if not cached and n_done < len(tiles):
            time.sleep(PAUSE_S)
    raw = pd.concat(parts, ignore_index=True)
    b = raw.drop_duplicates("osm_id").copy()
    b["h3_id"] = [cell(la, lo) for la, lo in zip(b["lat"], b["lon"])]
    inside = b[b["h3_id"].isin(ids)].copy()
    inside["small"] = inside["area_m2"] < SMALL_BLDG_M2
    agg = inside.groupby("h3_id").agg(bldg_count=("osm_id", "size"),
                                      bldg_footprint_m2=("area_m2", "sum"),
                                      small_bldg_frac=("small", "mean")).reset_index()
    grid = grid.merge(agg, on="h3_id", how="left")
    grid["bldg_count"] = grid["bldg_count"].fillna(0).astype(int)
    grid["bldg_footprint_m2"] = grid["bldg_footprint_m2"].fillna(0).round(1)
    grid["bldg_cover_frac"] = (grid["bldg_footprint_m2"] / (grid["area_km2"] * 1e6)).round(4)
    grid["small_bldg_frac"] = grid["small_bldg_frac"].round(3)

    print(f"\nBuildings: {len(raw):,} downloaded, {len(b):,} after removing tile-edge duplicates, "
          f"{len(inside):,} inside the hex grid")
    ratio = len(inside) / SCREEN["building"]
    print(f"  city screen counted {SCREEN['building']:,} -> {ratio:.0%} "
          f"{'OK' if ratio >= 0.85 else 'LOW: re-run with --refetch, a tile may have been cut short'}")
    print(f"  hexes with no buildings: {int((grid['bldg_count'] == 0).sum())} | "
          f"building cover: mean {grid['bldg_cover_frac'].mean():.2f}, "
          f"median {grid['bldg_cover_frac'].median():.2f}, max {grid['bldg_cover_frac'].max():.2f}")
    if grid["bldg_cover_frac"].max() > 0.9:
        print("  CHECK: a hex has more than 90% building cover, look for duplicate or overlapping footprints")

    # ---------------------------------------------------------- facilities
    print("\nFacilities (one query for the whole city)")
    s = grid["lat"].min() - PAD
    w = grid["lon"].min() - PAD
    n = grid["lat"].max() + PAD
    e = grid["lon"].max() + PAD
    fac = fetch_facilities((s, w, n, e), args.refetch)
    fac["h3_id"] = [cell(la, lo) for la, lo in zip(fac["lat"], fac["lon"])]
    fac["in_city"] = fac["h3_id"].isin(ids)

    types = ["hospital", "clinic", "school", "community_centre", "library", "park", "water"]
    for t in types:
        cnt = fac[(fac["type"] == t) & fac["in_city"]].groupby("h3_id").size()
        grid[f"n_{t}"] = grid["h3_id"].map(cnt).fillna(0).astype(int)

    hx, hy = TO_XY.transform(grid["lon"].values, grid["lat"].values)
    centres = np.c_[hx, hy]
    groups = {"hospital": ["hospital"], "clinic": ["clinic"], "school": ["school"],
              "park": ["park"], "water": ["water"], "cooling": COOLING_TYPES}
    for name, tlist in groups.items():
        sub = fac[fac["type"].isin(tlist)]
        if sub.empty:
            grid[f"nearest_{name}_m"] = np.nan
            continue
        fx, fy = TO_XY.transform(sub["lon"].values, sub["lat"].values)
        d, _ = cKDTree(np.c_[fx, fy]).query(centres)
        grid[f"nearest_{name}_m"] = d.round(0)

    print(f"  {'type':<17}{'in city':>8}{'city screen':>13}")
    for t in types:
        cnt = int(((fac["type"] == t) & fac["in_city"]).sum())
        ref = SCREEN.get(t)
        print(f"  {t:<17}{cnt:>8}{(str(ref) if ref else '-'):>13}")
    print("  (parks here include leisure=garden, so the park count can exceed the screen's)")
    n_cool = int((fac["type"].isin(COOLING_TYPES) & fac["in_city"]).sum())
    print(f"  cooling candidates ({', '.join(COOLING_TYPES)}): {n_cool} in city")
    for name in ("hospital", "cooling", "water", "park"):
        col = grid[f"nearest_{name}_m"]
        print(f"  nearest {name}: median {col.median():.0f} m | 90th percentile {col.quantile(0.9):.0f} m | max {col.max():.0f} m")
    if int(((fac['type'] == 'water') & fac['in_city']).sum()) < 100:
        print("  NOTE: very few mapped drinking-water points, so water-point sites will need labelled synthetic candidates")

    # ---------------------------------------------------------- save
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = ["h3_id", "bldg_count", "bldg_footprint_m2", "bldg_cover_frac", "small_bldg_frac"] + \
           [f"n_{t}" for t in types] + [f"nearest_{x}_m" for x in groups]
    out1 = PROCESSED / "chennai_hex_exposure.csv"
    out2 = PROCESSED / "chennai_facilities.csv"
    grid[cols].to_csv(out1, index=False)
    fac[["facility_id", "type", "name", "lat", "lon", "in_city"]].to_csv(out2, index=False)
    print(f"\nSaved {out1} ({len(grid)} rows, {len(cols)} columns)\nSaved {out2} ({len(fac)} rows)")

    if args.upload:
        for f in (out1, out2):
            cmd = ["aws", "s3", "cp", str(f), S3_PREFIX + f.name, "--profile", PROFILE, "--region", AWS_REGION]
            try:
                subprocess.run(cmd, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as err:
                print("Upload problem:", err, "\n  run: " + " ".join(cmd))
    print("\nDone.")


if __name__ == "__main__":
    main()