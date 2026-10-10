#!/usr/bin/env python
"""
T10 (part B, HRSL route): population per hex for the Chennai grid, from Meta's High Resolution
Population Density Maps (HRSL) on AWS Open Data. Only the Chennai window is read (a few MB), so there
are no multi-GB downloads.

Put this file in <project>/src/ and run from the project root:

    python src/hex_population_hrsl.py             # build the table
    python src/hex_population_hrsl.py --upload    # also copy it to S3

First check the folder names (they are guesses for the elderly and children layers):

    aws s3 ls s3://dataforgood-fb-data/hrsl-cogs/ --no-sign-request
    python src/hex_population_hrsl.py --elderly <folder> --children <folder>

Output (data/processed): chennai_hex_population.csv
    h3_id, pop_total, pop_elderly, pop_under5, pct_elderly, pct_under5, pop_density_km2

DEFINITIONS (differ from the WorldPop plan - update docs/spec.md):
    pop_elderly = aged 60 and over   (WorldPop route was 65+)
    pop_under5  = HRSL "children 0-5" layer (WorldPop route was ages 0-4)
Each ~30 m pixel is assigned to the hex that contains its centre. Citation: Facebook Connectivity Lab
and CIESIN (Columbia University), High Resolution Settlement Layer.
Needs: numpy pandas geopandas rasterio h3
"""
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")        # public bucket, no credentials needed
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")   # where the HRSL bucket lives
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GRID = PROCESSED / "chennai_h3_res9.geojson"
BASE = "/vsis3/dataforgood-fb-data/hrsl-cogs"
PAD_DEG = 0.01
PROFILE = os.environ.get("AWS_PROFILE", "climateguard")
S3_BUCKET = os.environ.get("S3_BUCKET", "urban-heat-risk-sai-002")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = os.environ.get("S3_PREFIX", f"s3://{S3_BUCKET}/processed/")

V4 = hasattr(h3, "latlng_to_cell")


def to_cell(lat, lon, res):
    return h3.latlng_to_cell(lat, lon, res) if V4 else h3.geo_to_h3(lat, lon, res)


def hex_sums(base, folder, bounds, res):
    """Sum one HRSL layer into H3 cells. Each layer uses its own pixel grid."""
    path = f"{base}/{folder}/{folder}-latest.vrt"
    try:
        with rasterio.open(path) as src:
            w0 = from_bounds(*bounds, transform=src.transform)
            w = Window(int(np.floor(w0.col_off)), int(np.floor(w0.row_off)),
                       int(np.ceil(w0.width)) + 1, int(np.ceil(w0.height)) + 1)
            w = w.intersection(Window(0, 0, src.width, src.height))
            arr = src.read(1, window=w).astype("float64")
            nodata = src.nodata
            tr = src.window_transform(w)
    except Exception as err:
        raise SystemExit(
            f"\nCould not open {path}\n  {type(err).__name__}: {str(err)[:200]}\n"
            "Check the folder name with:\n"
            "  aws s3 ls s3://dataforgood-fb-data/hrsl-cogs/ --no-sign-request\n"
            "then pass it with --total / --elderly / --children.")
    if nodata is not None:
        arr[arr == nodata] = 0
    arr[~np.isfinite(arr)] = 0
    arr[arr < 0] = 0
    rows, cols = np.nonzero(arr > 0)
    lon = tr.c + tr.a * (cols + 0.5)
    lat = tr.f + tr.e * (rows + 0.5)
    cells = [to_cell(la, lo, res) for la, lo in zip(lat, lon)]
    return pd.Series(arr[rows, cols]).groupby(cells).sum(), arr.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--res", type=int, default=9)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--total", default="hrsl_general")
    ap.add_argument("--elderly", default="hrsl_elderly_over_sixty")     # unverified guess
    ap.add_argument("--children", default="hrsl_children_under_five")   # unverified guess
    ap.add_argument("--out", default=str(PROCESSED / "chennai_hex_population.csv"))
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()

    g = gpd.read_file(a.grid)
    col_id = "h3_id" if "h3_id" in g.columns else "h3"
    area = g["area_km2"] if "area_km2" in g.columns else None
    w, s, e, n = g.total_bounds
    bounds = (w - PAD_DEG, s - PAD_DEG, e + PAD_DEG, n + PAD_DEG)
    out = pd.DataFrame({"h3_id": g[col_id].values})
    print(f"{len(out)} hexes | reading 3 HRSL layers, Chennai window only")

    layers = (("pop_total", a.total), ("pop_elderly", a.elderly), ("pop_under5", a.children))
    for col, folder in layers:
        sums, shape = hex_sums(a.base, folder, bounds, a.res)
        out[col] = out["h3_id"].map(sums).fillna(0.0)
        print(f"  {col:12s} {folder}: window {shape[1]}x{shape[0]} px, "
              f"{out[col].sum():,.0f} people in the hexes ({sums.sum():,.0f} in the padded window)")

    out["area_km2"] = area.values if area is not None else np.nan
    ok = out["pop_total"] >= 1
    out["pct_elderly"] = np.where(ok, out["pop_elderly"] / out["pop_total"], np.nan)
    out["pct_under5"] = np.where(ok, out["pop_under5"] / out["pop_total"], np.nan)
    out["pop_density_km2"] = out["pop_total"] / out["area_km2"]

    tot = out["pop_total"].sum()
    print(f"\nTotal population in the hex grid: {tot:,.0f}")
    print(f"  elderly (60+): {out['pop_elderly'].sum():,.0f} ({out['pop_elderly'].sum() / tot:.1%}) | "
          f"under 5: {out['pop_under5'].sum():,.0f} ({out['pop_under5'].sum() / tot:.1%})")
    print(f"  hexes with no population: {int((~ok).sum())} | people per hex: "
          f"median {out['pop_total'].median():.0f}, max {out['pop_total'].max():.0f}")
    print(f"  density: median {out['pop_density_km2'].median():,.0f} per km2 | "
          f"90th percentile {out['pop_density_km2'].quantile(0.9):,.0f}")
    if out["pop_elderly"].sum() > 0.25 * tot or out["pop_under5"].sum() > 0.20 * tot:
        print("  WARNING: an age share looks too high - check the folder names for the layers.")
    ex = PROCESSED / "chennai_hex_exposure.csv"
    if ex.exists():
        m = out.merge(pd.read_csv(ex, usecols=["h3_id", "bldg_cover_frac"]), on="h3_id")
        print(f"  correlation of population with building cover: "
              f"{m['pop_total'].corr(m['bldg_cover_frac']):.2f} (expect clearly positive)")

    for col in ("pop_total", "pop_elderly", "pop_under5", "pop_density_km2"):
        out[col] = out[col].round(1)
    out["pct_elderly"] = out["pct_elderly"].round(4)
    out["pct_under5"] = out["pct_under5"].round(4)
    cols = ["h3_id", "pop_total", "pop_elderly", "pop_under5", "pct_elderly", "pct_under5",
            "pop_density_km2"]
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(dest, index=False)
    print(f"\nSaved {dest} ({len(out)} rows)")

    if a.upload:
        cmd = ["aws", "s3", "cp", str(dest), S3_PREFIX + dest.name, "--profile", PROFILE, "--region", AWS_REGION]
        try:
            subprocess.run(cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as err:
            print("Upload problem:", err, "\n  run: " + " ".join(cmd))
    print("\nDone.")


if __name__ == "__main__":
    main()
