#!/usr/bin/env python
"""
T10 (part B): population, 65+ and under-5 per hex for the Chennai grid (WorldPop).
Run from the project root:

    python src/hex_population.py             # read the Chennai window from WorldPop, build the table
    python src/hex_population.py --upload    # also copy the output to S3
    python src/hex_population.py --refetch   # ignore the cache and read again

Output (data/processed): chennai_hex_population.csv
    h3_id, pop_total, pop_65plus, pop_under5, pct_65plus, pct_under5, pop_density_km2

Data: WorldPop Global 2015-2030 age-sex structures, release R2025A, year 2025, 100 m, constrained
(India). Licence CC BY 4.0. Each country file is ~740 MB. If the server allows partial reads only the
Chennai window is fetched; if not, each file is downloaded whole, cut to Chennai, then deleted.
Each 100 m pixel is assigned to the hex that contains its centre, so a hex total is a sum of whole pixels.
Needs: numpy pandas geopandas rasterio requests h3
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
import rasterio
import requests
from rasterio.windows import Window, from_bounds

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "processed" / "chennai_h3_res9.geojson"
PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "cache" / "worldpop"

H3_RES = 9
_PATH = "GIS/AgeSex_structures/Global_2015_2030/R2025A/2025/IND/v1/100m/constrained/ind_{code}_2025_CN_100m_R2025A_v1.tif"
URLS = ["https://data.worldpop.org/" + _PATH,
        "https://worldpop-public-data.soton.ac.uk/" + _PATH]      # mirror, tried second
# file codes: t_XX = both sexes, age group starting at XX (00 = under 1, 01 = 1-4, 65 = 65-69 ... 90 = 90+)
TOTAL = ["T_F", "T_M"]                                  # all ages, female + male
UNDER5 = ["t_00", "t_01"]
OVER65 = ["t_65", "t_70", "t_75", "t_80", "t_85", "t_90"]
PAD_DEG = 0.01                                          # margin around the grid so edge hexes are complete

PROFILE = os.environ.get("AWS_PROFILE", "climateguard")
S3_PREFIX = "s3://urban-heat-risk-sai-001/processed/"
V4 = hasattr(h3, "latlng_to_cell")


def cell(lat, lon):
    return h3.latlng_to_cell(lat, lon, H3_RES) if V4 else h3.geo_to_h3(lat, lon, H3_RES)


REMOTE_OK = True          # set to False after the first "range requests not supported" error


def read_local_or_remote(path, bounds):
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
               GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3")
    with rasterio.Env(**env):
        with rasterio.open(path) as src:
            w0 = from_bounds(*bounds, transform=src.transform)
            w = Window(int(np.floor(w0.col_off)), int(np.floor(w0.row_off)),
                       int(np.ceil(w0.width)) + 1, int(np.ceil(w0.height)) + 1)
            w = w.intersection(Window(0, 0, src.width, src.height))
            arr = src.read(1, window=w).astype("float64")
            nodata = src.nodata
            tr = src.window_transform(w)
    return arr, nodata, tr


def download(code, dest):
    """Stream one whole country file to disk (used only when the server cannot do partial reads)."""
    last = None
    for url in URLS:
        u = url.format(code=code)
        for attempt in range(3):
            try:
                tmp = dest.with_suffix(".part")
                with requests.get(u, stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    size = int(r.headers.get("content-length", 0))
                    got, mark = 0, 0
                    t_dl = time.time()
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(4 * 1024 * 1024):
                            fh.write(chunk)
                            got += len(chunk)
                            if got - mark >= 50 * 1024 * 1024 or (size and got >= size):
                                mark = got
                                dt = time.time() - t_dl
                                spd = (got / 1e6) / dt if dt > 0 else 0
                                pct = (got / size * 100) if size else 0
                                print(f"      {got / 1e6:,.0f} / {size / 1e6:,.0f} MB ({pct:.1f}%) @ {spd:.2f} MB/s", flush=True)
                if size and got < size:
                    raise IOError(f"download cut short: {got} of {size} bytes")
                tmp.replace(dest)
                return
            except Exception as err:
                last = f"{type(err).__name__}: {err}"
                print(f"      download problem ({last}), try {attempt + 1}/3", flush=True)
                time.sleep(10)
    raise RuntimeError(f"could not download {code}: {last}")


def read_window(code, bounds, refetch):
    """Read one WorldPop file inside `bounds` (west, south, east, north). Cached as .npy."""
    global REMOTE_OK
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{code}.npy"
    meta = CACHE / f"{code}_meta.txt"
    if f.exists() and meta.exists() and not refetch:
        vals = [float(v) for v in meta.read_text().split()]
        return np.load(f), vals
    t0 = time.time()
    got = None
    if REMOTE_OK:
        for url in URLS:
            try:
                got = read_local_or_remote(url.format(code=code), bounds)
                break
            except Exception as err:
                print(f"  {code}: partial read failed on {url.split('/')[2]} ({str(err)[:90]})", flush=True)
        if got is None:
            REMOTE_OK = False
            print("  The server does not support partial reads, so each whole file (~740 MB) is downloaded, "
                  "the Chennai window is cut out, then the big file is deleted.", flush=True)
    if got is None:
        raw = CACHE / "raw"
        raw.mkdir(exist_ok=True)
        big = raw / f"{code}.tif"
        print(f"  {code}: downloading the whole file ...", flush=True)
        download(code, big)
        got = read_local_or_remote(str(big), bounds)
        big.unlink()
    arr, nodata, tr = got
    if nodata is not None:
        arr[arr == nodata] = 0
    arr[~np.isfinite(arr)] = 0
    arr[arr < 0] = 0                                   # WorldPop uses large negatives for no data
    vals = [tr.a, tr.c, tr.e, tr.f]
    np.save(f, arr)
    meta.write_text(" ".join(repr(v) for v in vals))
    print(f"  {code}: {arr.shape[1]} x {arr.shape[0]} pixels, {arr.sum():,.0f} people, {time.time() - t0:.0f}s", flush=True)
    return arr, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    if not GRID.exists():
        sys.exit(f"Missing grid: {GRID}")
    g = gpd.read_file(GRID)
    grid = pd.DataFrame(g.drop(columns="geometry"))[["h3_id", "area_km2"]].reset_index(drop=True)
    ids = set(grid["h3_id"])
    w, s, e, n = g.total_bounds
    bounds = (w - PAD_DEG, s - PAD_DEG, e + PAD_DEG, n + PAD_DEG)
    codes = TOTAL + UNDER5 + OVER65
    print(f"{len(grid)} hexes | reading {len(codes)} WorldPop files, Chennai window only")

    arrays, ref = {}, None
    for code in codes:
        try:
            arr, tr = read_window(code, bounds, args.refetch)
        except Exception as err:
            sys.exit(f"\nCould not read {code}: {type(err).__name__}: {err}\n"
                     "Run the same command again (finished files are cached). If it keeps failing, "
                     "open the URL below in a browser to check it exists:\n  " + URLS[0].format(code=code))
        if ref is None:
            ref = (arr.shape, tr)
        elif arr.shape != ref[0] or not np.allclose(tr, ref[1]):
            sys.exit(f"{code} does not line up with the first file: {arr.shape} vs {ref[0]}")
        arrays[code] = arr

    shape, (a, c0, e_, f0) = ref
    total = arrays["T_F"] + arrays["T_M"]
    under5 = sum(arrays[k] for k in UNDER5)
    over65 = sum(arrays[k] for k in OVER65)

    # pixel centre -> hex
    rows, cols = np.nonzero(total > 0)
    lon = c0 + a * (cols + 0.5)
    lat = f0 + e_ * (rows + 0.5)
    hexid = np.array([cell(la, lo) for la, lo in zip(lat, lon)])
    px = pd.DataFrame({"h3_id": hexid, "pop_total": total[rows, cols],
                       "pop_under5": under5[rows, cols], "pop_65plus": over65[rows, cols]})
    px["in_city"] = px["h3_id"].isin(ids)
    captured = px[px["in_city"]]
    agg = captured.groupby("h3_id")[["pop_total", "pop_65plus", "pop_under5"]].sum().reset_index()
    out = grid.merge(agg, on="h3_id", how="left").fillna({"pop_total": 0, "pop_65plus": 0, "pop_under5": 0})
    ok = out["pop_total"] >= 1
    out["pct_65plus"] = np.where(ok, out["pop_65plus"] / out["pop_total"], np.nan)
    out["pct_under5"] = np.where(ok, out["pop_under5"] / out["pop_total"], np.nan)
    out["pop_density_km2"] = out["pop_total"] / out["area_km2"]
    for col in ("pop_total", "pop_65plus", "pop_under5", "pop_density_km2"):
        out[col] = out[col].round(1)
    out["pct_65plus"] = out["pct_65plus"].round(4)
    out["pct_under5"] = out["pct_under5"].round(4)

    # ---------------------------------------------------------- checks
    tot_in = out["pop_total"].sum()
    print(f"\nPopulation inside the hex grid: {tot_in:,.0f} "
          f"({tot_in / px['pop_total'].sum():.0%} of the people in the padded window; the rest is outside the city limits)")
    print(f"  65+: {out['pop_65plus'].sum():,.0f} ({out['pop_65plus'].sum() / tot_in:.1%}) | "
          f"under 5: {out['pop_under5'].sum():,.0f} ({out['pop_under5'].sum() / tot_in:.1%})")
    print(f"  hexes with no population: {int((out['pop_total'] < 1).sum())} | "
          f"people per hex: median {out['pop_total'].median():.0f}, max {out['pop_total'].max():.0f}")
    print(f"  density: median {out['pop_density_km2'].median():,.0f} per km2 | "
          f"90th percentile {out['pop_density_km2'].quantile(0.9):,.0f} | max {out['pop_density_km2'].max():,.0f}")
    big = out[ok & (out["pop_total"] >= 50)]
    print(f"  share 65+ across hexes with 50+ people: median {big['pct_65plus'].median():.1%}, "
          f"range {big['pct_65plus'].min():.1%} to {big['pct_65plus'].max():.1%}")

    ex = PROCESSED / "chennai_hex_exposure.csv"
    if ex.exists():
        m = out.merge(pd.read_csv(ex, usecols=["h3_id", "bldg_cover_frac"]), on="h3_id")
        print(f"  correlation of population with building cover: {m['pop_total'].corr(m['bldg_cover_frac']):.2f} "
              "(expect clearly positive)")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = ["h3_id", "pop_total", "pop_65plus", "pop_under5", "pct_65plus", "pct_under5", "pop_density_km2"]
    dest = PROCESSED / "chennai_hex_population.csv"
    out[cols].to_csv(dest, index=False)
    print(f"\nSaved {dest} ({len(out)} rows)")

    if args.upload:
        cmd = ["aws", "s3", "cp", str(dest), S3_PREFIX + dest.name, "--profile", PROFILE]
        try:
            subprocess.run(cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as err:
            print("Upload problem:", err, "\n  run: " + " ".join(cmd))
    print("\nDone.")


if __name__ == "__main__":
    main()