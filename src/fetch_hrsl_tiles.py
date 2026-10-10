#!/usr/bin/env python
"""
Download only the HRSL tiles that cover the Chennai grid, using the AWS CLI (no GDAL network access).

Why: GDAL cannot open the remote HRSL VRT on this machine, but `aws s3` works. Each HRSL layer is a small
VRT index that points at 10x10 degree tiles. This script
  1. downloads the VRT for each layer (about 110 KB)
  2. works out which tile(s) cover the grid's bounding box
  3. downloads just those tiles, keeping the same folder layout under cache/hrsl/
so that hex_population_hrsl.py can read them from disk with  --base cache\\hrsl

Put this file in <project>/src/ and run from the project root with the venv Python:

    .venv\\Scripts\\python.exe src\\fetch_hrsl_tiles.py --dry-run   # shows tile names and sizes, downloads no tiles
    .venv\\Scripts\\python.exe src\\fetch_hrsl_tiles.py             # downloads them
    .venv\\Scripts\\python.exe src\\hex_population_hrsl.py --base cache\\hrsl

Needs: geopandas (for the grid bounds) and the AWS CLI on PATH.
"""
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "s3://dataforgood-fb-data/hrsl-cogs"
DEST = ROOT / "cache" / "hrsl"
GRID = ROOT / "data" / "processed" / "chennai_h3_res9.geojson"
FOLDERS = ["hrsl_general", "hrsl_elderly_60_plus", "hrsl_children_under_five"]


def aws(*args, capture=False):
    cmd = ["aws", "s3", *args, "--no-sign-request"]
    return subprocess.run(cmd, check=True, capture_output=capture, text=True)


def tiles_for_bbox(vrt_path, bbox):
    """Names (relative to the VRT) of the source tiles whose footprint intersects bbox (w, s, e, n)."""
    w, s, e, n = bbox
    root = ET.parse(vrt_path).getroot()
    gt = [float(v) for v in root.find("GeoTransform").text.split(",")]
    xa, xb = (w - gt[0]) / gt[1], (e - gt[0]) / gt[1]
    ya, yb = (n - gt[3]) / gt[5], (s - gt[3]) / gt[5]
    x0, x1, y0, y1 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
    found = set()
    for el in root.iter():
        if not el.tag.endswith("Source"):          # SimpleSource, ComplexSource
            continue
        name, dst = el.find("SourceFilename"), el.find("DstRect")
        if name is None or dst is None or not (name.text or "").strip():
            continue
        xo, yo = float(dst.get("xOff")), float(dst.get("yOff"))
        xs, ys = float(dst.get("xSize")), float(dst.get("ySize"))
        if xo < x1 and xo + xs > x0 and yo < y1 and yo + ys > y0:
            found.add(name.text.strip())
    return sorted(found)


def remote_size(key):
    out = aws("ls", f"{BUCKET}/{key}", capture=True).stdout.strip().splitlines()
    for line in out:
        parts = line.split()
        if len(parts) >= 4 and parts[-1] == key.rsplit("/", 1)[-1]:
            return int(parts[2])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--folders", nargs="*", default=FOLDERS)
    ap.add_argument("--dry-run", action="store_true", help="list tiles and sizes, do not download tiles")
    a = ap.parse_args()

    import geopandas as gpd
    bbox = tuple(gpd.read_file(a.grid).total_bounds)
    pad = 0.01
    bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    dest = Path(a.dest)
    print(f"Grid bounding box (W,S,E,N): {tuple(round(v, 4) for v in bbox)}")

    plan, total = [], 0
    for folder in a.folders:
        vrt = dest / folder / f"{folder}-latest.vrt"
        vrt.parent.mkdir(parents=True, exist_ok=True)
        try:
            aws("cp", f"{BUCKET}/{folder}/{folder}-latest.vrt", str(vrt))
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            sys.exit(f"Could not download the VRT for {folder}: {err}")
        tiles = tiles_for_bbox(vrt, bbox)
        if not tiles:
            sys.exit(f"{folder}: no tile in the VRT covers the grid - the layer may not include this area.")
        for t in tiles:
            size = remote_size(f"{folder}/{t}")
            total += size or 0
            print(f"  {folder}: {t}  ({(size or 0) / 1e6:,.0f} MB)")
            plan.append((folder, t, size))
    print(f"\nTotal to download: {total / 1e6:,.0f} MB in {len(plan)} file(s)")
    if a.dry_run:
        print("Dry run - nothing downloaded. Run again without --dry-run to fetch the tiles.")
        return

    for folder, t, size in plan:
        target = dest / folder / t
        if target.exists() and size and target.stat().st_size == size:
            print(f"  have {folder}/{t}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {folder}/{t} ...", flush=True)
        aws("cp", f"{BUCKET}/{folder}/{t}", str(target))
    print("\nDone. Next:  .venv\\Scripts\\python.exe src\\hex_population_hrsl.py --base cache\\hrsl")


if __name__ == "__main__":
    main()
