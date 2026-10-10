#!/usr/bin/env python
"""
T20: building-level drill-down for the top-risk hexes.

    python src/building_drilldown.py                 # top 60 hexes by risk (all High / Very High are always included)
    python src/building_drilldown.py --top 100
    python src/building_drilldown.py --upload        # also copy the outputs to S3

Standard library only (no pandas, numpy, h3 or requests), so it runs even where those compiled packages are blocked.

For each selected hex it downloads the OpenStreetMap buildings and facilities inside the hex (one small Overpass request per
5 hexes, cached in cache/osm/), then lists:
  * critical facilities: hospitals, elder-care homes, kindergartens, clinics, schools, community centres, libraries,
    ranked by  priority = hex risk score x facility weight  (weights below are an assumption: change WEIGHT and re-run),
  * other large roofs (footprint >= --min-roof m2), as candidates for cool roofs or shading.
Each row carries roof area (the building footprint), height where OSM has it, the hex risk, the vulnerable population of the
hex and whether the hex has a cooling site within an 800 m walk.

Reads : data/processed/chennai_hex_risk.csv (h3_id, risk_score, risk_category, pop_vulnerable)
        data/processed/chennai_h3_res9.geojson (hex outlines)
        data/processed/chennai_cooling_access_resolved.csv (h3_id, access_status, nearest_cooling_walk_m)   [optional]
Writes: data/processed/chennai_building_drilldown.csv   one row per building / facility
        data/processed/chennai_building_drilldown.json  the same rows grouped by hex, for the /buildings API

Limits to state in the README
  * Height comes from the OSM 'height' tag, or 'building:levels' x 3.2 m; most Chennai buildings have neither, so the
    height column is often empty (height_source says which). Roof area is the footprint, not a measured roof.
  * Facility coverage is whatever OSM volunteers mapped: unmapped hospitals or schools will not appear.
  * Areas use a local flat-earth projection (error well under 1% for a building).
"""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
CACHE = ROOT / "cache" / "osm"

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
HEADERS = {"User-Agent": "ClimateGuard-AI-hackathon/0.1 (student project)"}
TRIES = 2
PAUSE_S = 6
BATCH = 5                        # hexes per Overpass request (a failing batch is split in halves)
LEVEL_M = 3.2                    # metres per floor when only building:levels is known
PROFILE = "climateguard"
S3_PREFIX = "s3://urban-heat-risk-sai-001/processed/"

# Assumption (confirm with the team): how much a facility type matters for people at heat risk. 0-1.
WEIGHT = {"hospital": 1.0, "elder_care": 1.0, "kindergarten": 0.9, "clinic": 0.8,
          "school": 0.7, "community_centre": 0.5, "library": 0.4}
AMENITIES = "hospital|clinic|doctors|school|college|kindergarten|childcare|nursing_home|social_facility|community_centre|library"
COLUMNS = ["h3_id", "hex_rank", "hex_risk_score", "hex_risk_category", "hex_pop_vulnerable", "access_status",
           "nearest_cooling_walk_m", "osm_id", "name", "facility_type", "building_tag", "footprint_m2", "height_m",
           "height_source", "lat", "lon", "is_critical", "priority_score", "rank_city", "rank_in_hex"]


# ----------------------------------------------------------------- Overpass (standard library)
def overpass(query):
    last = None
    body = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(TRIES):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            req = urllib.request.Request(url, data=body, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=300) as r:
                j = json.loads(r.read().decode("utf8"))
            remark = j.get("remark", "")
            if "runtime error" in remark or "timed out" in remark:
                raise RuntimeError(remark[:120])
            return j
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        wait = 15 * (attempt + 1)
        print(f"    retry {attempt + 1}/{TRIES} in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"Overpass failed after {TRIES} tries: {last}")


def build_query(hexes, rings):
    parts = []
    for h in hexes:
        poly = " ".join(f"{la:.6f} {lo:.6f}" for lo, la in rings[h][:-1])
        parts.append(f'way["building"](poly:"{poly}");')
        parts.append(f'node["amenity"~"^({AMENITIES})$"](poly:"{poly}");')
        parts.append(f'way["amenity"~"^({AMENITIES})$"](poly:"{poly}");')
    return "[out:json][timeout:120];(" + "".join(parts) + ");out geom tags;"


def cache_file(query):
    return CACHE / f"drilldown_{hashlib.md5(query.encode()).hexdigest()[:12]}.json"


def fetch_batch(hexes, rings, refetch, failed):
    """Download one batch; if the server fails, split it in halves, and give up on a single hex only after all tries."""
    q = build_query(hexes, rings)
    f = cache_file(q)
    if f.exists() and not refetch:
        return json.loads(f.read_text())["elements"], True
    try:
        j = overpass(q)
    except RuntimeError as err:
        if len(hexes) > 1:
            mid = len(hexes) // 2
            print(f"    splitting {len(hexes)} hexes into {mid} + {len(hexes) - mid}")
            a, _ = fetch_batch(hexes[:mid], rings, refetch, failed)
            time.sleep(PAUSE_S)
            b, _ = fetch_batch(hexes[mid:], rings, refetch, failed)
            return a + b, False
        print(f"    giving up on hex {hexes[0]}: {err}")
        failed.append(hexes[0])
        return [], False
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"elements": j.get("elements", [])}))
    return j.get("elements", []), False


# ----------------------------------------------------------------- geometry
def area_m2(lon, lat):
    lon0, lat0 = sum(lon) / len(lon), sum(lat) / len(lat)
    kx = 111320.0 * math.cos(math.radians(lat0))
    x = [(v - lon0) * kx for v in lon]
    y = [(v - lat0) * 110574.0 for v in lat]
    s = sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(x) - 1))
    return abs(s) / 2.0


def inside(px, py, lons, lats):
    """Ray casting in degrees (fine for one building or one hex)."""
    c = False
    j = len(lons) - 1
    for i in range(len(lons)):
        if (lats[i] > py) != (lats[j] > py) and px < (lons[j] - lons[i]) * (py - lats[i]) / (lats[j] - lats[i]) + lons[i]:
            c = not c
        j = i
    return c


# ----------------------------------------------------------------- tags
def facility_type(tags):
    a = tags.get("amenity", "")
    h = tags.get("healthcare", "")
    b = tags.get("building", "")
    sf = (tags.get("social_facility:for", "") + " " + tags.get("social_facility", "")).lower()
    if a == "hospital" or b == "hospital" or h == "hospital":
        return "hospital"
    if a == "nursing_home" or (a == "social_facility" and re.search(r"senior|elder|aged|old|group_home", sf)):
        return "elder_care"
    if a in ("kindergarten", "childcare") or b == "kindergarten":
        return "kindergarten"
    if a in ("clinic", "doctors"):
        return "clinic"
    if a in ("school", "college") or b in ("school", "college"):
        return "school"
    if a == "community_centre":
        return "community_centre"
    if a == "library":
        return "library"
    return ""


def height_m(tags):
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", tags.get("height", ""))
    if m:
        return float(m.group(1)), "osm_height"
    lv = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", tags.get("building:levels", ""))
    if lv:
        return round(float(lv.group(1)) * LEVEL_M, 1), "levels_x3.2"
    return None, "unknown"


def better(a, b):
    return a if WEIGHT.get(a, -1) >= WEIGHT.get(b, -1) else b


def assemble(elements):
    """OSM elements -> one record per building plus point-only facilities."""
    blds, pts, skipped = {}, [], 0
    for el in elements:
        tags = el.get("tags", {})
        t = el.get("type")
        if t == "node":
            if facility_type(tags):
                pts.append((f"node/{el['id']}", tags, el["lat"], el["lon"]))
            continue
        if t != "way":
            skipped += 1                         # multipolygon relations are skipped
            continue
        g = el.get("geometry")
        if not g or len(g) < 4 or g[0] != g[-1]:
            skipped += 1
            continue
        lon = [p["lon"] for p in g]
        lat = [p["lat"] for p in g]
        cx, cy = sum(lon[:-1]) / (len(lon) - 1), sum(lat[:-1]) / (len(lat) - 1)
        if "building" in tags:
            hgt, src = height_m(tags)
            blds[f"way/{el['id']}"] = {
                "osm_id": f"way/{el['id']}", "name": tags.get("name", ""), "building_tag": tags.get("building", ""),
                "facility_type": facility_type(tags), "footprint_m2": round(area_m2(lon, lat), 1),
                "height_m": hgt, "height_source": src, "lat": cy, "lon": cx, "_lon": lon, "_lat": lat}
        elif facility_type(tags):
            pts.append((f"way/{el['id']}", tags, cy, cx))
    loose = []
    for fid, tags, la, lo in pts:
        ft = facility_type(tags)
        hit = None
        for b in blds.values():
            if min(b["_lon"]) <= lo <= max(b["_lon"]) and min(b["_lat"]) <= la <= max(b["_lat"]) \
                    and inside(lo, la, b["_lon"], b["_lat"]):
                hit = b
                break
        if hit is not None:
            hit["facility_type"] = better(ft, hit["facility_type"]) if hit["facility_type"] else ft
            hit["name"] = hit["name"] or tags.get("name", "")
        else:
            loose.append({"osm_id": fid, "name": tags.get("name", ""), "building_tag": "", "facility_type": ft,
                          "footprint_m2": None, "height_m": None, "height_source": "unknown", "lat": la, "lon": lo})
    rows = [{k: v for k, v in b.items() if not k.startswith("_")} for b in blds.values()] + loose
    return rows, skipped


# ----------------------------------------------------------------- inputs
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def feature_id(f):
    p = f.get("properties") or {}
    for k in ("h3_id", "h3", "id", "index"):
        if k in p:
            return str(p[k])
    if "id" in f:
        return str(f["id"])
    for v in p.values():
        if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{15}", v):
            return v
    return None


def load_rings(path, wanted):
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    rings = {}
    for f in gj["features"]:
        hid = feature_id(f)
        if hid in wanted:
            geom = f["geometry"]
            ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
            rings[hid] = [(float(x), float(y)) for x, y in ring]       # (lon, lat), closed
    return rings


def hex_of(lon, lat, rings):
    for h, r in rings.items():
        xs = [p[0] for p in r]
        ys = [p[1] for p in r]
        if min(xs) <= lon <= max(xs) and min(ys) <= lat <= max(ys) and inside(lon, lat, xs, ys):
            return h
    return None


def build(sel, rings, batches, min_roof):
    recs, skipped, seen = [], 0, set()
    for els in batches:
        r, s = assemble(els)
        skipped += s
        for x in r:
            if x["osm_id"] in seen:
                continue
            seen.add(x["osm_id"])
            h = hex_of(x["lon"], x["lat"], rings)
            if h is None:
                continue                      # centroid lies in a hex we did not select
            s_ = sel[h]
            x = dict(x, h3_id=h, hex_rank=s_["rank"], hex_risk_score=s_["risk"], hex_risk_category=s_["cat"],
                     hex_pop_vulnerable=s_["pop"], access_status=s_.get("access"),
                     nearest_cooling_walk_m=s_.get("walk"))
            x["is_critical"] = bool(x["facility_type"])
            x["priority_score"] = round(s_["risk"] * WEIGHT.get(x["facility_type"], 0), 2) if x["is_critical"] else None
            if x["is_critical"] or (x["footprint_m2"] or 0) >= min_roof:
                recs.append(x)
    crit = sorted([x for x in recs if x["is_critical"]], key=lambda x: -x["priority_score"])
    for i, x in enumerate(crit, 1):
        x["rank_city"] = i
    roofs = [x for x in recs if not x["is_critical"]]
    for x in roofs:
        x["rank_city"] = None
    out = crit + sorted(roofs, key=lambda x: -(x["footprint_m2"] or 0))
    count = {}
    for x in out:                                 # critical first, then roofs: rank within each hex
        count[x["h3_id"]] = count.get(x["h3_id"], 0) + 1
        x["rank_in_hex"] = count[x["h3_id"]]
    out.sort(key=lambda x: (x["hex_rank"], x["rank_in_hex"]))
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60, help="number of highest-risk hexes (High / Very High are always added)")
    ap.add_argument("--min-roof", type=float, default=300.0, help="smallest non-facility roof to list, m2")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    for f in (P / "chennai_hex_risk.csv", P / "chennai_h3_res9.geojson"):
        if not f.exists():
            sys.exit(f"Missing: {f}")
    risk = read_csv(P / "chennai_hex_risk.csv")
    for c in ("h3_id", "risk_score", "risk_category", "pop_vulnerable"):
        if c not in risk[0]:
            sys.exit(f"chennai_hex_risk.csv lacks column {c}")
    risk.sort(key=lambda r: -(num(r["risk_score"]) or 0))
    keep = [r for i, r in enumerate(risk) if i < args.top or r["risk_category"] in ("High", "Very High")]
    sel = {r["h3_id"]: {"rank": i + 1, "risk": round(num(r["risk_score"]) or 0, 2), "cat": r["risk_category"],
                        "pop": round(num(r["pop_vulnerable"]) or 0, 1)} for i, r in enumerate(keep)}
    accf = P / "chennai_cooling_access_resolved.csv"
    if accf.exists():
        for r in read_csv(accf):
            if r["h3_id"] in sel:
                sel[r["h3_id"]]["access"] = r.get("access_status")
                sel[r["h3_id"]]["walk"] = num(r.get("nearest_cooling_walk_m"))
    rings = load_rings(P / "chennai_h3_res9.geojson", set(sel))
    missing = set(sel) - set(rings)
    if missing:
        sys.exit(f"{len(missing)} selected hexes were not found in the geojson (id property?): {sorted(missing)[:3]}")
    ids = list(sel)
    print(f"{len(ids)} hexes selected (risk {min(s['risk'] for s in sel.values()):.1f} to {max(s['risk'] for s in sel.values()):.1f})")

    groups = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    got, failed = [], []
    for i, g in enumerate(groups, 1):
        print(f"  Overpass batch {i}/{len(groups)} ({len(g)} hexes)")
        els, cached = fetch_batch(g, rings, args.refetch, failed)
        got.append(els)
        if not cached and i < len(groups):
            time.sleep(PAUSE_S)
    if failed:
        print(f"\nWARNING: {len(failed)} hexes could not be downloaded and are missing from the result: {failed}\n"
              "  Run the script again later; hexes already downloaded are cached and will not be fetched twice.")
    out, skipped = build(sel, rings, got, args.min_roof)
    if not out:
        sys.exit("No buildings found.")
    crit = [x for x in out if x["is_critical"]]
    kinds = {}
    for x in crit:
        kinds[x["facility_type"]] = kinds.get(x["facility_type"], 0) + 1
    print(f"\n{len(out):,} rows: {len(crit)} critical facilities + {len(out) - len(crit)} large roofs "
          f"(>= {args.min_roof:.0f} m2); {skipped} multipolygon/unclosed OSM shapes skipped")
    print("Critical facilities by type:", kinds)
    print(f"Height known for {sum(x['height_m'] is not None for x in out) / len(out):.0%} of rows; "
          f"roof area known for {sum(x['footprint_m2'] is not None for x in out) / len(out):.0%}")
    print("\nTop 15 critical facilities (priority = hex risk x facility weight):")
    print(f"{'rank':>4}  {'type':<17}{'name':<28}{'hex':<17}{'risk':>6}{'prio':>7}{'roof m2':>9}{'h m':>6}  access")
    for x in sorted(crit, key=lambda x: float(x.get("priority_score") or 0), reverse=True)[:15]:
        print(f"{x['rank_city']:>4}  {x['facility_type']:<17}{(x['name'] or '-')[:26]:<28}{x['h3_id']:<17}"
              f"{x['hex_risk_score']:>6.1f}{x['priority_score']:>7.1f}"
              f"{(x['footprint_m2'] if x['footprint_m2'] is not None else float('nan')):>9.0f}"
              f"{(x['height_m'] if x['height_m'] is not None else float('nan')):>6.1f}  {x['access_status'] or ''}")

    rows = [{c: (None if x.get(c) is None else x.get(c)) for c in COLUMNS} for x in out]
    for r in rows:
        r["lat"], r["lon"] = round(r["lat"], 6), round(r["lon"], 6)
    dest = P / "chennai_building_drilldown.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    by_hex = {}
    for r in rows:
        by_hex.setdefault(r["h3_id"], []).append({k: v for k, v in r.items() if k != "h3_id"})
    (P / "chennai_building_drilldown.json").write_text(json.dumps(by_hex))
    print(f"\nSaved {dest} and chennai_building_drilldown.json ({len(by_hex)} hexes)")
    if args.upload:
        for name in ("chennai_building_drilldown.csv", "chennai_building_drilldown.json"):
            cmd = ["aws", "s3", "cp", str(P / name), S3_PREFIX + name, "--profile", PROFILE]
            try:
                subprocess.run(cmd, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as err:
                print("Upload problem:", err, "\n  run: " + " ".join(cmd))
    print("\nDone.")


if __name__ == "__main__":
    main()
