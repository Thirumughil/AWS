#!/usr/bin/env python
"""
Re-check the "unreachable" hexes by snapping to the MAIN walking network instead of to tiny road fragments.

    python src/snap_main_component.py

Why: 198 hex centroids were nearest to road fragments of 2-4 nodes (service-road stubs) that connect to nothing, so no
walking route could be computed. A fragment cannot be walked to or from, so the honest approach is:
  * keep only the largest connected part of the walking network (nearly all of it),
  * snap each hex centre and each candidate facility to the nearest node of THAT network,
  * add the straight-line snap distance at both ends to the network distance,
  * if the nearest main-network node is farther than MAX_SNAP_M, call the hex "unknown" instead of guessing.
The 800 m limit is unchanged. Nothing is connected artificially.

Reads : cache/walking_network/chennai_walk.graphml, data/processed/chennai_unreachable_component_audit.csv,
        chennai_hex_features.csv, chennai_facilities.csv, chennai_hex_risk.csv
Writes: data/processed/chennai_unreachable_resolved.csv
Needs : numpy pandas networkx scipy pyproj osmnx
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
GRAPH = ROOT / "cache" / "walking_network" / "chennai_walk.graphml"
CANDIDATE_TYPES = ["community_centre", "library", "school"]
WALK_LIMIT_M = 800
MAX_SNAP_M = 400          # farther than this from the main network = unknown, not covered


def resolve(Gp, hex_ids, hx, hy, fx, fy, walk_limit=WALK_LIMIT_M, max_snap=MAX_SNAP_M):
    """Gp: projected walking graph (edge attribute 'length' in metres). Coordinates in the same metric CRS."""
    comp = max(nx.strongly_connected_components(Gp), key=len)
    main = Gp.subgraph(comp)
    nodes = list(main.nodes)            # keep a plain list: node IDs may be tuples
    xy = np.array([(main.nodes[n]["x"], main.nodes[n]["y"]) for n in nodes], dtype=float)
    tree = cKDTree(xy)

    hd, hi = tree.query(np.c_[hx, hy])
    fd, fi = tree.query(np.c_[fx, fy])
    hex_node = [nodes[i] for i in hi]
    fac_node = [nodes[i] for i in fi]

    # shortest walk from each node to its nearest candidate facility (reverse graph = distance TO the facility)
    rev = main.reverse(copy=True)
    best = {}
    for node, snap in zip(fac_node, fd):
        if snap > max_snap or snap >= walk_limit:
            continue
        dist = nx.single_source_dijkstra_path_length(rev, node, cutoff=walk_limit - snap, weight="length")
        for n, d in dist.items():
            total = d + snap
            if total < best.get(n, np.inf):
                best[n] = total

    walk = np.array([best.get(n, np.inf) for n in hex_node]) + hd
    out = pd.DataFrame({"h3_id": hex_ids, "main_snap_node": [str(n) for n in hex_node], "main_snap_m": hd.round(1),
                        "walk_to_nearest_candidate_m": np.where(np.isfinite(walk), walk.round(0), np.nan)})
    status = np.where(hd > max_snap, "unknown: no main-network node within %d m" % max_snap,
                      np.where(walk <= walk_limit, "within %d m" % walk_limit, "outside %d m" % walk_limit))
    out["status"] = status
    return out, {"main_nodes": len(comp), "all_nodes": Gp.number_of_nodes(), "facilities_used": int((fd <= max_snap).sum())}


def main():
    import osmnx as ox
    from pyproj import Transformer

    for p in (GRAPH, P / "chennai_unreachable_component_audit.csv", P / "chennai_hex_features.csv",
              P / "chennai_facilities.csv", P / "chennai_hex_risk.csv"):
        if not p.exists():
            sys.exit(f"Missing: {p}")
    G = ox.io.load_graphml(filepath=str(GRAPH))
    Gp = ox.project_graph(G)
    to_m = Transformer.from_crs(4326, Gp.graph["crs"], always_xy=True)

    audit = pd.read_csv(P / "chennai_unreachable_component_audit.csv")[["h3_id"]]
    feats = pd.read_csv(P / "chennai_hex_features.csv")[["h3_id", "lon", "lat"]]
    hexes = audit.merge(feats, on="h3_id", how="left", validate="one_to_one")
    fac = pd.read_csv(P / "chennai_facilities.csv")
    fac["in_city"] = fac["in_city"].astype(str).str.strip().str.lower().isin(["true", "1"])
    fac = fac[fac["in_city"] & fac["type"].isin(CANDIDATE_TYPES)]

    hx, hy = to_m.transform(hexes["lon"].to_numpy(), hexes["lat"].to_numpy())
    fx, fy = to_m.transform(fac["lon"].to_numpy(), fac["lat"].to_numpy())
    out, info = resolve(Gp, hexes["h3_id"].to_numpy(), np.asarray(hx), np.asarray(hy), np.asarray(fx), np.asarray(fy))
    print(f"Main network: {info['main_nodes']:,} of {info['all_nodes']:,} nodes | "
          f"candidate facilities usable: {info['facilities_used']} of {len(fac)}")

    risk = pd.read_csv(P / "chennai_hex_risk.csv")[["h3_id", "risk_score", "risk_category", "pop_vulnerable"]]
    out = out.merge(risk, on="h3_id", how="left")
    out["status_short"] = out["status"].str.split(":").str[0].str.replace(r"\s\d+\s?m$", "", regex=True)
    print(f"\nThe {len(out)} previously 'unreachable' hexes, snapped to the main network:")
    print(f"  distance from hex centre to the main network: median {out['main_snap_m'].median():.0f} m, "
          f"90th percentile {out['main_snap_m'].quantile(0.9):.0f} m, max {out['main_snap_m'].max():.0f} m")
    summary = out.groupby("status").agg(hexes=("h3_id", "size"), vulnerable_people=("pop_vulnerable", "sum")).round(0)
    print(summary.to_string())
    hi = out[out["risk_category"].isin(["High", "Very High"])].sort_values("risk_score", ascending=False)
    print("\nHigh / Very High hexes among them:")
    print(hi[["h3_id", "risk_score", "pop_vulnerable", "main_snap_m", "walk_to_nearest_candidate_m", "status"]]
          .round(1).to_string(index=False))

    out.drop(columns=["status_short"]).to_csv(P / "chennai_unreachable_resolved.csv", index=False)
    print(f"\nSaved {P / 'chennai_unreachable_resolved.csv'}")
    print("Report these as three states: within 800 m / outside 800 m / unknown. Do not count 'unknown' as covered.")


if __name__ == "__main__":
    main()