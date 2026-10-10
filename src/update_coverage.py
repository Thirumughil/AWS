#!/usr/bin/env python
"""
Update cooling access and coverage calculations for Chennai:
1. Read resolved status from data/processed/chennai_unreachable_resolved.csv.
2. For those 198 hexes, update their access result with the resolved status:
   "within 800 m", "outside 800 m", or "unknown: no main-network node within 400 m".
3. Recompute vulnerable-population coverage (pop_vulnerable) across the three states:
   within 800 m / outside 800 m / unknown.
4. Print old and new coverage, hex counts by state, and High/Very High hexes outside or unknown.
5. Save the updated access table with a new column access_status to a new file.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ACCESS_FILE = PROCESSED / "chennai_cooling_access.csv"
RESOLVED_FILE = PROCESSED / "chennai_unreachable_resolved.csv"
RISK_FILE = PROCESSED / "chennai_hex_risk.csv"
OUT_FILE = PROCESSED / "chennai_cooling_access_resolved.csv"


def update_coverage():
    # 1. Load data
    cool = pd.read_csv(ACCESS_FILE)
    res = pd.read_csv(RESOLVED_FILE)
    risk = pd.read_csv(RISK_FILE)

    total_hexes = len(cool)
    total_pop_vuln = cool["pop_vulnerable"].sum()

    # --- Old coverage calculation ---
    unreach_ids = set(res["h3_id"])
    cool["old_status"] = np.where(
        cool["h3_id"].isin(unreach_ids),
        "unreachable",
        np.where(cool["covered_800m_walk"], "within 800 m", "outside 800 m")
    )

    old_covered_pop = cool.loc[cool["old_status"] == "within 800 m", "pop_vulnerable"].sum()
    old_outside_pop = cool.loc[cool["old_status"] == "outside 800 m", "pop_vulnerable"].sum()
    old_unreach_pop = cool.loc[cool["old_status"] == "unreachable", "pop_vulnerable"].sum()

    # 2. Update status for the 198 hexes
    # Base existing result for other hexes
    updated = cool.copy()
    updated["access_status"] = np.where(updated["covered_800m_walk"], "within 800 m", "outside 800 m")

    res_status_map = dict(zip(res["h3_id"], res["status"]))
    res_walk_map = dict(zip(res["h3_id"], res["walk_to_nearest_candidate_m"]))
    res_snap_map = dict(zip(res["h3_id"], res["main_snap_m"]))

    # Apply resolved status & walk / snap updates
    for hid in res["h3_id"]:
        st = res_status_map[hid]
        walk = res_walk_map[hid]
        snap = res_snap_map[hid]

        mask = updated["h3_id"] == hid
        updated.loc[mask, "access_status"] = st
        if pd.notna(snap):
            updated.loc[mask, "hex_snap_distance_m"] = snap
        if pd.notna(walk):
            updated.loc[mask, "nearest_cooling_walk_m"] = walk

        if st == "within 800 m":
            updated.loc[mask, "covered_800m_walk"] = True
            updated.loc[mask, "vulnerable_pop_covered"] = updated.loc[mask, "pop_vulnerable"]
        else:
            updated.loc[mask, "covered_800m_walk"] = False
            updated.loc[mask, "vulnerable_pop_covered"] = 0.0

    # Clean up temporary column from output
    if "old_status" in updated.columns:
        updated = updated.drop(columns=["old_status"])

    # 3. Three states for coverage summary
    # Map to simplified state labels: "within 800 m", "outside 800 m", "unknown"
    updated["state_3way"] = np.where(
        updated["access_status"].str.startswith("unknown"),
        "unknown",
        updated["access_status"]
    )

    new_within_mask = updated["state_3way"] == "within 800 m"
    new_outside_mask = updated["state_3way"] == "outside 800 m"
    new_unknown_mask = updated["state_3way"] == "unknown"

    new_within_pop = updated.loc[new_within_mask, "pop_vulnerable"].sum()
    new_outside_pop = updated.loc[new_outside_mask, "pop_vulnerable"].sum()
    new_unknown_pop = updated.loc[new_unknown_mask, "pop_vulnerable"].sum()

    # 4. Print results
    print("=" * 80)
    print("CHENNAI COOLING ACCESS COVERAGE CALCULATION UPDATE")
    print("=" * 80)
    print(f"Total city hexagons: {total_hexes:,}")
    print(f"Total vulnerable population (pop_vulnerable): {total_pop_vuln:,.1f}\n")

    print("-" * 80)
    print("1. COVERAGE SUMMARY: OLD VS NEW")
    print("-" * 80)
    print(f"{'State':<16} | {'Old Hexes':<10} | {'Old Pop Vuln':<14} | {'Old Share':<10} | {'New Hexes':<10} | {'New Pop Vuln':<14} | {'New Share':<10}")
    print("-" * 80)
    print(f"{'within 800 m':<16} | {(cool['old_status']=='within 800 m').sum():<10} | {old_covered_pop:<14,.1f} | {old_covered_pop/total_pop_vuln*100:>8.2f}% | {new_within_mask.sum():<10} | {new_within_pop:<14,.1f} | {new_within_pop/total_pop_vuln*100:>8.2f}%")
    print(f"{'outside 800 m':<16} | {(cool['old_status']=='outside 800 m').sum():<10} | {old_outside_pop:<14,.1f} | {old_outside_pop/total_pop_vuln*100:>8.2f}% | {new_outside_mask.sum():<10} | {new_outside_pop:<14,.1f} | {new_outside_pop/total_pop_vuln*100:>8.2f}%")
    print(f"{'unknown':<16} | {'-':<10} | {'-':<14} | {'-':>9} | {new_unknown_mask.sum():<10} | {new_unknown_pop:<14,.1f} | {new_unknown_pop/total_pop_vuln*100:>8.2f}%")
    print(f"{'unreachable':<16} | {(cool['old_status']=='unreachable').sum():<10} | {old_unreach_pop:<14,.1f} | {old_unreach_pop/total_pop_vuln*100:>8.2f}% | {'-':<10} | {'-':<14} | {'-':>9}")
    print("-" * 80)
    print(f"{'Total':<16} | {total_hexes:<10} | {total_pop_vuln:<14,.1f} | {'100.00%':>9} | {total_hexes:<10} | {total_pop_vuln:<14,.1f} | {'100.00%':>9}")
    print("\n* Note: 'unknown' is never counted as covered. Only 'within 800 m' is counted as covered.")
    print(f"  Vulnerable population coverage increased from {old_covered_pop/total_pop_vuln*100:.2f}% to {new_within_pop/total_pop_vuln*100:.2f}% (+{(new_within_pop-old_covered_pop)/total_pop_vuln*100:.2f} percentage points, +{new_within_pop-old_covered_pop:,.1f} people).")
    print(f"  Hexagon coverage increased from {(cool['old_status']=='within 800 m').sum()/total_hexes*100:.2f}% to {new_within_mask.sum()/total_hexes*100:.2f}% (+{(new_within_mask.sum() - (cool['old_status']=='within 800 m').sum())/total_hexes*100:.2f} percentage points, +{new_within_mask.sum() - (cool['old_status']=='within 800 m').sum()} hexes).\n")

    print("-" * 80)
    print("2. BREAKDOWN OF THE 198 PREVIOUSLY UNREACHABLE HEXES")
    print("-" * 80)
    res_summary = res.groupby("status").agg(
        hex_count=("h3_id", "size"),
        vulnerable_population=("pop_vulnerable", "sum")
    ).reset_index()
    res_summary["hex_share"] = (res_summary["hex_count"] / len(res) * 100).round(2).astype(str) + "%"
    res_summary["pop_share_of_city"] = (res_summary["vulnerable_population"] / total_pop_vuln * 100).round(2).astype(str) + "%"
    print(res_summary.to_string(index=False))
    print()

    # 4. High / Very High hexes outside or unknown
    # A. Among the 198 resolved hexes
    print("-" * 80)
    print("3. HIGH / VERY HIGH RISK HEXES THAT ARE OUTSIDE OR UNKNOWN")
    print("-" * 80)
    res_hi = res[
        res["risk_category"].isin(["High", "Very High"]) &
        res["status"].isin(["outside 800 m", "unknown: no main-network node within 400 m"])
    ].sort_values("risk_score", ascending=False)

    print("A. Among the 198 previously unreachable resolved hexes:")
    if len(res_hi) == 0:
        print("   None found.")
    else:
        print(res_hi[["h3_id", "risk_score", "risk_category", "pop_vulnerable", "main_snap_m", "walk_to_nearest_candidate_m", "status"]].to_string(index=False))

    # B. Across the whole city
    city_risk = updated.merge(risk[["h3_id", "risk_score", "risk_category"]], on="h3_id", how="left")
    city_hi = city_risk[
        city_risk["risk_category"].isin(["High", "Very High"]) &
        city_risk["state_3way"].isin(["outside 800 m", "unknown"])
    ].sort_values("risk_score", ascending=False)

    print("\nB. Across the entire city (all 21 High-risk hexes):")
    cols_to_show = ["h3_id", "risk_score", "risk_category", "pop_vulnerable", "nearest_cooling_walk_m", "access_status"]
    print(city_hi[cols_to_show].to_string(index=False))
    print()

    # 5. Save the updated access table as a new file
    # Drop temporary calculation column before saving
    save_df = updated.drop(columns=["state_3way"])
    save_df.to_csv(OUT_FILE, index=False)
    print("-" * 80)
    print(f"5. SAVED UPDATED ACCESS TABLE: {OUT_FILE.name}")
    print("-" * 80)
    print(f"Destination: {OUT_FILE}")
    print(f"Shape: {save_df.shape[0]} rows, {save_df.shape[1]} columns")
    print(f"Columns: {list(save_df.columns)}")
    print(f"Original file intact: {ACCESS_FILE} (not overwritten)")
    print("=" * 80)


if __name__ == "__main__":
    update_coverage()
